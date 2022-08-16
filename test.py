#%% test
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)


torch.autograd.set_detect_anomaly(True)
from vae_modules import *
class NN_upconv_more(nn.Module):
    """This is recursive Wiener filter version, with Rb threshold of [1e-3, 1e2]
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M=3, K=3, im_size=100):
        super().__init__()
        self.dz = 32
        self.J, self.M = K, M
        self.mainnet = nn.Sequential(
            nn.Linear(18,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
        )
        self.hnet1 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,6)
        )
        self.hnet2 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,6)
        )
        self.hnet3 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,6)
        )

        self.encoder = nn.Sequential(
            Down(in_channels=1, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            Down(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=1),
            )
        self.fc1 = nn.Linear(25*25, 2*self.dz)
        self.decoder = nn.Sequential(
            nn.Linear(self.dz, 25*25),
            Reshape(-1, 1, 25, 25),
            Up_(in_channels=1, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            Up_(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=4),
            OutConv(in_channels=4, out_channels=1),
            ) 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        batch_size, _, N, F = x.shape
        z_all, v_all, h_all = [], [], []
        I = x.shape[0]
        x0 = x.permute(0,2,3,1)[...,None]
        Rx = x0 @ x.permute(0,2,3,1).conj()[...,None,:]
        Rx = Rx.mean(dim=(1,2)) # shape of [I,M,M]
        rx_inv = Rx.inverse()
        temp = self.mainnet(torch.stack((Rx.real, Rx.imag), dim=1).reshape(I,-1))
        H = self.hnet1(temp), self.hnet2(temp), self.hnet3(temp)
        b = x0
        for i in range(self.J):
            hhat = H[i][:,:3] + 1j*H[i][:,3:]
            hhat = hhat/hhat.detach()[:,0:1]
            h_all.append(hhat)
            w = rx_inv@hhat[...,None] / \
                    (hhat[:,None,:].conj()@rx_inv@hhat[...,None])
            shat = w.permute(0,2,1).conj()[:,None,None]@x0
            b = b - shat*hhat[:, None,None,:,None]
        
            "Encoder"
            xx = self.encoder(shat.squeeze()[:,None].abs())
            "Get latent variable"
            zz = self.fc1(xx.reshape(batch_size,-1))
            mu = zz[:,::2]
            logvar = zz[:,1::2]
            z = self.reparameterize(mu, logvar)
            z_all.append(z)
            
            "Decoder to get V"
            v = self.decoder(z).exp()
            v_all.append(threshold(v, floor=1e-3, ceiling=1e2)) # 1e-3 to 1e2
        Rb = (b@b.conj().permute(0,1,2,4,3)).mean(dim=(1,2)).squeeze()
        Hhat = torch.stack(h_all, 2) # shape:[I, M, K]
        vhat = torch.stack(v_all, 4).squeeze().to(torch.cfloat) # shape:[I, N, F, K]

        return vhat.diag_embed(), Hhat, Rb, mu, logvar

def loss_fun(x, Rs, Hhat, Rb, mu, logvar, beta=1e-3):
    x = x.permute(0,2,3,1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb
    Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
    rx = (x[...,None]@x[...,None,:].conj()).mean(dim=(1,2))
    rx_inv = rx.inverse()

    y = 0
    for i in range(3):
        hhat = Hhat[...,i]
        w = rx_inv@hhat[...,None] / \
                (hhat[:,None,:].conj()@rx_inv@hhat[...,None])
        y += (w.permute(0,2,1).conj()[:,None,None]@x[...,None]) * hhat[:,None,None,:,None]
    term3 = ((y.squeeze() - x).abs()**2).mean()
    try:
        ll = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze() 
    except:
        torch.save((x, Rx, Rs, Hhat, Rb), f'rid{rid}x_Rx_Rs_Hhat_Rb.pt')
        print('error happpened, data saved and stop')
        ll = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze()
    return -ll.sum().real, beta*kl + 0.1*term3

#%% load data
I = 3000 # how many samples
M, N, F, J = 3, 100, 100, 3
NF = N*F
eps = 5e-4
opts = {}
opts['batch_size'] = 64

xval, _ , hgt0 = torch.load('../data/nem_ss/val500M3FT100_xsh_ang6915-30.pt')
hgt = torch.tensor(hgt0).to(torch.cfloat).cuda()
xval = xval/xval.abs().amax(dim=(1,2,3), keepdim=True)
xval_cuda = xval[:128].to(torch.cfloat).cuda()

#%%
NN = NN_upconv_more
rid = 's19'
model = torch.load(f'../data/data_ss/models/{rid}/model_epoch1240.pt')
model.eval()
with torch.no_grad():
    Rs, Hhat, Rb, mu, logvar= model(xval_cuda)
    l1, l2 = loss_fun(xval_cuda, Rs, Hhat, Rb, mu, logvar)
    
    for i in range(3):
        hh = Hhat[i].detach()
        rs0 = Rs[i].detach() 
        Rx = hh @ rs0 @ hh.conj().t() + Rb.detach()[i]
        shat = (rs0 @ hh.conj().t() @ Rx.inverse()@xval_cuda.permute(0,2,3,1)[i,:,:,:, None]).cpu() 
        for ii in range(J):
            plt.figure()
            plt.imshow(shat[:,:,ii,0].abs())
            plt.title(f'Estimated sources-{ii}')
            plt.show()

            # plt.figure()
            # plt.imshow(rs0[:,:,ii, ii].abs().cpu())
            # plt.title(f'Epoch{epoch}_estimated V-{ii}')
            # plt.savefig(fig_loc + f'Epoch{epoch}_estimated V-{ii}')
            # plt.show()
            plt.close('all')
        print('h_corr', h_corr(hh, hgt[i]))           

print('done')