#%% v24000
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)

if torch.__version__[:5] != '1.8.1':
    def mydet(x):
        return x.det()
    RAdam = torch.optim.RAdam
else:
    RAdam = optim.RAdam

# torch.autograd.set_detect_anomaly(True)
from vae_model import *
class NN14(nn.Module):
    """This is recursive Wiener filter version, with Rb threshold of [1e-3, 1e2]
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M=3, K=3, im_size=100):
        super().__init__()
        self.dz = 32
        self.K, self.M = K, M

        # Estimate H and coarse V
        self.est = nn.Sequential(
            Down(in_channels=M*2, out_channels=64),
            Down(in_channels=64, out_channels=32),
            Down(in_channels=32, out_channels=4),
            Reshape(-1, 4*12*12),
            LinearBlock(4*12*12, 64),
            nn.Linear(64, 1),
            )
        self.b1 = nn.Linear(100, 1)
        self.b2 = nn.Linear(100, 1)
           
        # Estimate V using auto encoder
        self.encoder = nn.Sequential(
            Down(in_channels=1, out_channels=64),
            Down(in_channels=64, out_channels=16),
            OutConv(in_channels=16, out_channels=1),
            )
        self.fc1 = nn.Linear(25*25, 2*self.dz)
        self.decoder = nn.Sequential(
            DoubleConv(in_channels=self.dz+2, out_channels=64),
            DoubleConv(in_channels=64, out_channels=16),
            OutConv(in_channels=16, out_channels=1),
            ) 
        self.im_size = im_size
        x = torch.linspace(-1, 1, im_size)
        y = torch.linspace(-1, 1, im_size)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        batch_size, _, N, F = x.shape
        z_all, v_all, h_all = [], [], []

        "Neural nets for H,V"
        for i in range(self.K):
            if i == 0:
                inp = x
            else:
                tmp = hj[...,None]@W@inp.permute(2,3,0,1)[...,None]
                inp = inp - tmp.squeeze().permute(2,3,0,1)
            ang = self.est(torch.cat((inp.real, inp.imag), dim=1)) #vj,Rb,ang

            # sb = self.b2(self.b1(inp.abs()).squeeze()).mean(dim=1).exp()
            # Rb = (sb[:None]*torch.ones(batch_size, self.M, \
            #     device=x.device)).diag_embed().to(torch.cfloat) # shape:[I, M, M]
            Rb = (1.4e-3*torch.ones(batch_size, self.M, \
                device=x.device)).diag_embed().to(torch.cfloat) # shape:[I, M, M]

            ch = torch.pi*torch.arange(self.M, device=inp.device)
            hj = ((ang.tanh() @ ch[None,:])*1j).exp() # shape:[I, M]
            h_all.append(hj)

            "Wienter filter to get coarse shat"
            Rx = hj[...,None] @ hj[:,None].conj() + Rb # shape of [I,M,M]
            W = hj[:, None,].conj() @ Rx.inverse()  # shape of [N,F,I,1,M]
            shat = (W @ x.permute(2,3,0,1)[...,None]).squeeze().permute(2,0,1) #[I, N, F]
            shat = shat/shat.detach().abs().max()

            "Encoder"
            xx = self.encoder(shat[:,None].abs())
            "Get latent variable"
            zz = self.fc1(xx.reshape(batch_size,-1))
            mu = zz[:,::2]
            logvar = zz[:,1::2]
            z = self.reparameterize(mu, logvar)
            z_all.append(z)
            
            "Decoder to get V"
            # View z as 4D tensor to be tiled across new N and F dimensions            
            zr = z.view((batch_size, self.dz)+ (1, 1))  #Shape: IxDxNxF
            # Tile across to match image size
            zr = zr.expand(-1, -1, self.im_size, self.im_size)  #Shape: IxDx64x64
            # Expand grids to batches and concatenate on the channel dimension
            zbd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                        self.y_grid.expand(batch_size, -1, -1, -1), zr), dim=1) # Shape: Ix(dz*K+2)xNxF
            v = self.decoder(zbd).exp()
            v_all.append(threshold(v, floor=1e-3, ceiling=1e2)) # 1e-3 to 1e2
        Hhat = torch.stack(h_all, 2) # shape:[I, M, K]
        vhat = torch.stack(v_all, 4).squeeze().to(torch.cfloat) # shape:[I, N, F, K]

        return vhat.diag_embed(), Hhat, Rb, mu, logvar
    
def loss_fun(x, Rs, Hhat, Rb, mu, logvar, beta=0.5):
    x = x.permute(0,2,3,1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb
    Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
    try:
        ll = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze() 
    except:
        torch.save((x, Rx, Rs, Hhat, Rb), f'rid{rid}x_Rx_Rs_Hhat_Rb.pt')
        print('error happpened, data saved and stop')
        ll = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze()
    return -ll.sum().real + beta*kl

#%%

I = 3000 # how many samples
M, N, F, K = 3, 100, 100, 3
NF = N*F
eps = 5e-4
opts = {}
opts['batch_size'] = 64
opts['lr'] = 1e-3
opts['n_epochs'] = 1500

xval, _ , hgt0 = torch.load('../data/nem_ss/val500M3FT100_xsh.pt')
hgt = torch.tensor(hgt0).to(torch.cfloat).cuda()
xval_cuda = xval[:128].to(torch.cfloat).cuda()

loss_iter, loss_tr, loss_eval = [], [], []

class NN_gtsbd(nn.Module):
    """This is recursive Wiener filter version, with Rb threshold of [1e-3, 1e2]
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M=3, K=3, im_size=100):
        super().__init__()
        self.dz = 32
        self.K, self.M = K, M

        # Estimate H and coarse V
        # self.est = nn.Sequential(
        #     Down(in_channels=M*2, out_channels=64),
        #     Down(in_channels=64, out_channels=32),
        #     Down(in_channels=32, out_channels=4),
        #     Reshape(-1, 4*12*12),
        #     LinearBlock(4*12*12, 64),
        #     nn.Linear(64, 1),
        #     )
        # self.b1 = nn.Linear(100, 1)
        # self.b2 = nn.Linear(100, 1)
           
        # Estimate V using auto encoder
        self.encoder = nn.Sequential(
            Down(in_channels=1, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            Down(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=1),
            )
        self.fc1 = nn.Linear(25*25, 2*self.dz)
        self.decoder = nn.Sequential(
            DoubleConv(in_channels=self.dz+2, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            DoubleConv(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=4),
            OutConv(in_channels=4, out_channels=1),
            ) 
        self.im_size = im_size
        x = torch.linspace(-1, 1, im_size)
        y = torch.linspace(-1, 1, im_size)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        batch_size, _, N, F = x.shape
        z_all, v_all, h_all = [], [], []
        I = x.shape[0]
        "Neural nets for H,V"
        shat_all = []
        for i in range(self.K):
            if i == 0:
                inp = x
            else:
                tmp = hj[...,None]@W@inp.permute(2,3,0,1)[...,None]
                inp = inp - tmp.squeeze().permute(2,3,0,1)

            # sb = self.b2(self.b1(inp.abs()).squeeze()).mean(dim=1).exp()
            # Rb = (sb[:None]*torch.ones(batch_size, self.M, \
            #     device=x.device)).diag_embed().to(torch.cfloat) # shape:[I, M, M]
            Rb = (1.4e-3*torch.ones(batch_size, self.M, \
                device=x.device)).diag_embed().to(torch.cfloat) # shape:[I, M, M]
            hj = hgt[:,i].repeat(I).reshape(I,3) # shape:[I, M]
            h_all.append(hj)

            "Wienter filter to get coarse shat"
            Rx = hj[...,None] @ hj[:,None].conj() + Rb # shape of [I,M,M]
            W = hj[:, None,].conj() @ Rx.inverse()  # shape of [N,F,I,1,M]
            shat = (W @ inp.permute(2,3,0,1)[...,None]).squeeze().permute(2,0,1) #[I, N, F]
            shat = shat/shat.detach().abs().max()
            shat_all.append(shat)
        
        for i in range(self.K):
            "Encoder"
            xx = self.encoder(shat_all[i][:,None].abs())
            "Get latent variable"
            zz = self.fc1(xx.reshape(batch_size,-1))
            mu = zz[:,::2]
            logvar = zz[:,1::2]
            z = self.reparameterize(mu, logvar)
            z_all.append(z)
            
            "Decoder to get V"
            # View z as 4D tensor to be tiled across new N and F dimensions            
            zr = z.view((batch_size, self.dz)+ (1, 1))  #Shape: IxDxNxF
            # Tile across to match image size
            zr = zr.expand(-1, -1, self.im_size, self.im_size)  #Shape: IxDx64x64
            # Expand grids to batches and concatenate on the channel dimension
            zbd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                        self.y_grid.expand(batch_size, -1, -1, -1), zr), dim=1) # Shape: Ix(dz*K+2)xNxF
            v = self.decoder(zbd).exp()
            v_all.append(threshold(v, floor=1e-3, ceiling=1e2)) # 1e-3 to 1e2
        Hhat = torch.stack(h_all, 2) # shape:[I, M, K]
        vhat = torch.stack(v_all, 4).squeeze().to(torch.cfloat) # shape:[I, N, F, K]

        return vhat.diag_embed(), Hhat, Rb, mu, logvar


def loss_fun(x, Rs, Hhat, Rb, mu, logvar, beta=1e-3):
    x = x.permute(0,2,3,1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb
    Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
    try:
        ll = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze() 
    except:
        torch.save((x, Rx, Rs, Hhat, Rb), f'rid{rid}x_Rx_Rs_Hhat_Rb.pt')
        print('error happpened, data saved and stop')
        ll = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze()
    return -ll.sum().real, beta*kl


# %%
model = torch.load('../data/nem_ss/models/s8/model_epoch140.pt')
import pdb
model.eval()
with torch.no_grad():
    Rs, Hhat, Rb, mu, logvar= model(xval_cuda)  
    for i in [0,10,20,30,40,50]:
        hh, rs0= Hhat[i], Rs[i]
        Rx = hh @ rs0 @ hh.conj().t() + Rb[i]
        shat = (rs0 @ hh.conj().t() @ Rx.inverse()@xval_cuda.permute(0,2,3,1)[i,:,:,:, None]).cpu()

        for ii in range(3):
            plt.figure()
            plt.imshow(shat[:,:,ii,0].abs())
            plt.colorbar()
            plt.title(f'Sample{i}-Estimated sources-{ii+1}')
            # plt.savefig(fig_loc + f'Epoch{epoch}_estimated sources-{ii}')
            plt.show()

            plt.figure()
            plt.imshow(rs0[:,:,ii, ii].abs().cpu())
            plt.colorbar()
            plt.title(f'Sample{i}-Estimated V-{ii+1}')
            # plt.savefig(fig_loc + f'Epoch{epoch}_estimated V-{ii}')
            plt.show()
            # plt.close('all')
print('done')