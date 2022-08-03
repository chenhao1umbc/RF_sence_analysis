#%% s22_
rid = 's22_' # running id
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)

fig_loc = '../data/data_ss/figures/'
mod_loc = '../data/data_ss/models/'
if not(os.path.isdir(fig_loc + f'/{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'{rid}/')
    os.mkdir(mod_loc + f'{rid}/')
fig_loc = fig_loc + f'{rid}/'
mod_loc = mod_loc + f'{rid}/'

if torch.__version__[:5] != '1.8.1':
    def mydet(x):
        return x.det()
    RAdam = torch.optim.RAdam
else:
    RAdam = optim.RAdam

torch.autograd.set_detect_anomaly(True)
from vae_model import *
class NN_s0(nn.Module):
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
            nn.Linear(64,1)
        )
        self.hnet2 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,1)
        )
        self.hnet3 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,1)
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
        z_all, v_all = [], []
        I = x.shape[0]
        x0 = x.permute(0,2,3,1)[...,None]
        Rx = x0 @ x.permute(0,2,3,1).conj()[...,None,:]
        Rx = Rx.mean(dim=(1,2)) # shape of [I,M,M]
        rx_inv = Rx.inverse()
        temp = self.mainnet(torch.stack((Rx.real, Rx.imag), dim=1).reshape(I,-1))
        ang = torch.stack((self.hnet1(temp), self.hnet2(temp), self.hnet3(temp)), dim=2)
        ch = np.pi*torch.arange(self.M, device=ang.device)
        Hhat = ((ch[:,None] @ ang).tanh()*1j).exp()

        b = x0
        for i in range(self.J):
            hhat = Hhat[:,:,i]
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
            v = self.decoder(z).abs()
            v_all.append(threshold(v, floor=1e-3, ceiling=1e2)) # 1e-3 to 1e2
        Rb = (b@b.conj().permute(0,1,2,4,3)).mean(dim=(1,2)).squeeze()
        # Hhat = torch.stack(h_all, 2) # shape:[I, M, K]
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
    return -ll.mean().real, beta*kl + 0.1*term3

#%% load data
I = 90 # how many samples
M, N, F, J = 3, 100, 100, 3
NF = N*F
eps = 5e-4
opts = {}
opts['batch_size'] = 64
opts['lr'] = 1e-3
opts['n_epochs'] = 1001

d = torch.load('../data/nem_ss/tr9kM3FT100_ang6915-30.pt')
d = awgn_batch(d, snr=30, seed=1)
xtr = (d/d.abs().amax(dim=(1,2,3), keepdim=True)) # [sample,M,N,F]
xtr = xtr.to(torch.cfloat)
data = Data.TensorDataset(xtr[:I])
tr = Data.DataLoader(data, batch_size=opts['batch_size'], shuffle=True, drop_last=True)
xval, _ , hgt0 = torch.load('../data/nem_ss/val500M3FT100_xsh_ang6915-30.pt')
hgt = torch.tensor(hgt0).to(torch.cfloat).cuda()
xval = xval/xval.abs().amax(dim=(1,2,3), keepdim=True)
xval_cuda = xval[:128].to(torch.cfloat).cuda()

#%%
loss_iter, loss_tr, loss1, loss2, loss_eval = [], [], [], [], []
NN = NN_s0
model = NN(M,J,N).cuda()
for w in model.parameters():
    nn.init.normal_(w, mean=0., std=0.01)

optimizer = RAdam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)

for epoch in range(opts['n_epochs']):
    model.train()
    for i, (x,) in enumerate(tr): 
        x = x.cuda()
        optimizer.zero_grad()         
        Rs, Hhat, Rb, mu, logvar= model(x)
        l1, l2 = loss_fun(x, Rs, Hhat, Rb, mu, logvar)
        loss = l1 + l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        torch.cuda.empty_cache()

    loss_tr.append(loss.detach().cpu().item()/opts['batch_size'])
    loss1.append(l1.detach().cpu().item()/opts['batch_size'])
    loss2.append(l2.detach().cpu().item()/opts['batch_size'])
    if epoch%10 == 0:
        print(epoch)
        plt.figure()
        plt.plot(loss_tr, '-or')
        plt.title(f'Loss fuction at epoch{epoch}')
        plt.savefig(fig_loc + f'Epoch{epoch}_LossFunAll')

        plt.figure()
        plt.plot(loss1, '-og')
        plt.title(f'Reconstruction loss at epoch{epoch}')
        plt.savefig(fig_loc + f'Epoch{epoch}_Loss1')

        plt.figure()
        plt.plot(loss2, '-og')
        plt.title(f'KL loss at epoch{epoch}')
        plt.savefig(fig_loc + f'Epoch{epoch}_Loss2')

        plt.figure()
        plt.plot(loss_tr[-50:], '-or')
        plt.title(f'Last 50 of loss at epoch{epoch}')
        plt.savefig(fig_loc + f'Epoch{epoch}_last50')

        model.eval()
        with torch.no_grad():
            Rs, Hhat, Rb, mu, logvar= model(xval_cuda)
            l1, l2 = loss_fun(xval_cuda, Rs, Hhat, Rb, mu, logvar)
            loss_eval.append((l1+l2).cpu().item()/128)
            plt.figure()
            plt.plot(loss_eval[-50:], '-xb')
            plt.title(f'last 50 validation loss at epoch{epoch}')
            plt.savefig(fig_loc + f'Epoch{epoch}_val') 
            plt.close('all')           

            hh = Hhat[0].detach()
            rs0 = Rs[0].detach() 
            Rx = hh @ rs0 @ hh.conj().t() + Rb.detach()[0]
            shat = (rs0 @ hh.conj().t() @ Rx.inverse()@xval_cuda.permute(0,2,3,1)[0,:,:,:, None]).cpu() 
            for ii in range(J):
                plt.figure()
                plt.imshow(shat[:,:,ii,0].abs())
                plt.title(f'Epoch{epoch}_estimated sources-{ii}')
                plt.savefig(fig_loc + f'Epoch{epoch}_estimated sources-{ii}')
                plt.show()

                # plt.figure()
                # plt.imshow(rs0[:,:,ii, ii].abs().cpu())
                # plt.title(f'Epoch{epoch}_estimated V-{ii}')
                # plt.savefig(fig_loc + f'Epoch{epoch}_estimated V-{ii}')
                # plt.show()
                plt.close('all')
            print('h_corr', h_corr(hh, hgt[0]))
        torch.save(model, mod_loc+f'model_epoch{epoch}.pt')
print('done')