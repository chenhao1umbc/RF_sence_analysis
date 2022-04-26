#%% v23000
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

torch.autograd.set_detect_anomaly(True)
from vae_model import *
class NN11(nn.Module):
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
            Down(in_channels=32, out_channels=8),
            Reshape(-1, 8*12*12),
            LinearBlock(8*12*12, 64),
            nn.Linear(64, 3),
            )   
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

        "Neural nets for H,V"
        for i in range(self.K):
            if i == 0:
                inp = x
            else:
                tmp = hj[...,None]@W@inp.permute(2,3,0,1)[...,None]
                inp = inp - tmp.squeeze().permute(2,3,0,1)
            res = self.est(torch.cat((inp.real, inp.imag), dim=1)) #vj,Rb,ang
            vj = threshold(res[:, 0:1].exp(), floor=1e-3, ceiling=1e2)
            sb = threshold(res[:, 1:2].exp(), floor=1e-3, ceiling=1e2)
            Rb = (sb*torch.ones(batch_size, self.M, \
                device=sb.device)).diag_embed().to(torch.cfloat) # shape:[I, M, M]

            ch = torch.pi*torch.arange(self.M, device=res.device)
            hj = ((res[:, 2:].tanh() @ ch[None,:])*1j).exp() # shape:[I, M]
            h_all.append(hj)

            "Wienter filter to get coarse shat"
            Rs = vj[..., None].to(torch.cfloat)  #shape of [I,1,1]
            Rx = hj[...,None] @ Rs @ hj[:,None].conj() + Rb # shape of [I,M,M]
            W = Rs @ hj[:, None,].conj() @ Rx.inverse()  # shape of [N,F,I,1,M]
            shat = (W @ x.permute(2,3,0,1)[...,None]).squeeze().permute(2,0,1) #[I, N, F]
        
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
rid = '' # running id
fig_loc = '../data/nem_ss/figures/'
mod_loc = '../data/nem_ss/models/'
if not(os.path.isdir(fig_loc + f'/rid{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'rid{rid}/')
    os.mkdir(mod_loc + f'rid{rid}/')
fig_loc = fig_loc + f'rid{rid}/'
mod_loc = mod_loc + f'rid{rid}/'

I = 3000 # how many samples
M, N, F, K = 3, 100, 100, 3
NF = N*F
eps = 5e-4
opts = {}
opts['batch_size'] = 64
opts['lr'] = 1e-3
opts['n_epochs'] = 1500

d = torch.load('../data/nem_ss/tr3kM3FT100.pt')
d = awgn_batch(d, snr=30, seed=1)
xtr = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]) # [sample,M,N,F]
xtr = xtr.to(torch.cfloat)
data = Data.TensorDataset(xtr[:I])
tr = Data.DataLoader(data, batch_size=opts['batch_size'], shuffle=True, drop_last=True)
xval, _ , hgt = d = torch.load('../data/nem_ss/val500M3FT100_xsh.pt')
xval_cuda = xval[:128].to(torch.cfloat).cuda()

loss_iter, loss_tr, loss_eval = [], [], []
NN = NN9
model = NN(M,K,N).cuda()
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
        loss = loss_fun(x, Rs, Hhat, Rb, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        torch.cuda.empty_cache()
 
    loss_tr.append(loss.detach().cpu().item()/opts['batch_size'])
    if epoch%10 == 0:
        plt.figure()
        plt.plot(loss_tr, '-or')
        plt.title(f'Loss fuction at epoch{epoch}')
        plt.savefig(fig_loc + f'Epoch{epoch}_LossFunAll')

        plt.figure()
        plt.plot(loss_tr[-50:], '-or')
        plt.title(f'Last 50 of loss at epoch{epoch}')
        plt.savefig(fig_loc + f'Epoch{epoch}_last50')

        model.eval()
        with torch.no_grad():
            Rs, Hhat, Rb, mu, logvar= model(xval_cuda)
            loss = loss_fun(xval_cuda, Rs, Hhat, Rb, mu, logvar)
            loss_eval.append(loss.cpu().item()/128)
            plt.figure()
            plt.plot(loss_eval, '-xb')
            plt.title(f'Accumulated validation loss at epoch{epoch}')
            plt.savefig(fig_loc + f'Epoch{epoch}_val')

            hh, rs0= Hhat[0], Rs[0]
            Rx = hh @ rs0 @ hh.conj().t() + Rb[0]
            shat = (rs0 @ hh.conj().t() @ Rx.inverse()@x.permute(0,2,3,1)[0,:,:,:, None]).cpu() 
            print(epoch, Rb[0], Rb.sum()/3/128)
            for ii in range(K):
                plt.figure()
                plt.imshow(shat[:,:,ii,0].abs())
                plt.colorbar()
                plt.title(f'Epoch{epoch}_estimated sources-{ii}')
                plt.savefig(fig_loc + f'Epoch{epoch}_estimated sources-{ii}')
                plt.show()

                plt.figure()
                plt.imshow(rs0[:,:,ii, ii].abs().cpu())
                plt.colorbar()
                plt.title(f'Epoch{epoch}_estimated V-{ii}')
                plt.savefig(fig_loc + f'Epoch{epoch}_estimated V-{ii}')
                plt.show()
                plt.close('all')
            print(f'epoch{epoch} h_corr is ', h_corr(hh.cpu(), torch.tensor(hgt)))
        torch.save(model, mod_loc+f'model_epoch{epoch}.pt')
print('done')

# %%
