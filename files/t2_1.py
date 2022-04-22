from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)

#%%
"raw data processing"
FT = 100
var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
data = {}
for i in range(6):
    temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
    dd = (np.sum((abs(temp['x'])**2), 1)**0.5).reshape(2000, 1)
    # dd = np.abs(temp['x']).max(axis=1).reshape(2000, 1)
    data[i] = temp['x'] / dd  # normalized very sample to 1

*_, Z = stft(data[2], fs=4e7, nperseg=FT, boundary=None)
s = np.roll(Z, FT//2, axis=1)  # roll nperseg//2
theta = np.array([15, 60, -45])*np.pi/180  #len=M, signal AOAs  
h = np.exp(-1j*np.pi*np.arange(0, 3)[:,None]@np.sin(theta)[None, :])  # shape of [M, J]

s = torch.tensor(s)
hj = torch.tensor(h[:,0])
d = (hj[:, None] @ s[...,None, None]).squeeze().permute(0,3,1,2)
d = s[:,None]

#%%
if torch.__version__[:5] != '1.8.1':
    def mydet(x):
        return x.det()
    RAdam = torch.optim.RAdam
else:
    RAdam = optim.RAdam

from vae_model import *
class NN(nn.Module):
    """This is recursive Wiener filter version
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M=3, K=3, im_size=100):
        super().__init__()
        self.dz = 32
        self.K, self.M = K, M

        # Estimate H and coarse V
        self.v_net = nn.Sequential(
            DoubleConv(in_channels=M*2, out_channels=32),
            DoubleConv(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=4),
            ) 
        self.v_out = OutConv(in_channels=4, out_channels=1)
        self.hb_net = nn.Sequential(
            Down(in_channels=1, out_channels=32),
            Down(in_channels=32, out_channels=16),
            Down(in_channels=16, out_channels=8),
            Reshape(-1, 8*12*12),
            )
        # Estimate H
        self.h_net = nn.Sequential(
            LinearBlock(8*12*12, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 1),
            nn.Tanh()
            )   
        # Estimate Rb
        self.b_net = nn.Sequential(
            LinearBlock(8*12*12, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 1),
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
                temp = hj[...,None]@W@inp.permute(2,3,0,1)[...,None]
                inp = inp - temp.squeeze().permute(2,3,0,1)
            temp = self.v_net(torch.cat((inp.real, inp.imag), dim=1)).exp() 
            vj = self.v_out(temp).exp() #sigma_s**2 >=0
            vj = threshold(vj, floor=1e-3, ceiling=1e3)  # shape of [I, 1, N, F]
            hb = self.hb_net(vj)
            ang = self.h_net(hb)  # shape of [I,1]
            sig_b_squared = self.b_net(hb).exp() # shape of [I,1]
            "Get H"
            ch = torch.pi*torch.arange(self.M, device=ang.device)
            hj = ((ang @ ch[None,:])*1j).exp() # shape:[I, M]
            h_all.append(hj)

            "Get Rb, the energy of the rest"
            Rb = (sig_b_squared*torch.ones(batch_size, self.M, device=ch.device))\
                .diag_embed().to(torch.cfloat) # shape:[I, M, M]

            "Wienter filter to get coarse shat"
            Rs = vj.permute(2,3,0,1)[..., None].to(torch.cfloat)  #shape of [N,F,I,1,1]
            Rx = hj[...,None] @ Rs @ hj[:,None].conj() + Rb # shape of [N,F,I,M,M]
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

        return vhat[...,None, None], Hhat, Rb, mu, logvar

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

rid = 't2_1' # running id
fig_loc = '../data/nem_ss/figures/'
mod_loc = '../data/nem_ss/models/'
if not(os.path.isdir(fig_loc + f'/rid{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'rid{rid}/')
    os.mkdir(mod_loc + f'rid{rid}/')
fig_loc = fig_loc + f'rid{rid}/'
mod_loc = mod_loc + f'rid{rid}/'

I = 2000 # how many samples
M, N, F, K = 1, 100, 100, 1
NF = N*F
eps = 5e-4
opts = {}
opts['batch_size'] = 64
opts['lr'] = 1e-4
opts['n_epochs'] = 1000

xtr = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]) # [sample,M,N,F]
xtr = xtr.to(torch.cfloat)
data = Data.TensorDataset(xtr[:I])
tr = Data.DataLoader(data, batch_size=opts['batch_size'], shuffle=True, drop_last=True)

loss_iter, loss_tr, loss_eval = [], [], []
model = NN(M,K,N).cuda()
for w in model.parameters():
    nn.init.normal_(w, mean=0., std=0.01)
optimizer = torch.optim.Adam(model.parameters(),
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

        plt.figure()
        plt.imshow(Rs[0].detach().cpu().squeeze().abs())
        plt.colorbar()
        plt.title(f'vj at epoch{epoch}')
        plt.savefig(fig_loc + f'Epoch{epoch}_vj')

        # with torch.no_grad():
        #     r = []
        #     for ii in range(opts['batch_size']):
        #         r.append(h_corr(Hhat[ii].cpu(), hj[:, None]))
        # print(sum(r)/opts['batch_size'], Rb[0].cpu().detach())
        print(f'epoch{epoch} ', Rb[0].cpu().detach())
        plt.close('all')
print('done')