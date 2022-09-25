#%% tr_val_plot
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())

"make the result reproducible"
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)       # current GPU seed
torch.cuda.manual_seed_all(seed)   # all GPUs seed
torch.backends.cudnn.deterministic = True  #True uses deterministic alg. for cuda
torch.backends.cudnn.benchmark = False  #False cuda use the fixed alg. for conv, may slower

rid = 'tr_val_plot' 
fig_loc = '../data/data_ss/figures/'
mod_loc = '../data/data_ss/models/'
if not(os.path.isdir(fig_loc + f'/{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'{rid}/')
    os.mkdir(mod_loc + f'{rid}/')
fig_loc = fig_loc + f'{rid}/'
mod_loc = mod_loc + f'{rid}/'
# torch.autograd.set_detect_anomaly(True)

from vae_modules import *
def lower2matrix(rx12):
    ind = torch.tril_indices(3,3)
    indx = np.diag_indices(3)
    rx_inv_hat = torch.zeros(rx12.shape[0], 3, 3, dtype=torch.cfloat).cuda()
    rx_inv_hat[:, ind[0], ind[1]] = rx12[:, :6] + 1j*rx12[:,6:]
    rx_inv_hat = rx_inv_hat + rx_inv_hat.permute(0,2,1).conj()
    rx_inv_hat[:, indx[0], indx[1]] = rx_inv_hat[:, indx[0], indx[1]]/2
    return rx_inv_hat

class NNet_s10(nn.Module):
    """This is recursive Wiener filter version, with Rb threshold of [1e-3, 1e2]
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M, K, im_size):
        super().__init__()
        self.dz = 32
        self.J, self.M = K, M
        down_size = int(im_size/4)
        self.mainnet = nn.Sequential(
            FC_layer_g(12, 128),
            FC_layer_g(128, 128),
        )
        self.hnet = nn.Sequential(
            FC_layer_g(128, 128),
            FC_layer_g(128, 128),
            FC_layer_g(128, 64),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self.rxnet = nn.Sequential(
            FC_layer_g(128, 128),
            FC_layer_g(128, 128),
            FC_layer_g(128, 64),
            FC_layer_g(64, 32),
            nn.Linear(32, 12)
        )

        self.encoder = nn.Sequential(
            Down_g(in_channels=1, out_channels=64),
            DoubleConv_g(in_channels=64, out_channels=32),
            Down_g(in_channels=32, out_channels=16),
            DoubleConv_g(in_channels=16, out_channels=1),
            )
        self.fc1 = nn.Linear(down_size*down_size, 2*self.dz)
        self.decoder = nn.Sequential(
            nn.Linear(self.dz, down_size*down_size),
            Reshape(-1, 1, down_size, down_size),
            Up_g(in_channels=1, out_channels=64),
            DoubleConv_g(in_channels=64, out_channels=32),
            Up_g(in_channels=32, out_channels=16),
            DoubleConv_g(in_channels=16, out_channels=8),
            nn.Conv2d(8, 8, kernel_size=3, padding=(1,2)),
            nn.GroupNorm(num_groups=max(8//4,1), num_channels=8),
            nn.LeakyReLU(inplace=True),
            OutConv(in_channels=8, out_channels=1),
            ) 
        self.bilinear = nn.Linear(self.dz, self.dz, bias=False)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):     
        btsize, M, N, F = x.shape
        z_all, v_all, h_all = [], [], []
        ch = np.pi*torch.arange(self.M, device=x.device)
        xj = x.permute(0,2,3,1)[...,None]  # shape of [I,N,F,M,1]
        for i in range(self.J):
            "Get H estimation"
            ind = torch.tril_indices(3,3)
            rx = (xj@xj.transpose(-1,-2).conj()).mean(dim=(1,2))
            rx_lower = rx[:, ind[0], ind[1]]
            mid =self.mainnet(torch.stack((rx_lower.real,rx_lower.imag),\
                 dim=1).reshape(btsize,-1))
            ang = self.hnet(mid)
            temp = ang@ch[None,:]
            hhat = (1j*temp).exp()  # shape of [I, M]
            h_all.append(hhat)

            "Get Rx inverse"
            rx_index = self.rxnet(mid)
            rx_inv = lower2matrix(rx_index) # shape of [I, M, M]
        
            "Encoder part"
            w = rx_inv@hhat[...,None] / \
                (hhat[:,None,:].conj()@rx_inv@hhat[...,None])
            shat = w.permute(0,2,1).conj()[:,None,None]@xj
            xx = self.encoder(shat.squeeze()[:,None].abs())

            "Get latent variable"
            zz = self.fc1(xx.reshape(btsize,-1))
            mu = zz[:,::2]
            logvar = zz[:,1::2]
            z = self.reparameterize(mu, logvar)
            wz = self.bilinear(z)
            z_all.append(z)
            z_all.append(wz)
            
            "Decoder to get V"
            v = self.decoder(z).square().squeeze()  # shape of [I,N,F]
            v_all.append(threshold(v, floor=1e-6, ceiling=1e2)) # 1e-6 to 1e2

            "Remove the current component"
            rxinvh = rx_inv@hhat[...,None]  # shape of [I, M, 1]
            v_rxinv_h_herm = (v[...,None, None]*rxinvh[:,None, None]).transpose(-1,-2).conj() 
            cj = hhat[:,None,None,:,None] * (v_rxinv_h_herm @ xj) # shape of [I,N,F,M,1]
            xj = xj - cj
       
        Hhat = torch.stack(h_all, 2) # shape:[I, M, J]
        vhat = torch.stack(v_all, 3).to(torch.cfloat) # shape:[I, N, F, J]
        zall = torch.stack(z_all, dim=1)

        # Rb = (b@b.conj().permute(0,1,2,4,3)).mean(dim=(1,2)).squeeze()
        eye = torch.eye(M, device='cuda')
        Rb = torch.stack(tuple(eye for ii in range(btsize)), 0)*1e-4

        return vhat.diag_embed(), Hhat, Rb, mu, logvar, zall

def loss_fun(x, Rs, Hhat, Rb, mu, logvar, zall, beta=1):
    I, M, J = x.shape[0], x.shape[1], Rs.shape[-1]
    x = x.permute(0,2,3,1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb
    Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
    ll = -(np.pi*Rx.det()).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze() 

    return -ll.sum(), beta*kl #+ loss_slotCEL

I = 18000 # how many samples
M, N, F, J = 3, 64, 66, 3
eps = 5e-4
opts = {}
opts['batch_size'] = 128
opts['n_epochs'] = 701
opts['lr'] = 1e-3

d = torch.load('../data/nem_ss/tr18kM3FT64_data3.pt')
xtr = (d/d.abs().amax(dim=(1,2,3), keepdim=True)) # [sample,M,N,F]
xtr = xtr.to(torch.cfloat)
data = Data.TensorDataset(xtr[:I])
tr = Data.DataLoader(data, batch_size=opts['batch_size'], shuffle=True, drop_last=True)

xval, sval, hgt = torch.load('../data/nem_ss/val1kM3FT64_xsh_data3.pt')
sval= sval.permute(0,2,3,1)
xval = xval/xval.abs().amax(dim=(1,2,3), keepdim=True)
data = Data.TensorDataset(xval, sval, hgt)
dval = Data.DataLoader(data, batch_size=200, drop_last=True)


print('starting date time ', datetime.now())
loss_tr, loss1, loss2, loss_eval = [], [], [], []
for epoch in range(0,opts['n_epochs'],5):
    model = torch.load(f'../data/data_ss/models/s71/model_epoch{epoch}.pt')
    model.eval()
    loss_iter = []
    with torch.no_grad():
        for i, (x,) in enumerate(tr): 
            if i > 0:
                break
            x = x.cuda()        
            Rs, Hhat, Rb, mu, logvar, zall= model(x)
            l1, l2 = loss_fun(x, Rs, Hhat, Rb, mu, logvar, zall)
            loss = l1 + l2
            torch.cuda.empty_cache()
            loss_iter.append(loss.cpu().item().real/x.shape[0])
        loss_tr.append(sum(loss_iter)/len(loss_iter))

plt.figure()
plt.plot(loss_tr, '-o')
plt.plot(loss_eval, '-x')
plt.title('Learning curve for 3 classes')
plt.savefig('3-class learning curve')

print('done')
print('End date time ', datetime.now())

#%%

#%% s87
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())

"make the result reproducible"
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)       # current GPU seed
torch.cuda.manual_seed_all(seed)   # all GPUs seed
torch.backends.cudnn.deterministic = True  #True uses deterministic alg. for cuda
torch.backends.cudnn.benchmark = False  #False cuda use the fixed alg. for conv, may slower

rid = 's87' 
fig_loc = '../data/data_ss/figures/'
mod_loc = '../data/data_ss/models/'
if not(os.path.isdir(fig_loc + f'/{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'{rid}/')
    os.mkdir(mod_loc + f'{rid}/')
fig_loc = fig_loc + f'{rid}/'
mod_loc = mod_loc + f'{rid}/'
# torch.autograd.set_detect_anomaly(True)

from vae_modules import *
def lower2matrix(rx42):
    ind = torch.tril_indices(6,6)
    indx = np.diag_indices(6)
    rx_inv_hat = torch.zeros(rx42.shape[0], 6, 6, dtype=torch.cfloat).cuda()
    rx_inv_hat[:, ind[0], ind[1]] = rx42[:, :21] + 1j*rx42[:,21:]
    rx_inv_hat = rx_inv_hat + rx_inv_hat.permute(0,2,1).conj()
    rx_inv_hat[:, indx[0], indx[1]] = rx_inv_hat[:, indx[0], indx[1]]/2
    return rx_inv_hat

class NNet_s10(nn.Module):
    """This is recursive Wiener filter version, with Rb threshold of [1e-3, 1e2]
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M, K, im_size):
        super().__init__()
        self.dz = 32
        self.J, self.M = K, M
        down_size = int(im_size/4)
        self.mainnet = nn.Sequential(
            FC_layer_g(42, 128),
            FC_layer_g(128, 128),
        )
        self.hnet = nn.Sequential(
            FC_layer_g(128, 128),
            FC_layer_g(128, 128),
            FC_layer_g(128, 64),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self.rxnet = nn.Sequential(
            FC_layer_g(128, 128),
            FC_layer_g(128, 128),
            FC_layer_g(128, 64),
            FC_layer_g(64, 64),
            nn.Linear(64, 42)
        )

        self.encoder = nn.Sequential(
            Down_g(in_channels=1, out_channels=64),
            DoubleConv_g(in_channels=64, out_channels=32),
            Down_g(in_channels=32, out_channels=16),
            DoubleConv_g(in_channels=16, out_channels=1),
            )
        self.fc1 = nn.Linear(down_size*down_size, 2*self.dz)
        self.decoder = nn.Sequential(
            nn.Linear(self.dz, down_size*down_size),
            Reshape(-1, 1, down_size, down_size),
            Up_g(in_channels=1, out_channels=64),
            DoubleConv_g(in_channels=64, out_channels=32),
            Up_g(in_channels=32, out_channels=16),
            DoubleConv_g(in_channels=16, out_channels=8),
            nn.Conv2d(8, 8, kernel_size=3, padding=(1,2)),
            nn.GroupNorm(num_groups=max(8//4,1), num_channels=8),
            nn.LeakyReLU(inplace=True),
            OutConv(in_channels=8, out_channels=1),
            ) 
        self.bilinear = nn.Linear(self.dz, self.dz, bias=False)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):     
        btsize, M, N, F = x.shape
        z_all, v_all, h_all = [], [], []
        ch = np.pi*torch.arange(self.M, device=x.device)
        xj = x.permute(0,2,3,1)[...,None]  # shape of [I,N,F,M,1]
        for i in range(self.J):
            "Get H estimation"
            ind = torch.tril_indices(M,M)
            rx = (xj@xj.transpose(-1,-2).conj()).mean(dim=(1,2))
            rx_lower = rx[:, ind[0], ind[1]]
            mid =self.mainnet(torch.stack((rx_lower.real,rx_lower.imag),\
                 dim=1).reshape(btsize,-1))
            ang = self.hnet(mid)
            temp = ang@ch[None,:]
            hhat = (1j*temp).exp()  # shape of [I, M]
            h_all.append(hhat)

            "Get Rx inverse"
            rx_index = self.rxnet(mid)
            rx_inv = lower2matrix(rx_index) # shape of [I, M, M]
        
            "Encoder part"
            w = rx_inv@hhat[...,None] / \
                (hhat[:,None,:].conj()@rx_inv@hhat[...,None])
            shat = w.permute(0,2,1).conj()[:,None,None]@xj
            xx = self.encoder(shat.squeeze()[:,None].abs())

            "Get latent variable"
            zz = self.fc1(xx.reshape(btsize,-1))
            mu = zz[:,::2]
            logvar = zz[:,1::2]
            z = self.reparameterize(mu, logvar)
            wz = self.bilinear(z)
            z_all.append(z)
            z_all.append(wz)
            
            "Decoder to get V"
            v = self.decoder(z).square().squeeze()  # shape of [I,N,F]
            v_all.append(threshold(v, floor=1e-6, ceiling=1e2)) # 1e-6 to 1e2

            "Remove the current component"
            rxinvh = rx_inv@hhat[...,None]  # shape of [I, M, 1]
            v_rxinv_h_herm = (v[...,None, None]*rxinvh[:,None, None]).transpose(-1,-2).conj() 
            cj = hhat[:,None,None,:,None] * (v_rxinv_h_herm @ xj) # shape of [I,N,F,M,1]
            xj = xj - cj
       
        Hhat = torch.stack(h_all, 2) # shape:[I, M, J]
        vhat = torch.stack(v_all, 3).to(torch.cfloat) # shape:[I, N, F, J]
        zall = torch.stack(z_all, dim=1)

        # Rb = (b@b.conj().permute(0,1,2,4,3)).mean(dim=(1,2)).squeeze()
        eye = torch.eye(M, device='cuda')
        Rb = torch.stack(tuple(eye for ii in range(btsize)), 0)*1e-3

        return vhat.diag_embed(), Hhat, Rb, mu, logvar, zall

def loss_fun(x, Rs, Hhat, Rb, mu, logvar, zall, beta=1):
    I, M, J = x.shape[0], x.shape[1], Rs.shape[-1]
    x = x.permute(0,2,3,1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb
    Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
    ll = -(np.pi*Rx.det()).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze() 

    # "Slot contrastive loss"
    # inp = (zall[:,0::2]@zall[:,1::2].permute(0,2,1)).reshape(I*J, J) # shape of [N,J,J]
    # target = torch.cat([torch.arange(J) for i in range(I)]).cuda()
    # loss_slotCEL = nn.CrossEntropyLoss(reduction='none')(inp, target).sum()

    # "My own loss for H"
    # HHt = Hhat@Hhat.permute(0,2,1).conj() 
    # temp = x[...,None]@ x[:,:,:,None].conj()
    # rx = temp.mean(dim=(1,2))
    # term = (((rx- HHt/100).abs())**2).mean()

    return -ll.sum(), beta*kl #+ loss_slotCEL

I = 18000 # how many samples
M, N, F, J = 6, 64, 66, 6
eps = 5e-4
opts = {}
opts['batch_size'] = 128
opts['n_epochs'] = 701
opts['lr'] = 1e-3

d = torch.load('../data/nem_ss/tr18kM6FT64_data3.pt')
xtr = (d/d.abs().amax(dim=(1,2,3), keepdim=True)) # [sample,M,N,F]
xtr = xtr.to(torch.cfloat)
data = Data.TensorDataset(xtr[:I])
tr = Data.DataLoader(data, batch_size=opts['batch_size'], shuffle=True, drop_last=True)

xval, sval, hgt = torch.load('../data/nem_ss/val1kM6FT64_xsh_data3.pt')
sval= sval.permute(0,2,3,1)
xval = xval/xval.abs().amax(dim=(1,2,3), keepdim=True)
data = Data.TensorDataset(xval, sval, hgt)
dval = Data.DataLoader(data, batch_size=200, drop_last=True)

print('starting date time ', datetime.now())
loss_tr1, loss1, loss2, loss_eval = [], [], [], []
for epoch in range(0,opts['n_epochs'],5):
    model = torch.load(f'../data/data_ss/models/s87/model_epoch{epoch}.pt')
    model.eval()
    loss_iter = []
    with torch.no_grad():
        for i, (x,) in enumerate(tr): 
            if i > 0:
                break
            x = x.cuda()        
            Rs, Hhat, Rb, mu, logvar, zall= model(x)
            l1, l2 = loss_fun(x, Rs, Hhat, Rb, mu, logvar, zall)
            loss = l1 + l2
            torch.cuda.empty_cache()
            loss_iter.append(loss.cpu().item().real/x.shape[0])
        loss_tr.append(sum(loss_iter)/len(loss_iter))

plt.figure()
plt.plot(loss_tr1, '-o')
plt.plot(loss_tr1, '-o')
plt.title('Learning curves')
plt.savefig('Learning curves')

print('done')
print('End date time ', datetime.now())