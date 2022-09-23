#%% s76a
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


#%% define models and functions
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

#%%
I = 18000 # how many samples
M, N, F, J = 6, 64, 66, 6
eps = 5e-4
opts = {}
opts['batch_size'] = 128
opts['n_epochs'] = 701
opts['lr'] = 1e-3

xval, sval, hgt = torch.load('../data/nem_ss/test1kM3FT64_xsh_data3.pt')
sval= sval.permute(0,2,3,1)
xval = xval/xval.abs().amax(dim=(1,2,3), keepdim=True)
data = Data.TensorDataset(xval, sval, hgt)
dval = Data.DataLoader(data, batch_size=200, drop_last=True)

loss_iter, loss_tr, loss1, loss2, loss_eval = [], [], [], [], []
# model = NNet_s10(M,J,N).cuda()
model = torch.load('../data/data_ss/models/s71/model_epoch400.pt')

#%%
print('Start date time ', datetime.now())
model.eval()
hall, sall = [], []
with torch.no_grad():
    for snr in ['inf', 20, 10, 5, 0]:
        av_hcorr, av_scorr = [], []
        for i, (x, s, h) in enumerate(dval):
            print(snr, i)
            if snr != 'inf':
                x = awgn_batch(x, snr)
            xval_cuda = x.cuda()
            Rs, Hhat_val, Rb, mu, logvar, zall= model(xval_cuda)                
            Rxperm = Hhat_val@Rs.permute(1,2,0,3,4)@Hhat_val.transpose(-1,-2).conj() + Rb
            shatperm = Rs.permute(1,2,0,3,4)@Hhat_val.conj().transpose(-1,-2)\
                    @Rxperm.inverse()@xval_cuda.permute(2,3,0,1)[...,None]
            shat = shatperm.permute(2,0,1,3,4).squeeze().cpu().abs()
            for ind in range(x.shape[0]):
                hh = Hhat_val[ind]
                av_hcorr.append(h_corr_cuda(hh, h[ind].cuda()).cpu().item())
                av_scorr.append(s_corr_cuda(s[ind:ind+1].abs().cuda(), \
                    shat[ind:ind+1].cuda()).cpu().item())
        hall.append(sum(av_hcorr)/len(av_hcorr))
        sall.append(sum(av_scorr)/len(av_scorr))
print(hall, sall)
print('done')
print('End date time ', datetime.now())