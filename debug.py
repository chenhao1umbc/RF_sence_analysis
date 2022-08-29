#%% v86
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())

#%%
"make the result reproducible"
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)       # current GPU seed
torch.cuda.manual_seed_all(seed)   # all GPUs seed
torch.backends.cudnn.deterministic = True  #True uses deterministic alg. for cuda
torch.backends.cudnn.benchmark = False  #False cuda use the fixed alg. for conv, may slower

rid = 'v86' 
fig_loc = '../data/data_ss/figures/'
mod_loc = '../data/data_ss/models/'
if not(os.path.isdir(fig_loc + f'/{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'{rid}/')
    os.mkdir(mod_loc + f'{rid}/')
fig_loc = fig_loc + f'{rid}/'
mod_loc = mod_loc + f'{rid}/'
# torch.autograd.set_detect_anomaly(True)

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

class NNet(nn.Module):
    """This is recursive Wiener filter version, with Rb threshold of [1e-3, 1e2]
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M, K, im_size):
        super().__init__()
        self.dz = 32
        self.J, self.M = K, M
        down_size = int(im_size/4)
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

        
        "Encoder part"
        shat = xj
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
       
        vhat = torch.stack(v_all, 3).to(torch.cfloat) # shape:[I, N, F, J]
        zall = torch.stack(z_all, dim=1)

        # Rb = (b@b.conj().permute(0,1,2,4,3)).mean(dim=(1,2)).squeeze()
        eye = torch.eye(M, device='cuda') 
        Rb = torch.stack(tuple(eye for ii in range(btsize)), 0)*1e-3

        return vhat.squeeze(), Rb, mu, logvar, zall

def loss_fun(x, Rs, Rb, mu, logvar, zall, beta=1):
    I, M, J = x.shape[0], x.shape[1], Rs.shape[-1]
    x = x.squeeze()
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    Rx = Rs.squeeze() + Rb # shape of [I, N, F]
    ll = -(np.pi*Rx.abs()).log() - (x.conj()*(1/Rx)*x)
    return -ll.sum(), beta*kl 

#%%
"raw data processing"
FT = 64  #48, 64, 80, 100, 128, 200, 256
var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
data = {}

def get_ftdata(data_pool):
    *_, Z = stft(data_pool, fs=4e7, nperseg=FT, boundary=None)
    x = torch.tensor(np.roll(Z, FT//2, axis=1))  # roll nperseg//2
    return x.to(torch.cfloat)

for i in range(6):
    temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
    x = torch.tensor(temp['x'])
    x =  x/((x.abs()**2).sum(dim=(1),keepdim=True)**0.5)# normalize
    data[i] = x
s1 = get_ftdata(data[0]) # ble [2000,F,T]
s2 = get_ftdata(data[2]) # fhss1
s3 = get_ftdata(data[5]) # wifi2
s = [s1, s2, s3]

torch.manual_seed(1)
J, M = 3, 1
"training data"
x = []
for i in range(4):
    temp = 0
    for j in range(J):
        idx = torch.randperm(2000)
        temp =s[j][idx]
        x.append(temp)
x = torch.stack(x, dim=1)
xtr = x[:,:9].reshape(-1,1,FT,FT)
d = awgn_batch(xtr, snr=40, seed=1) # added white noise

xvt = x[:,9:].reshape(-1,1,FT,FT)
sval = xvt[:1000]
xval = awgn_batch(sval, snr=40, seed=10)

#%%
I = 18000 # how many samples
M, N, F, J = 1, 64, 64, 1
NF = N*F
eps = 5e-4
opts = {}
opts['batch_size'] = 128
opts['n_epochs'] = 301
opts['lr'] = 1e-3

xtr = (d/d.abs().amax(dim=(1,2,3), keepdim=True)) # [sample,M,N,F]
xtr = xtr.to(torch.cfloat)
data = Data.TensorDataset(xtr[:I])
tr = Data.DataLoader(data, batch_size=opts['batch_size'], shuffle=True, drop_last=True)

sval= sval.permute(0,2,3,1)
xval = xval/xval.abs().amax(dim=(1,2,3), keepdim=True)
data = Data.TensorDataset(xval, sval)
dval = Data.DataLoader(data, batch_size=1000, drop_last=True)

#%%
loss_iter, loss_tr, loss1, loss2, loss_eval = [], [], [], [], []
model = NNet(M,J,N).cuda()
# for w in model.parameters():
#     nn.init.normal_(w, mean=0., std=0.01)

optimizer = torch.optim.RAdam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)

for epoch in range(opts['n_epochs']):
    model.train()
    for i, (x,) in enumerate(tr): 
        x = x.cuda()
        optimizer.zero_grad()         
        Rs, Rb, mu, logvar, zall= model(x)
        l1, l2 = loss_fun(x, Rs, Rb, mu, logvar, zall)
        loss = l1 + l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        optimizer.step()
        torch.cuda.empty_cache()

        if i%30 == 0:
            loss_tr.append(loss.detach().cpu().item()/opts['batch_size'])
            loss1.append(l1.detach().cpu().item()/opts['batch_size'])
            loss2.append(l2.detach().cpu().item()/opts['batch_size'])

    if epoch%5 == 0:
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
            av_hcorr, av_scorr, temp = [], [], []
            for i, (x, s) in enumerate(dval):
                xval_cuda = x.cuda()
                Rs, Rb, mu, logvar, zall= model(xval_cuda)
                l1, l2 = loss_fun(xval_cuda, Rs, Rb, mu, logvar, zall)
                temp.append((l1+l2).cpu().item()/x.shape[0])
                     
                Rx = Rs + Rb
                shatperm = Rs/Rx*xval_cuda.squeeze()
                shat = shatperm[...,None].cpu().abs()
                for ind in range(x.shape[0]):
                    av_scorr.append(s_corr(s[ind].abs(), shat[ind]))
                
                if i == 0:
                    plt.figure()
                    for ind in range(3):
                        for ii in range(J):
                            plt.subplot(3,3,ii+1+ind*3)
                            plt.imshow(shat[ind,:,:,ii])
                            # plt.tight_layout(pad=1.1)
                            # if ii == 0 : plt.title(f'Epoch{epoch}_sample{ind}')
                    plt.savefig(fig_loc + f'Epoch{epoch}_estimated sources')
                    plt.show()
                    plt.close('all')

            loss_eval.append(sum(temp)/len(temp))
            print('first 3 s_corr',av_scorr[:3],' averaged:', sum(av_scorr)/len(av_scorr))

            plt.figure()
            plt.plot(loss_eval[-50:], '-xb')
            plt.title(f'last 50 validation loss at epoch{epoch}')
            plt.savefig(fig_loc + f'Epoch{epoch}_val') 
            plt.close('all') 

        torch.save(model, mod_loc+f'model_epoch{epoch}.pt')
print('done')
print('End date time ', datetime.now())