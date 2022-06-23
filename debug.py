#%%v4 
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

rid = 'v6'
fig_loc = '../data/data_ss/figures/'
mod_loc = '../data/data_ss/models/'
if not(os.path.isdir(fig_loc + f'/{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'{rid}/')
    os.mkdir(mod_loc + f'{rid}/')
fig_loc = fig_loc + f'{rid}/'
mod_loc = mod_loc + f'{rid}/'


#%%
M = 3
dicts = sio.loadmat('../data/nem_ss/v2.mat')
v0 = dicts['v'][..., 0]
v1 = dicts['v'][..., 1]
from skimage.transform import resize
v0 = torch.tensor(resize(v0, (100, 100), preserve_range=True))
v0 = awgn(v0, snr=30, seed=0).abs().to(torch.cfloat)
plt.imshow(v0.abs())
plt.colorbar()
v1 = torch.tensor(resize(v1, (100, 100), preserve_range=True))
v1 = awgn(v1, snr=30, seed=0).abs().to(torch.cfloat)

snr=0; n_data=int(1.1e4)
delta = v0.mean()*10**(-snr/10)
angs = (torch.rand(n_data,1)*20 +10)/180*np.pi  # signal aoa [10, 30]
h = (1j*angs.sin()@torch.arange(M).to(torch.cfloat)[None,:]).exp() #shape of [n,M]

angs_n1 = (torch.rand(n_data,1)*20 -70)/180*np.pi  # noise aoa [-70, -50]
hs_n1 = (1j*angs_n1.sin()@torch.arange(M).to(torch.cfloat)[None,:]).exp() #shape of [n,M]

angs_n2 = (torch.rand(n_data,1)*20 +120)/180*np.pi  # noise aoa [120, 140]
hs_n2 = (1j*angs_n2.sin()@torch.arange(M).to(torch.cfloat)[None,:]).exp() #shape of [n,M]

signal = (h[..., None]@(torch.randn(v0.shape, dtype=torch.cfloat)*(v0**0.5)).flatten()[None,:]).reshape(n_data, M, 100, 100)
# n1 = (hs_n1[..., None]@(torch.randn(v1.shape, dtype=torch.cfloat)*(v1**0.5)).flatten()[None,:]).reshape(n_data, M, 100, 100)
n1 = hs_n1[...,None] @ torch.randn(1, torch.tensor(v0.shape).prod(), dtype=torch.cfloat)*delta**0.5 
n2 = hs_n2[...,None] @ torch.randn(1, torch.tensor(v0.shape).prod(), dtype=torch.cfloat)*delta**0.5
mix =  signal + n1.reshape(n_data, M, 100, 100) + n2.reshape(n_data, M, 100, 100)   
mix_all, sig_all = mix.permute(0,2,3,1), signal.permute(0,2,3,1)

# torch.save((mix, sig, h), 'toy_matrix_inv.pt') # generate data is faster than loading it...
plt.figure()
plt.imshow(mix_all[0,:,:,0].abs())
plt.colorbar()

if False: # check data low rank or not
    for i in range(n_data):
        x = mix[i,:,:].reshape(10000, 3)
        xbar = x - x.mean(0)
        cov = x.conj().t() @ x
        r = torch.linalg.matrix_rank(cov)
        if r != 3:
            print('low rank', i, 'rank is ', r)

#%% load data and model
class DOA(nn.Module):
    def __init__(self):
        super().__init__()
        self.rx_net = nn.Sequential(
            nn.Linear(18,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,18),
            nn.ReLU(inplace=True),
            nn.Linear(18,18)
        )
    def forward(self, x):
        rx_inv = self.rx_net(x)

        return rx_inv 
model = DOA().cuda()

#%%
h_all, M = h, sig_all.shape[-1]
n_tr = int(1e4)
const_range = torch.arange(M).to(torch.cfloat)[None,:].cuda()
data = Data.TensorDataset(mix_all[:n_tr], sig_all[:n_tr], h_all[:n_tr])
val0 = mix_all[n_tr:n_tr+200]
rx_val = (val0[...,None] @ val0[:,:,:,None,:].conj()).mean(dim=(1,2))
rx_val_cuda = rx_val.cuda()
inp_val = torch.stack((rx_val.real, rx_val.imag), dim=1).reshape(rx_val.shape[0], -1).cuda()
tr = Data.DataLoader(data, batch_size=32, drop_last=True, shuffle=True)

Is = torch.stack([torch.eye(3,3)]*32, dim=0).to(torch.cfloat).cuda()
Is2 = torch.stack([torch.eye(3,3)]*200, dim=0).to(torch.cfloat).cuda()

loss_all, loss_val_all = [], []

for epoch in range(0, 2001, 20):
    model = torch.load(mod_loc+f'model_epoch{epoch}.pt')
    with torch.no_grad():
        rx_raw = model(inp_val)
        rx_reshape = rx_raw.reshape(inp_val.shape[0],M,M,2)
        rx_inv_hat = rx_reshape[...,0] + 1j*rx_reshape[...,1]
        rx_inv_hat = (rx_inv_hat + rx_inv_hat.conj().permute(0,2,1))/2
        loss_val = ((Is2-rx_inv_hat@rx_val_cuda).abs()**2).mean()
        loss_val_all.append(loss_val.cpu().item())
        print('epoch', epoch, rx_inv_hat[:3]@rx_val_cuda[:3])
        plt.figure()
        plt.plot(loss_val_all, '-x')
        plt.title(f'val loss at epoch {epoch}')
        # plt.savefig(fig_loc + f'Epoch{epoch}_validation_loss')

        plt.figure()
        plt.plot(loss_val_all[-50:], '-rx')
        plt.title(f'last 50 val loss epoch {epoch}')
        # plt.savefig(fig_loc + f'last_50val_at_epoch{epoch}')
        plt.show()
        plt.close('all')           

print('done')


