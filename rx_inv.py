"For now, I need to verify that the covariance inverse works"
#%% 
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

#%%
M, n_data= 3, int(1.1e3)
dicts = sio.loadmat('../data/nem_ss/v2.mat')
v0 = dicts['v'][..., 0]
v1 = dicts['v'][..., 1]
v2 = dicts['v'][..., 2]
from skimage.transform import resize
v0 = torch.tensor(resize(v0, (100, 100), preserve_range=True))
v0 = awgn(v0, snr=30, seed=0).abs().to(torch.cfloat)
plt.imshow(v0.abs())
plt.colorbar()
v1 = torch.tensor(resize(v1, (100, 100), preserve_range=True))
v1 = awgn(v1, snr=30, seed=1).abs().to(torch.cfloat)
v2 = torch.tensor(resize(v2, (100, 100), preserve_range=True))
v2 = awgn(v2, snr=30, seed=2).abs().to(torch.cfloat)

snr=0
delta = v0.mean()*10**(-snr/10)
angs = (torch.rand(n_data,1)*20 +10)/180*np.pi  # signal aoa [10, 30]
H = (1j*angs.sin()@torch.arange(M).to(torch.cfloat)[None,:]).exp() #shape of [n,M]

angs_n1 = (torch.rand(n_data,1)*20 -70)/180*np.pi  # noise aoa [-70, -50]
hs_n1 = (1j*angs_n1.sin()@torch.arange(M).to(torch.cfloat)[None,:]).exp() #shape of [n,M]

angs_n2 = (torch.rand(n_data,1)*20 +120)/180*np.pi  # noise aoa [120, 140]
hs_n2 = (1j*angs_n2.sin()@torch.arange(M).to(torch.cfloat)[None,:]).exp() #shape of [n,M]

signal = (H[..., None]@(torch.randn(v0.shape, dtype=torch.cfloat)*(v0**0.5)).flatten()[None,:])
n1 = (hs_n1[..., None]@(torch.randn(v1.shape, dtype=torch.cfloat)*(v1**0.5)).flatten()[None,:])
n2 = (hs_n2[..., None]@(torch.randn(v2.shape, dtype=torch.cfloat)*(v2**0.5)).flatten()[None,:])
mix =  (signal + n1 + n2).reshape(n_data, M, 100, 100)   
mix_all, sig_all = mix.permute(0,2,3,1), signal.reshape(n_data, M, 100, 100).permute(0,2,3,1)
mixn = awgn_batch(mix_all)
H_all = torch.stack((H, hs_n1, hs_n2), dim=-1)

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
        self.mainnet = nn.Sequential(
            nn.Linear(12,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
        )
        self.hnet = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,6)
        )
        self.rxnet = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 12)
        )
    def forward(self, x):
        m = self.mainnet(x)
        h6 = self.hnet(m)
        rx12 = self.rxnet(m)

        return h6, rx12 

def lower2matrix(rx12):
    rx_inv_hat = torch.zeros(rx12.shape[0], 3, 3, dtype=torch.cfloat).cuda()
    rx_inv_hat[:, ind[0], ind[1]] = rx12[:, :6] + 1j*rx12[:,6:]
    rx_inv_hat = rx_inv_hat + rx_inv_hat.permute(0,2,1).conj()
    rx_inv_hat[0,0], rx_inv_hat[1,1], rx_inv_hat[2,2] = \
        rx_inv_hat[0,0]/2, rx_inv_hat[1,1]/2, rx_inv_hat[2,2]/2
    return rx_inv_hat

model = DOA().cuda()

#%%
M, btsize, n_tr = 3, 48, int(1e2)
lamb = 1
const_range = torch.arange(M).to(torch.cfloat)[None,:].cuda()
data = Data.TensorDataset(mix_all[:n_tr], H_all[:n_tr])
tr = Data.DataLoader(data, batch_size=btsize, drop_last=True, shuffle=True)
optimizer = RAdam(model.parameters(),
                lr= 1e-4,
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)

"Pre calc constants"
Is = torch.stack([torch.eye(3,3)]*btsize, dim=0).to(torch.cfloat).cuda()
Is2 = torch.stack([torch.eye(3,3)]*200, dim=0).to(torch.cfloat).cuda()
ind = torch.tril_indices(3,3)
indx = ind[0]*3+ind[1]

"validation"
val0, h0 = mix_all[n_tr:n_tr+200], H_all[n_tr:n_tr+200].cuda()
rx_val = (val0[...,None] @ val0[:,:,:,None,:].conj()).mean(dim=(1,2))
rx_val_cuda = rx_val.cuda()

loss_all, loss_val_all = [], []
for epoch in range(3):
    for i, (mix, H) in enumerate(tr):
        loss = 0
        optimizer.zero_grad()
        mix, H = mix.cuda(), H.cuda()
        for j in range(3): # recursive for each source
            if j == 0:
                Rx = mix[...,None] @ mix[:,:,:,None,:].conj()
                rx = Rx.mean(dim=(1,2))
            else:
                w = rx_inv_hat@hhat[...,None] / \
                        (hhat[:,None,:].conj()@rx_inv_hat@hhat[...,None])
                p = Is - hhat[...,None]@w.permute(0,2,1).conj()
                rx = p@rx@p.permute(0,2,1).conj()

            inp = torch.stack((rx.real, rx.imag), dim=1)[:,:,ind[0], ind[1]].reshape(btsize, -1)
            h6, rx12 = model(inp)
            hhat = h6[:,:3] + 1j*h6[:,3:]
            rx_inv_hat = lower2matrix(rx12)
            loss = loss + ((Is-rx_inv_hat@rx).abs()**2).mean() + \
                lamb*((H[:,:,j]-hhat).abs()**2).mean()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        torch.cuda.empty_cache()
        if i % 30 == 0:
            loss_all.append(loss.detach().cpu().item())
        
    if epoch % 10 == 0:
        plt.figure()
        plt.plot(loss_all, '-x')
        plt.title(f'epoch {epoch}')
        plt.show()

        if epoch >50:
            plt.figure()
            plt.plot(loss_all[-100:], '-rx')
            plt.title(f'epoch {epoch}')
            plt.show()

        "Validation"
        with torch.no_grad():
            loss_val = 0
            for j in range(3): # recursive for each source
                if j ==0 :
                    rx = rx_val_cuda
                else:
                    w = rx_inv_hat_val@hhat_val[...,None] / \
                            (hhat_val[:,None,:].conj()@rx_inv_hat_val@hhat_val[...,None])
                    p = Is2 - hhat_val[...,None]@w.permute(0,2,1).conj()
                    rx = p@rx@p.permute(0,2,1).conj()

                rl = rx.reshape(200, -1)[:,indx] # take lower triangle
                inp = torch.stack((rl.real, rl.imag), dim=1).reshape(200, -1)
                h6, rx12 = model(inp)
                hhat_val = h6[:,:3] + 1j*h6[:,3:]
                rx_inv_hat_val = lower2matrix(rx12)

            loss_val = loss_val + ((Is2-rx_inv_hat_val@rx).abs()**2).mean() + \
                lamb*((h0[:,:,j]-hhat_val).abs()**2).mean()

            loss_val_all.append(loss_val.cpu().item())
            plt.figure()
            plt.plot(loss_val_all, '-x')
            plt.title(f'val loss at epoch {epoch}')
            plt.show()

print('done')


