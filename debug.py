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

#%%
xall, sall , hgt = d = torch.load('../data/nem_ss/val500M3FT100_xsh.pt')
h = torch.tensor(hgt)
ind = 0
xi = xall[ind]
si = sall[ind]
print((xi - (h@si.permute(1,2,0)[...,None]).squeeze().permute(2,0,1)).abs().sum()) # should be 0
hh = h.clone()
hh[:,2] = h[:,1].clone()
hh[:,1] = h[:,2].clone()
print((xi - (hh@si.permute(1,2,0)[...,None]).squeeze().permute(2,0,1)).abs().sum()) # should not be 0

#%%
# for j in range(3):
#     if j == 0:
#          inp = xi
#     else:
#         inp = xi - hj[:,None]*W

j = 1
inp = xi
hj = h[:,j]
vj_bar = (si[j].abs()**2).mean()
print('vj bar is ', vj_bar)

for  sigb in [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
    Rx = hj[:,None]@hj[None,:]*vj_bar +sigb*torch.ones(3).diag()
    W = vj_bar*hj[None,:].conj()@Rx.inverse()
    sj = (W@inp.permute(1,2,0)[...,None]).squeeze()
    plt.figure()
    plt.imshow(sj.abs())
    plt.colorbar()
    plt.title('sig_b^2='+str(sigb)+' estimated sj')

plt.figure()
plt.imshow(si[j].abs())
plt.colorbar()
plt.title(f'Source {j+1}')
plt.show()

plt.figure()
plt.imshow(xi[j].abs())
plt.colorbar()
plt.title('Mixture magniture of 1st channel')
plt.show()

Rx = hj[:,None]@hj[None,:]*vj_bar +\
    h[:,2][:,None]@h[:,2][None,:]*(si[2].abs()**2).mean() + \
    h[:,0][:,None]@h[:,0][None,:]*(si[0].abs()**2).mean()
W = vj_bar*hj[None,:]@Rx.inverse()
sj = (W@inp.permute(1,2,0)[...,None]).squeeze()
plt.figure()
plt.imshow(sj.abs())
plt.colorbar()
plt.title('sig_b^2 replaced as \sum hj*vj*hj^H')
plt.show()