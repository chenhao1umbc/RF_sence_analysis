#%% v10500
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

from vae_model import NN5 as NN
def loss_fun(x, Rs, Hhat, Rb, mu, logvar, beta=0.5):
    x = x.permute(0,2,3,1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb
    Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
    ll = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze() 
    return -ll.sum().real + beta*kl

rid = 'v10500' # running id
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
opts['n_epochs'] = 2000

d = torch.load('../data/nem_ss/tr3kM3FT100.pt')
xtr = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]) # [sample,M,N,F]
xtr = xtr.to(torch.cfloat)
data = Data.TensorDataset(xtr)
tr = Data.DataLoader(data, batch_size=opts['batch_size'], shuffle=True, drop_last=True)

# h = torch.load('../data/nem_ss/HCinit_hhat_M3_FT100.pt').to(torch.cdouble).cuda()
_, _ , hgt = d = torch.load('../data/nem_ss/val500M3FT100_xsh.pt')
Hgt = torch.tensor(hgt).to(torch.cfloat).cuda()

loss_iter, loss_tr = [], []
model = NN(M,K,N).cuda()
optimizer = torch.optim.RAdam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)

for epoch in range(opts['n_epochs']):
    for i, (x,) in enumerate(tr): 
        x = x.cuda()
        optimizer.zero_grad()         
        Rs, Hhat, Rb, mu, logvar= model(x)
        loss = loss_fun(x, Rs, Hhat, Rb, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        torch.cuda.empty_cache()
 
    loss_tr.append(loss.detach().cpu().item())
    if epoch%10 == 0:
        plt.figure()
        plt.plot(loss_tr, '-or')
        plt.title(f'Loss fuction at epoch{epoch}')
        plt.savefig(fig_loc + f'Epoch{epoch}_LossFunAll')

        plt.figure()
        plt.plot(loss_tr[-50:], '-or')
        plt.title(f'Last 50 of loss at epoch{epoch}')
        plt.savefig(fig_loc + f'Epoch{epoch}_last50')

        hh = Hhat[0].detach()
        rs0 = Rs[0].detach() 
        Rx = hh @ rs0 @ hh.conj().t() + Rb.detach()[0]
        shat = (rs0 @ hh.conj().t() @ Rx.inverse()@x.permute(0,2,3,1)[0,:,:,:, None]).cpu() 
        for ii in range(K):
            plt.figure()
            plt.imshow(shat[:,:,ii,0].abs())
            plt.title(f'Epoch{epoch}_estimated sources-{ii}')
            plt.savefig(fig_loc + f'Epoch{epoch}_estimated sources-{ii}')
            plt.show()

            plt.figure()
            plt.imshow(rs0[:,:,ii, ii].abs().cpu())
            plt.title(f'Epoch{epoch}_estimated V-{ii}')
            plt.savefig(fig_loc + f'Epoch{epoch}_estimated V-{ii}')
            plt.show()
            plt.close('all')
        print(f'done with epoch{epoch}', h_corr(hh.cpu(), torch.tensor(hgt)))
        torch.save(model, mod_loc+f'modle_epoch{epoch}.pt')
print('done')
# %%
