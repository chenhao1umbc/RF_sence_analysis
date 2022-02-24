#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)

from vae_model import NN
rid = 0 # running id
fig_loc = '/home/chenhao1/Hpython/data/nem_ss/figures/'
mod_loc = '/home/chenhao1/Hpython/data/nem_ss/models/'
if not(os.path.isdir(fig_loc + f'/rid{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'rid{rid}/')
    os.mkdir(mod_loc + f'rid{rid}/')
fig_loc = fig_loc + f'rid{rid}/'
mod_loc = mod_loc + f'rid{rid}/'

I = 3000 # how many samples
M, N, F, J = 3, 100, 100, 3
NF = N*F
eps = 5e-4
opts = {}
opts['batch_size'] = 64
opts['lr'] = 1e-3
opts['n_epochs'] = 500

d = torch.load('/home/chenhao1/Hpython/data/nem_ss/tr3kM3FT100.pt')
xtr = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]) # [sample,M,N,F]
xtr = xtr.to(torch.cfloat)
data = Data.TensorDataset(xtr)
tr = Data.DataLoader(data, batch_size=opts['batch_size'], drop_last=True)

def loss_fun(x, vhat, Hhat, Rb, mu, logvar, beta=1):
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ll, _, _ = log_likelihood(x.permute(0,2,3,1), \
        vhat.to(torch.cfloat), Hhat, Rb.to(torch.cfloat))
    return -ll+ beta*kl

loss_iter, loss_tr = [], []
model = NN(3,3,100).cuda()
optimizer = torch.optim.Adam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)

for epoch in range(opts['n_epochs']):    
    for i, (x,) in enumerate(tr): 
        x = x.cuda()
        optimizer.zero_grad()         
        vhat, Hhat, Rb, mu, logvar= model(x)
        loss = loss_fun(x, vhat, Hhat, Rb, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        torch.cuda.empty_cache()

    loss_tr.append(loss.detach().cpu().item())
    if epoch%50 == 0:
        plt.figure()
        plt.plot(loss_tr, '-or')
        plt.title(f'Loss fuction at epoch{epoch}')
        plt.show()

        plt.figure()
        plt.plot(loss_tr[-50:], '-or')
        plt.title(f'Last 50 of loss at epoch{epoch}')
        plt.show()

print('done')
# %%
