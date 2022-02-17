#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)

def loss_vae(x, x_hat, z, mu, logvar, beta=0.5):
    """This is a regular beta-vae loss

    Args:
        x (input data): [I, ?]
        x_hat (reconstructed data): shape of [I, ?]
        z (lattent variable): shape of [I, n_enmbeding]
        mu (mean): shape of [I]
        logvar (log of variance): shape of [I]
        beta (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = ((x-x_hat).abs()**2).sum() - beta*kl
    return loss

#%%
from vae_model import VAE1 as VAE
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
opts['lr'] = 0.001
opts['n_epochs'] = 10 

d = torch.load('/home/chenhao1/Hpython/data/nem_ss/tr3kM3FT100.pt')
xtr = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]).permute(0,2,3,1).to(torch.cfloat)# [sample, N, F, channel]
data = Data.TensorDataset(xtr)
tr = Data.DataLoader(data, batch_size=opts['batch_size'], drop_last=True)

loss_iter, loss_tr = [], []
model = VAE().cuda()
optimizer = optim.RAdam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)
"initialize"
Hhat = torch.randn(M, J).to(torch.cfloat).cuda()
Rbtr = torch.ones(I, M).diag_embed().to(torch.cfloat)*100

for epoch in range(opts['n_epochs']):    
    for i, (x,) in enumerate(tr): # gamma [n_batch, 4, 4]
        
        optimizer.zero_grad()         
        x_hat, z, mu, logvar = model(x)
        loss = loss_vae(x, x_hat, z, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        torch.cuda.empty_cache()
        loss_iter.append(loss.detach().cpu().item())

    print(f'done with epoch{epoch}')
    plt.figure()
    plt.plot(loss_iter, '-xr')
    plt.title(f'Loss fuction of all the iterations at epoch{epoch}')
    plt.savefig(fig_loc + f'id{rid}_LossFunAll_epoch{epoch}')

    loss_tr.append(loss.detach().cpu().item())
    plt.figure()
    plt.plot(loss_tr, '-or')
    plt.title(f'Loss fuction at epoch{epoch}')
    plt.savefig(fig_loc + f'id{rid}_LossFun_epoch{epoch}')

    plt.close('all')  # to avoid warnings
    torch.save(loss_tr, mod_loc +f'loss_rid{rid}.pt')
    torch.save(model, mod_loc +f'model_rid{rid}_{epoch}.pt')
    # torch.save(Hhat, mod_loc +f'Hhat_rid{rid}_{epoch}.pt')    

# %%
