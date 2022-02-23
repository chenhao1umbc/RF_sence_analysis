#%%
from cmath import rect
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)
import pandas as pd

def loss_vae(x, x_hat, z, mu, logvar, beta=1):
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
    loss = ((x-x_hat).abs()**2).sum() + beta*kl
    return loss

#%%
from vae_model import VAE2 as VAE
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
opts['batch_size'] = 128
opts['lr'] = 1e-4
opts['n_epochs'] = 5000
K = 2

#%%
dd = pd.read_csv("../data/mnist_train.csv", delimiter=",", header=None).values
lb, d = torch.from_numpy(dd[:,0]), torch.from_numpy(dd[:,1:])
xtr = (d/d.abs().amax(dim=1, keepdim=True).to(torch.float32)).cuda() # [sample, D)
data = Data.TensorDataset(xtr)
tr = Data.DataLoader(data, batch_size=opts['batch_size'], drop_last=True)

loss_iter, loss_tr = [], []
model = VAE(784, 1).cuda()
optimizer = torch.optim.Adam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)
rec = []
for epoch in range(opts['n_epochs']):    
    for i, (x,) in enumerate(tr): 
        # x = x.cuda()
        optimizer.zero_grad()         
        x_hat, z, mu, logvar, s = model(x)
        loss = loss_vae(x, x_hat, z, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        rec.append(loss.detach().cpu().item())
        optimizer.step()
        torch.cuda.empty_cache()
        if loss.isnan() : print(nan)

    loss_tr.append(loss.detach().cpu().item())
    if epoch%50 == 0:
        plt.figure()
        plt.plot(loss_tr, '-or')
        plt.title(f'Loss fuction at epoch{epoch}')
        plt.show()

        plt.figure()
        plt.plot(rec, '-ob')
        plt.title(f'Loss fuction at epoch{epoch}')
        plt.show()

        plt.figure()
        plt.imshow(x_hat.detach().cpu().abs().reshape(-1,28,28)[0])
        plt.title('first sample of channel 1')
        plt.show()

print('done')
# %%
