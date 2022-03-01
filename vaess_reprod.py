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

from vae_model import LinearBlock

class VAE2(nn.Module):
    """This is MLP version  -- ref VAESS
    Input shape [I,MNF], e.g.[32, 3*100*100]"""
    def __init__(self, dimx=30000, K=3):
        super(VAE2, self).__init__()

        self.K = K
        self.dz = 32
        chans = (700, 600, 500, 400, 300)
        # chans = (2560, 2048, 1536, 1024, 512)
        self.encoder = nn.Sequential(
            LinearBlock(dimx, chans[0]),
            LinearBlock(chans[0],chans[1]),
            LinearBlock(chans[1],chans[2]),
            LinearBlock(chans[2],chans[3]),
            LinearBlock(chans[3],chans[4]),
            nn.Linear(chans[4], 2*self.dz*K)
            )
        self.decoder = nn.Sequential(
            LinearBlock(self.dz, chans[4]),
            LinearBlock(chans[4],chans[3]),
            LinearBlock(chans[3],chans[2]),
            LinearBlock(chans[2],chans[1]),
            LinearBlock(chans[1],chans[0]),
            LinearBlock(chans[0],dimx,activation=False),
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        "Encoder and Get latent variable"
        zz = self.encoder(x)
        mu = zz[:,::2]
        logvar = zz[:,1::2]
        z = self.reparameterize(mu, logvar)
        "Decoder"
        sources = self.decoder(z.view(-1,self.dz))
        s = sources.view(-1,self.K, x.shape[-1])
        x_hat = s.sum(1)

        return x_hat, z, mu, logvar, s

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
model = VAE2(784, 1).cuda()
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
        loss = loss_vae(x, x_hat, mu, logvar)
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
