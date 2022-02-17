#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from unet.unet_model import UNetHalf8to100_vjto1_5 as UNetHalf
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)

#%%
from vae_model import VAE1 as VAE
model = VAE().cuda()
x = torch.rand(32,3,100,100).cuda()
b = model(x)
print('Finishing time', datetime.now())

#%%
from vae_model import VAE2 as VAE
model = VAE().cuda()
x = torch.rand(32,3*100*100).cuda()
b = model(x)
print('Finishing time', datetime.now())