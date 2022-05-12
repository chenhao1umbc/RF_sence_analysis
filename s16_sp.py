#%% s16_sp
rid = 's16_sp' # running id
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
torch.autograd.set_detect_anomaly(True)
from vae_model import *
class NN_upconv(nn.Module):
    """This is recursive Wiener filter version, with Rb threshold of [1e-3, 1e2]
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M=3, K=3, im_size=100):
        super().__init__()
        self.dz = 32
        self.K, self.M = K, M
        # Estimate H 
        self.v_net = nn.Sequential(
            DoubleConv(in_channels=M*2, out_channels=32),
            DoubleConv(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=4),
            ) 
        self.v_out = OutConv(in_channels=4, out_channels=1)
        self.hb_net = nn.Sequential(
            Down(in_channels=1, out_channels=32),
            Down(in_channels=32, out_channels=16),
            Down(in_channels=16, out_channels=8),
            Reshape(-1, 8*12*12),
            )
        # Estimate H
        self.h_net = nn.Sequential(
            LinearBlock(8*12*12, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 1),
            nn.Tanh()
            )   
        # self.b1 = nn.Linear(100, 1)
        # self.b2 = nn.Linear(100, 1)
           
        # Estimate V using auto encoder
        self.encoder = nn.Sequential(
            Down(in_channels=1, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            Down(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=1),
            )
        self.fc1 = nn.Linear(25*25, 2*self.dz)
        self.decoder = nn.Sequential(
            nn.Linear(self.dz, 25*25),
            Reshape(-1, 1, 25, 25),
            Up_(in_channels=1, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            Up_(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=4),
            OutConv(in_channels=4, out_channels=1),
            ) 
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        batch_size, _, N, F = x.shape
        z_all, v_all, h_all = [], [], []
        I = x.shape[0]
        "Neural nets for H,V"
        shat_all = []
        ch = torch.pi*torch.arange(self.M, device=x.device)
        for i in range(self.K):
            if i == 0:
                inp = x
            else:
                tmp = hj[...,None]@W@inp.permute(2,3,0,1)[...,None]
                inp = inp - tmp.squeeze().permute(2,3,0,1)
                # inp = (inp - tmp.squeeze().permute(2,3,0,1)).detach()

            temp = self.v_net(torch.cat((inp.real, inp.imag), dim=1)).exp() 
            vj = self.v_out(temp).exp() #sigma_s**2 >=0
            vj = threshold(vj, floor=1e-3, ceiling=1e2)  # shape of [I, 1, N, F]
            hb = self.hb_net(vj)
            ang = self.h_net(hb)  # shape of [I,1]
            hj = ((ang@ ch[None,:])*1j).exp() # shape:[I, M]
            h_all.append(hj)

            Rb = (1.4e-3*torch.ones(batch_size, self.M, \
                device=x.device)).diag_embed().to(torch.cfloat) # shape:[I, M, M]

            "Wienter filter to get coarse shat"
            Rx = hj[...,None] @ hj[:,None].conj() + Rb # shape of [I,M,M]
            W = hj[:, None,].conj() @ Rx.inverse()  # shape of [N,F,I,1,M]
            shat = (W @ inp.permute(2,3,0,1)[...,None]).squeeze().permute(2,0,1) #[I, N, F]
            shat = shat/shat.detach().abs().max()
            shat_all.append(shat)
        
        for i in range(self.K):
            "Encoder"
            xx = self.encoder(shat_all[i][:,None].abs())
            "Get latent variable"
            zz = self.fc1(xx.reshape(batch_size,-1))
            mu = zz[:,::2]
            logvar = zz[:,1::2]
            z = self.reparameterize(mu, logvar)
            z_all.append(z)
            
            "Decoder to get V"
            v = self.decoder(z).exp()
            v_all.append(threshold(v, floor=1e-3, ceiling=1e2)) # 1e-3 to 1e2
        Hhat = torch.stack(h_all, 2) # shape:[I, M, K]
        vhat = torch.stack(v_all, 4).squeeze().to(torch.cfloat) # shape:[I, N, F, K]

        return vhat.diag_embed(), Hhat, Rb, mu, logvar

def loss_fun(x, Rs, Hhat, Rb, mu, logvar, beta=1e-3):
    x = x.permute(0,2,3,1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb
    Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
    try:
        ll = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze() 
    except:
        torch.save((x, Rx, Rs, Hhat, Rb), f'rid{rid}x_Rx_Rs_Hhat_Rb.pt')
        print('error happpened, data saved and stop')
        ll = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze()
    return -ll.sum().real, beta*kl


#%% EM 
xval, _ , hgt0 = torch.load('../data/nem_ss/val500M3FT100_xsh.pt')
hgt = torch.tensor(hgt0).to(torch.cfloat)
xval_cuda = xval[:128].to(torch.cfloat).cuda()
#%% get angles
if False:
    _, Hhat0, _, _ = em_func(xval[0].permute(1,2,0), show_plot=True)
    _, Hhat1, _, _ = em_func(xval[10].permute(1,2,0), show_plot=True)
    _, Hhat2, _, _ = em_func(xval[15].permute(1,2,0), show_plot=True)
    print(h_corr(hgt, Hhat0)) #3,2,1
    print(h_corr(hgt, Hhat1)) #1,2,3
    print(h_corr(hgt, Hhat2)) #3,1,2

    h0 = Hhat0.angle() - Hhat0.angle()[0,:]
    h1 = Hhat1.angle() - Hhat1.angle()[0,:]
    h2 = Hhat2.angle() - Hhat2.angle()[0,:]
    def back2range(hh):
        h = hh.clone()
        ind = h>np.pi
        h[ind] = h[ind] - np.pi*2
        ind = h<-np.pi
        h[ind] = h[ind] + np.pi*2
        return h
    hh0 = back2range(h0)
    hh1 = back2range(h1)
    hh2 = back2range(h2)

    "process the sequeence order"
    temp = hh0.clone()
    hh0[:,0], hh0[:,2] = temp[:,2], temp[:,0]
    temp = hh2.clone()
    hh2[:,0], hh2[:,1], hh2[:,2] = temp[:,1], temp[:,2], temp[:,0]
    torch.save((hh0, hh1, hh2), 'hh012.pt')

#%%
torch.autograd.set_detect_anomaly(True)
hh0, hh1, hh2 = torch.load('hh012.pt')
H = (torch.stack((hh0, hh1, hh2), dim=0)*1j).exp().to(torch.cfloat).cuda()
hh = H.angle().mean(dim=0)
HH = (torch.stack((hh,)*64, dim=0)*1j).exp().to(torch.cfloat).cuda()

model = NN_upconv().cuda()
for w in model.parameters():
    nn.init.normal_(w, mean=0., std=0.01)

xcuda = torch.stack((xval_cuda[0], xval_cuda[10], xval_cuda[15]), dim=0)
lf = nn.MSELoss()
optimizer = RAdam(model.parameters(),
                lr= 1e-4,
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)
data = Data.TensorDataset(xval_cuda)
tr = Data.DataLoader(data, batch_size=64, shuffle=True)
l_all = []
for epoch in range(801):
    for i, (x,) in enumerate(tr): 
        optimizer.zero_grad()  
        Rs, Hhat, Rb, mu, logvar = model(x)
        loss = ((Hhat - HH).abs()**2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        torch.cuda.empty_cache()
        l_all.append(loss.detach().cpu().item())

    if epoch%50 == 0:
        plt.figure()
        plt.plot(l_all, '-or')
        plt.title(f'Loss fuction at epoch{epoch}')
        plt.savefig(f'./files/Epoch{epoch}_LossFunAll')
        plt.figure()
        plt.plot(l_all[-50:], '-or')
        plt.title(f'last 50 at epoch{epoch}')
        plt.savefig(f'./files/Epoch{epoch}_last50')
        plt.show()

        torch.save(model, 'EM4model.pt')
        plt.close('all')

print('done')


#%%
fig_loc = '../data/nem_ss/figures/'
mod_loc = '../data/nem_ss/models/'
if not(os.path.isdir(fig_loc + f'/{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'{rid}/')
    os.mkdir(mod_loc + f'{rid}/')
fig_loc = fig_loc + f'{rid}/'
mod_loc = mod_loc + f'{rid}/'

I = 3000 # how many samples
M, N, F, K = 3, 100, 100, 3
NF = N*F
eps = 5e-4
opts = {}
opts['batch_size'] = 64
opts['lr'] = 1e-3
opts['n_epochs'] = 2000

d = torch.load('../data/nem_ss/tr3kM3FT100.pt')
d = awgn_batch(d, snr=30, seed=1)
xtr = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]) # [sample,M,N,F]
xtr = xtr.to(torch.cfloat)
data = Data.TensorDataset(xtr[:I])
tr = Data.DataLoader(data, batch_size=opts['batch_size'], shuffle=True, drop_last=True)
xval, _ , hgt0 = torch.load('../data/nem_ss/val500M3FT100_xsh.pt')
hgt = torch.tensor(hgt0).to(torch.cfloat).cuda()
xval_cuda = xval[:128].to(torch.cfloat).cuda()

loss_iter, loss_tr, loss1, loss2, loss_eval = [], [], [], [], []
NN = NN_upconv
model = NN(M,K,N).cuda()
# for w in model.parameters():
#     nn.init.normal_(w, mean=0., std=0.01)

optimizer = RAdam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)

for epoch in range(opts['n_epochs']):
    model.train()
    for i, (x,) in enumerate(tr): 
        x = x.cuda()
        optimizer.zero_grad()         
        Rs, Hhat, Rb, mu, logvar= model(x)
        l1, l2 = loss_fun(x, Rs, Hhat, Rb, mu, logvar)
        loss = l1 + l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        torch.cuda.empty_cache()
 
    loss_tr.append(loss.detach().cpu().item()/opts['batch_size'])
    loss1.append(l1.detach().cpu().item()/opts['batch_size'])
    loss2.append(l2.detach().cpu().item()/opts['batch_size'])
    if epoch%10 == 0:
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
            Rs, Hhat, Rb, mu, logvar= model(xval_cuda)
            l1, l2 = loss_fun(xval_cuda, Rs, Hhat, Rb, mu, logvar)
            loss_eval.append((l1+l2).cpu().item()/128)
            plt.figure()
            plt.plot(loss_eval, '-xb')
            plt.title(f'Accumulated validation loss at epoch{epoch}')
            plt.savefig(fig_loc + f'Epoch{epoch}_val') 

            hh, rs0= Hhat[0], Rs[0]
            Rx = hh @ rs0 @ hh.conj().t() + Rb[0]
            shat = (rs0 @ hh.conj().t() @ Rx.inverse()@x.permute(0,2,3,1)[0,:,:,:, None]).cpu() 
            print(f'epoch{epoch} h_corr is ', h_corr(hh.cpu(), torch.tensor(hgt0)), '\n')
            for ii in range(K):
                plt.figure()
                plt.imshow(shat[:,:,ii,0].abs())
                plt.colorbar()
                plt.title(f'Epoch{epoch}_estimated sources-{ii}')
                plt.savefig(fig_loc + f'Epoch{epoch}_estimated sources-{ii}')

                plt.figure()
                plt.imshow(rs0[:,:,ii, ii].abs().cpu())
                plt.colorbar()
                plt.title(f'Epoch{epoch}_estimated V-{ii}')
                plt.savefig(fig_loc + f'Epoch{epoch}_estimated V-{ii}')
                
            plt.close('all')           
        torch.save(model, mod_loc+f'model_epoch{epoch}.pt')
print('done')



#%%