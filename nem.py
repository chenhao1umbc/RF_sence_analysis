#%%@title 'n1'
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)

rid = 'n1' # running id
fig_loc = '../data/nem_ss/figures/'
mod_loc = '../data/nem_ss/models/'
if not(os.path.isdir(fig_loc + f'/rid{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'rid{rid}/')
    os.mkdir(mod_loc + f'rid{rid}/')
fig_loc = fig_loc + f'rid{rid}/'
mod_loc = mod_loc + f'rid{rid}/'

from unet.unet_model import *
class UNetHalf(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
        """
        super().__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = MyUp(self.n_ch, self.n_ch//2)
        self.up2 = MyUp(self.n_ch//2, self.n_ch//4)
        self.up3 = MyUp(self.n_ch//4, self.n_ch//8)
        self.up4 = MyUp(self.n_ch//8, self.n_ch//16)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        x = self.sig(x)
        out = x/x.detach().amax(keepdim=True, dim=(-1,-2))
        return out

I = 3000 # how many samples
M, N, F, J = 3, 100, 100, 3
NF = N*F
eps, delta, glr = 5e-4, 1, 0.01 # delta is scale for Rb, glr is gamma learning rate
opts = {}
opts['n_ch'] = [1,1]  
opts['batch_size'] = 48
opts['EM_iter'] = 201
opts['lr'] = 0.001
opts['n_epochs'] = 71
opts['d_gamma'] = 8 
n = 5  # for stopping 

d = torch.load('../data/nem_ss/tr3kM3FT100.pt')
xtr = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]).permute(0,2,3,1)# [sample, N, F, channel]
data = Data.TensorDataset(xtr)
tr = Data.DataLoader(data, batch_size=opts['batch_size'], drop_last=True)

loss_iter, loss_tr = [], []
model = UNetHalf(opts['n_ch'][0], opts['n_ch'][1]).cuda()
optimizer = torch.optim.RAdam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)

"initial"
# Hhat = torch.randn(M, J).to(torch.cdouble).cuda()
h = torch.load('../data/nem_ss/HCinit_hhat_M3_FT100.pt')
# _, _ , h = d = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
Htr = torch.tensor(h).to(torch.cdouble).repeat(I,1,1)
Rbtr = torch.ones(I, M).diag_embed().to(torch.cdouble)*delta
gtr0 = torch.load('../data/nem_ss/xx_all_8by8.pt')
gtr0 = gtr0/gtr0.amax(dim=[3,4])[...,None,None]
gtr0 = torch.cat([gtr0 for j in range(J)], dim=1).to(torch.float)
noise = torch.rand(I,J,1,opts['d_gamma'], opts['d_gamma']) # shape of [J,1,8,8], cpu()
gtr = (gtr0 + noise/10).to(torch.float) # 10db snr to resized signal


#%%
#@title gamma does not have inner loop
for epoch in range(opts['n_epochs']):    
    for i, (x,) in enumerate(tr): # gamma [n_batch, 4, 4]
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
        #%% EM part       
        Rb = Rbtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        g = gtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        Hhat = Htr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        outs = []
        for j in range(J):
            outs.append(model(g[:,j]))
        out = torch.cat(outs, dim=1).permute(0,2,3,1).to(torch.double)
        vhat = out.to(torch.cdouble)  # shape of [I,N,F,J]

        x = x.cuda()
        g.requires_grad_()
        optim_gamma = torch.optim.RAdam([g],
                lr= glr,
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((1,2))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        ll_traj = []

        for ii in range(opts['EM_iter']):
            "E-step"
            W = Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() @ Rx.inverse()  # shape of [N, F, I, J, M]
            shat = W.permute(2,0,1,3,4) @ x[...,None]
            Rsshatnf = shat @ shat.transpose(-1,-2).conj() + Rs - (W@Hhat@Rs.permute(1,2,0,3,4)).permute(2,0,1,3,4)
            Rsshat = Rsshatnf.sum([1,2])/NF # shape of [I, J, J]
            Rxshat = (x[..., None] @ shat.transpose(-1,-2).conj()).sum((1,2))/NF # shape of [I, M, J]

            "M-step"
            Hhat = Rxshat @ Rsshat.inverse() # shape of [I, M, J]
            Rb = Rxxhat - Hhat@Rxshat.transpose(-1,-2).conj() - \
                Rxshat@Hhat.transpose(-1,-2).conj() + Hhat@Rsshat@Hhat.transpose(-1,-2).conj()
            Rb = Rb.diagonal(dim1=-1, dim2=-2).diag_embed()
            Rb.imag = Rb.imag - Rb.imag

            # vj = Rsshatnf.diagonal(dim1=-1, dim2=-2)
            # vj.imag = vj.imag - vj.imag
            outs = []
            for j in range(J):
                outs.append(model(g[:,j]))
            out = torch.cat(outs, dim=1).permute(0,2,3,1).to(torch.double)
            vhat.real = threshold(out)
            loss = loss_func(vhat, Rsshatnf.cuda())
            optim_gamma.zero_grad()   
            loss.backward()
            torch.nn.utils.clip_grad_norm_([g], max_norm=1)
            optim_gamma.step()
            torch.cuda.empty_cache()
            
            "compute log-likelyhood"
            vhat = vhat.detach()
            ll, Rs, Rx = log_likelihood(x, vhat, Hhat, Rb)
            ll_traj.append(ll.item())
            if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
            if ii > 5 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3])<eps:
                print(f'EM early stop at iter {ii}, batch {i}, epoch {epoch}')
                break
    
        print(f'batch {i} is done')
        if i == 0 :
            plt.figure()
            plt.plot(ll_traj, '-x')
            plt.title(f'the log-likelihood of the first batch at epoch {epoch}')
            plt.savefig(fig_loc + f'id{rid}_log-likelihood_epoch{epoch}')

            plt.figure()
            plt.imshow(vhat[0,...,0].real.cpu())
            plt.colorbar()
            plt.title(f'1st source of vj in first sample from the first batch at epoch {epoch}')
            plt.savefig(fig_loc + f'id{rid}_vj1_epoch{epoch}')

            plt.figure()
            plt.imshow(vhat[0,...,1].real.cpu())
            plt.colorbar()
            plt.title(f'2nd source of vj in first sample from the first batch at epoch {epoch}')
            plt.savefig(fig_loc + f'id{rid}_vj2_epoch{epoch}')

            plt.figure()
            plt.imshow(vhat[0,...,2].real.cpu())
            plt.colorbar()
            plt.title(f'3rd source of vj in first sample from the first batch at epoch {epoch}')
            plt.savefig(fig_loc + f'id{rid}_vj3_epoch{epoch}')

        #%% update variable
        with torch.no_grad():
            gtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = g.cpu()
            # vtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = vhat.cpu()
            Htr[i*opts['batch_size']:(i+1)*opts['batch_size']] = Hhat.cpu()
            Rbtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = Rb.cpu()
        g.requires_grad_(False)
        model.train()
        for param in model.parameters():
            param.requires_grad_(True)

        outs = []
        for j in range(J):
            outs.append(model(g[:,j]))
        out = torch.cat(outs, dim=1).permute(0,2,3,1).to(torch.double)
        vhat.real = threshold(out)
        optimizer.zero_grad()         
        ll, *_ = log_likelihood(x, vhat, Hhat, Rb)
        loss = -ll
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
    torch.save(Hhat, mod_loc +f'Hhat_rid{rid}_{epoch}.pt')    

    # if epoch >10 :
    #     s1, s2 = sum(loss_tr[-n*2:-n])/n, sum(loss_tr[-n:])/n
    #     if s1 - s2 < 0 :
    #         print('break-1')
    #         break
    #     print(f'{epoch}-abs((s1-s2)/s1):', abs((s1-s2)/s1))
    #     if abs((s1-s2)/s1) < 5e-4 :
    #         print('break-2')
    #         break
print('done')
print('starting date time ', datetime.now())