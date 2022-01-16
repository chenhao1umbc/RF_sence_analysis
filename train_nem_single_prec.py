#%%
# #@title rid160100 based on rid150000 for 6 classes
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"   # specify GPU to use
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.float)
from unet.unet_model import UNetHalf8to100_vjto1_6 as UNetHalf
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)

rid = 0 # running id
fig_loc = '/media/ssd2/home/sjkim/hao_mixture/figures/'
mod_loc = '/media/ssd2/home/sjkim/hao_mixture/models/'
if not(os.path.isdir(fig_loc + f'/rid{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'rid{rid}/')
    os.mkdir(mod_loc + f'rid{rid}/')
fig_loc = fig_loc + f'rid{rid}/'
mod_loc = mod_loc + f'rid{rid}/'

I = 3000 # how many samples
M, N, F, J = 6, 100, 100, 6  # M: channels N: time points F: freq. J: number of srcs
NF = N*F
eps = 5e-4  # EM tolerance
opts = {}
opts['n_ch'] = [2,1]  # input has 2 channels (gamma1,gamma2), output has 1 channel
opts['batch_size'] = 24
opts['EM_iter'] = 10
opts['lr'] = 0.001   # learning rate
opts['n_epochs'] = 71
opts['d_gamma'] = 8   # 8 x 8 
n = 5  # number of averaging to check convergence (for stopping)

d = torch.load('/home/chenhao1/Hpython/data/nem_ss/tr3kM6FT100.pt')
xtr = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]).permute(0,2,3,1).to(torch.cfloat)# [sample, N, F, channel]
data = Data.TensorDataset(xtr)
tr = Data.DataLoader(data, batch_size=opts['batch_size'], drop_last=True)
# from skimage.transform import resize
# gtr = torch.tensor(resize(xtr[...,0].abs(), [I,opts['d_gamma'],opts['d_gamma']],\
#     order=1, preserve_range=True ))
# gtr = gtr/gtr.amax(dim=[1,2])[...,None,None]  #standardization 
# gtr = torch.cat([gtr[:,None] for j in range(J)], dim=1)[:,:,None] # shape of [I,J,1,8,8]
"gamma1"
gtr = torch.load('/home/chenhao1/Hpython/data/nem_ss/gtr_c6_IJ188.pt').to(torch.float) # shape of [I,J,1,8,8] 

loss_iter, loss_tr = [], []
model = UNetHalf(opts['n_ch'][0], opts['n_ch'][1]).cuda()
optimizer = optim.RAdam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)
"initialize"
vtr = torch.randn(N, F, J).abs().to(torch.cfloat).repeat(I, 1, 1, 1) # output v initialized
# Htr = torch.randn(M, J).to(torch.cdouble).repeat(I, 1, 1)
Hhat = torch.randn(M, J).to(torch.cfloat).cuda()
Rbtr = torch.ones(I, M).diag_embed().to(torch.cfloat)*100

"gamma2"
lb = torch.load('/home/chenhao1/Hpython/data/nem_ss/lb_c6_J188.pt').to(torch.float) # shape of [J,1,8,8], cpu()
lb = lb.repeat(opts['batch_size'], 1, 1, 1, 1).cuda()

for epoch in range(opts['n_epochs']):    
       
    for i, (x,) in enumerate(tr): # gamma [n_batch, 4, 4]

        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        #%% EM part
        # Hhat = Htr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        vhat = vtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()        
        Rb = Rbtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        g = gtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda().requires_grad_()

        x = x.cuda()
        optim_gamma = torch.optim.SGD([g], lr=0.001)
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
                outs.append(model(torch.cat((g[:,j], lb[:,j]), dim=1)))
            out = torch.cat(outs, dim=1).permute(0,2,3,1)
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
            vtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = vhat.cpu()
            # Htr[i*opts['batch_size']:(i+1)*opts['batch_size']] = Hhat.cpu()
            Rbtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = Rb.cpu()
        
        # update NN paramters
        g.requires_grad_(False)
        model.train()
        for param in model.parameters():
            param.requires_grad_(True)

        outs = []
        for j in range(J):
            outs.append(model(torch.cat((g[:,j], lb[:,j]), dim=1)))
        out = torch.cat(outs, dim=1).permute(0,2,3,1)
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

    if epoch > 10 :
        s1, s2 = sum(loss_tr[-n*2:-n])/n, sum(loss_tr[-n:])/n
        if s1 - s2 < 0 :
            print('break-1')
            break
        print(f'{epoch}-abs((s1-s2)/s1):', abs((s1-s2)/s1))
        if abs((s1-s2)/s1) < 5e-4 :
            print('break-2')
            break

# %%
