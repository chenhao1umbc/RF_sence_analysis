# #%%
# from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# plt.rcParams['figure.dpi'] = 150
# torch.set_printoptions(linewidth=160)
# torch.set_default_dtype(torch.double)
# from skimage.transform import resize
# import itertools
# import time

#@title rid180120, M=3, same random for all samples, gamma with inner loop
#%% this is hybrid precision, float32 for neural network, float64 for EM
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from unet.unet_model import UNetHalf8to100_vjto1_5 as UNetHalf
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)

#%%
rid = 180120 # running id
fig_loc = '../data/nem_ss/figures/'
mod_loc = '../data/nem_ss/models/'
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
opts['n_ch'] = [1,1]  
opts['batch_size'] = 32
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
optimizer = optim.RAdam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)
"initial"
vtr = torch.ones(I, N, F, J).abs().to(torch.cdouble)
# Hhat = torch.randn(M, J).to(torch.cdouble).cuda()
Hhat = torch.load('../data/nem_ss/HCinit_hhat_M3_FT100.pt').to(torch.cdouble).cuda()
Rbtr = torch.ones(I, M).diag_embed().to(torch.cdouble)*1e-3
gtr = torch.rand(I,J,1,opts['d_gamma'], opts['d_gamma'])

#%%
for epoch in range(opts['n_epochs']):    
    for i, (x,) in enumerate(tr): # gamma [n_batch, 4, 4]
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
        #%% EM part
        vhat = vtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()        
        Rb = Rbtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        g = gtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda().requires_grad_()

        x = x.cuda()
        optim_gamma = torch.optim.SGD([g], lr=0.01)
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
            l0, lpre = 0, 0
            torch.manual_seed(0)
            m1, m2 = torch.rand(8,100).cuda(), torch.randn(8,100).cuda()
            rec = []
            for ig in range(100):
                # outs = []
                # for j in range(J):
                    # outs.append(model(g[:,j]))
                # out = torch.cat(outs, dim=1).permute(0,2,3,1).to(torch.double)
                t = g@m1
                out = (t[:,:,0].permute(0,1,3,2)@m2).permute(0,2,3,1)
                vhat = vhat.detach()
                vhat.real = threshold(out)
                loss = loss_func(vhat, Rsshatnf.cuda())
                optim_gamma.zero_grad()   
                loss.backward()
                rec.append(loss.detach().item())
                if ig == 0:
                    l0 = loss.detach().item()
                    lpre = l0
                else:
                    curr = loss.detach().item()
                    if (curr/l0).real < 5e-4 :
                        print(f'gamma break at {ig}, due to loss decreasing slow')
                        break
                    # elif curr.real>lpre.real:
                    #     print(f'gamma break at {ig}, , due to loss increasing')
                    #     break
                    print('diff ratio', lpre.real-curr.real/ curr.real)
                    lpre = curr
                print('g.grad.norm()', g.grad.norm())
                torch.nn.utils.clip_grad_norm_([g], max_norm=10)
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

#%%
#@title pytorch play with gradient clip
torch.manual_seed(1)
a = torch.rand(8, 8)
x = (a.clone() + torch.randn(8,8)*1e-10).requires_grad_()
print(x)

optim = torch.optim.SGD([x], lr=0.1)
l0, lpre = 0, 0
la = []
for i in range(100):
    t = (a-x).to(torch.cdouble)
    loss = (t**2).sum()
    optim.zero_grad()
    loss.backward()
    print('x.grad.norm--', x.grad.norm())
    # if i == 0:
    #     l0 = loss.detach().item()
    #     lpre = l0
    # else:
    #     curr = loss.detach().item()
    #     if curr/l0 < 5e-4 or curr>lpre:
    #         print(f'gamma break at {i}')
    #         break
    #     print('diff ratio', (lpre-curr)/lpre)
    #     lpre = curr
    la.append(loss.detach().cpu().item())
    # if l0/la[0] <5e-4: 
    #     print(i)
    #     break
    torch.nn.utils.clip_grad_norm_([x], max_norm=10)
    print(x.grad.norm(), 'after clip')
    optim.step()
    
plt.plot(la)
