#%%
from PIL.Image import FASTOCTREE
from numpy import broadcast_arrays
from numpy.core.fromnumeric import nonzero
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)

#%%
#@title rid170000 based on rid160100 for 6 classes
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)
from unet.unet_model import UNetHalf8to100_vjto1_6 as UNetHalf
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)

def batch_process(model, g, lb, idx):
    """This function try to process forward pass in a batch way

    Args:
        model (object): [Neural network]
        g (gamma): [shape of [I,J,1,8,8]]
        lb (regularizer): [shape of[I,J,1,8,8]]
        idx (which g,lb to use): [shape of [?,2]]
    """
    G = g[idx[:,0], idx[:,1]]  # shape of [?<I,?<J,1, 8, 8]
    L = lb[idx[:,0], idx[:,1]]
    inputs = torch.cat((G, L), dim=2).reshape(-1, 2, L.shape[-2], L.shape[-1])
    outs = []
    bs, i = 30, 0  # batch size
    while i*bs < inputs.shape[0]:
        outs.append(model(inputs[i*bs:i*bs+bs]).squeeze())
        i += 1 
    res = torch.cat(outs).reshape(G.shape[0], G.shape[1],100,100)
    return res

rid = 170000 # running id
fig_loc = '../data/nem_ss/figures/'
mod_loc = '../data/nem_ss/models/'
if not(os.path.isdir(fig_loc + f'/rid{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'rid{rid}/')
    os.mkdir(mod_loc + f'rid{rid}/')
fig_loc = fig_loc + f'rid{rid}/'
mod_loc = mod_loc + f'rid{rid}/'

I = 2850 # how many samples
M, N, F, J = 6, 100, 100, 6
NF = N*F
eps = 5e-4
opts = {}
opts['n_ch'] = [2,1]  
opts['batch_size'] = 50
opts['EM_iter'] = 5
opts['lr'] = 0.001
opts['n_epochs'] = 71
opts['d_gamma'] = 8 
n = 5  # how many samples to average for stopping 

d, lb = torch.load('../data/nem_ss/weakly50percomb_tr3kM6FT100_xlb.pt')
xtr = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]).permute(0,2,3,1)# [sample, N, F, channel]
lbs = torch.zeros(I, J)
for i, v in enumerate(lb):
    lbs[i*50:(i+1)*50, v] = 1
"shuffle data"
ind = torch.randperm(I)
xtr, lbs = xtr[ind], lbs[ind]
data = Data.TensorDataset(xtr, lbs)
tr = Data.DataLoader(data, batch_size=opts['batch_size'], drop_last=True)

from skimage.transform import resize
gtr = torch.tensor(resize(xtr[...,0].abs(), [I,opts['d_gamma'],opts['d_gamma']],\
    order=1, preserve_range=True ))
gtr = gtr/gtr.amax(dim=[1,2])[...,None,None]  #standardization 
gtr = torch.cat([gtr[:,None] for j in range(J)], dim=1)[:,:,None] # shape of [I,J,1,8,8]

loss_iter, loss_tr = [], []
model = UNetHalf(opts['n_ch'][0], opts['n_ch'][1]).cuda()
optimizer = optim.RAdam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)
"initial"
vtr = torch.randn(N, F, J).abs().to(torch.cdouble).repeat(I, 1, 1, 1)
# Htr = torch.ones(M, J).to(torch.cdouble).repeat(I, 1, 1)
Hhat = torch.ones(M, J).to(torch.cdouble).cuda()
Rbtr = torch.ones(I, M).diag_embed().to(torch.cdouble)*100
lb = torch.load('../data/nem_ss/lb_c6_J188.pt') # shape of [J,1,8,8], cpu()
lb = lb.repeat(opts['batch_size'], 1, 1, 1, 1).cuda()

for epoch in range(opts['n_epochs']):    
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    for i, (x, y) in enumerate(tr): # x is data, y is label
        # Hhat = Htr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        vhat = vtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()        
        Rb = Rbtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        g = gtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda().requires_grad_()

        x = x.cuda()
        optim_gamma = torch.optim.SGD([g], lr=0.001)
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((1,2))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat@Rs.permute(1,2,0,3,4)@Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        ll_traj, s = [], y.sum() # s means how many components in one batch
        idx = torch.nonzero(y)

        for ii in range(opts['EM_iter']):
            "E-step"
            W = Rs.permute(1,2,0,3,4)@Hhat.transpose(-1,-2).conj()@Rx.inverse()  #shape of [N,F,I,J,M]
            shat = W.permute(2,0,1,3,4) @ x[...,None]
            Rsshatnf = shat @ shat.transpose(-1,-2).conj() + Rs - \
                (W@Hhat@Rs.permute(1,2,0,3,4)).permute(2,0,1,3,4)
            Rsshat = Rsshatnf.sum([1,2])/NF # shape of [I, J, J]
            Rxshat = (x[..., None] @ shat.transpose(-1,-2).conj()).sum((1,2))/NF #shape of [I, M, J]

            "M-step"
            Hhat = Rxshat @ Rsshat.inverse() # shape of [I, M, J]
            Rb = Rxxhat - Hhat@Rxshat.transpose(-1,-2).conj() - \
                Rxshat@Hhat.transpose(-1,-2).conj() + Hhat@Rsshat@Hhat.transpose(-1,-2).conj()
            Rb = Rb.diagonal(dim1=-1, dim2=-2).diag_embed()
            Rb.imag = Rb.imag - Rb.imag

            "calculate vj"
            out = torch.ones(vhat.shape, device=vhat.device)*1e-20
            out[idx[:,0],:,:,idx[:,1]] = batch_process(model, g, lb, idx)[:,0]
            vhat.real = threshold(out)
            
            loss = loss_func(vhat, Rsshatnf.cuda())
            optim_gamma.zero_grad()   
            loss.backward()
            torch.nn.utils.clip_grad_norm_([g], max_norm=1)
            optim_gamma.step()
            torch.cuda.empty_cache()
            
            "compute log-likelyhood"
            vhat = vhat.detach()
            ll, Rs, Rx = log_lh(x, vhat, Hhat, Rb)
            ll_traj.append(ll.detach().item()/s)
            if torch.isnan(ll_traj[-1]) : input('nan happened')
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

        out = torch.ones(vhat.shape, device=vhat.device)*1e-20
        out[idx[:,0],:,:,idx[:,1]] = batch_process(model, g, lb, idx)[:,0]
        vhat.real = threshold(out)
        optimizer.zero_grad()         
        loss = loss_func(vhat, Rsshatnf.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        torch.cuda.empty_cache()
        loss_iter.append(loss.detach().cpu().item()/s)

    print(f'done with epoch{epoch}')
    plt.figure()
    plt.plot(loss_iter, '-xr')
    plt.title(f'Loss fuction of all the iterations at epoch{epoch}')
    plt.savefig(fig_loc + f'id{rid}_LossFunAll_epoch{epoch}')

    loss_tr.append(loss.detach().cpu().item()/s)
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