#@title rid133201 , interpolate, half unet, warm start, warm shared Hhat
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)
from unet.unet_model import UNetHalf8to100_stack3
torch.manual_seed(1)

rid = 133201 # running id
fig_loc = '../data/nem_ss/figures/'
if not(os.path.isdir(fig_loc + f'/rid{rid}/')): 
    print('made a new folder')
    os.mkdir(fig_loc + f'rid{rid}/')
fig_loc = fig_loc + f'rid{rid}/'

I = 3000 # how many samples
M, N, F, J = 3, 100, 100, 3
NF = N*F
eps = 5e-4
opts = {}
opts['n_ch'] = 1  
opts['batch_size'] = 64
opts['EM_iter'] = 201
opts['lr'] = 0.001
opts['n_epochs'] = 71
opts['d_gamma'] = 8

d = torch.load('../data/nem_ss/tr3kM3FT100.pt')
xtr = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]*3).permute(0,2,3,1)# [sample, N, F, channel]
data = Data.TensorDataset(xtr)
tr = Data.DataLoader(data, batch_size=opts['batch_size'], drop_last=True)
gtr = torch.randn(J,1,opts['d_gamma'],opts['d_gamma']).abs().repeat(I,1,1,1,1)
from skimage.transform import resize
# xx_all = torch.tensor(resize(xtr[...,0].abs(), [I,opts['d_gamma'],opts['d_gamma']],\
    # order=1, preserve_range=True ))[:,None, None]
xx_all = torch.load('../data/nem_ss/xx_all_8by8.pt')
loss_iter, loss_tr = [], []
model = UNetHalf8to100_stack3(opts['n_ch'], opts['n_ch']).cuda()
optimizer = optim.RAdam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999), 
                eps=1e-8,
                weight_decay=0)
"initial"
vtr = torch.randn(N, F, J).abs().to(torch.cdouble).repeat(I, 1, 1, 1)
# Htr = torch.randn(M, J).to(torch.cdouble).repeat(I, 1, 1)
Hhat = torch.randn(M, J).to(torch.cdouble).cuda()
Rbtr = torch.ones(I, M).diag_embed().to(torch.cdouble)*100

for epoch in range(opts['n_epochs']):    
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    for i, (x,) in enumerate(tr): # gamma [n_batch, 4, 4]
        #%% EM part
        # Hhat = Htr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        vhat = vtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()        
        Rb = Rbtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        g = gtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda().requires_grad_()

        x = x.cuda()
        inp = torch.rand(opts['batch_size'], J, 1, opts['d_gamma'],opts['d_gamma']*2).cuda()
        inp[...,0::2] = xx_all[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
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
            # Hhat = (Rxshat @ Rsshat.inverse()).mean(0) # shape of [M, J]
            Rb = Rxxhat - Hhat@Rxshat.transpose(-1,-2).conj() - \
                Rxshat@Hhat.transpose(-1,-2).conj() + Hhat@Rsshat@Hhat.transpose(-1,-2).conj()
            Rb = Rb.diagonal(dim1=-1, dim2=-2).diag_embed()
            Rb.imag = Rb.imag - Rb.imag

            # vj = Rsshatnf.diagonal(dim1=-1, dim2=-2)
            # vj.imag = vj.imag - vj.imag
            outs = []
            inp = inp.detach()
            for j in range(J):
                inp[:,j,...,1::2] = g[:, j]
                outs.append(torch.sigmoid(model(inp[:,j])))
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
        g.requires_grad_(False)
        inp = inp.detach()
        model.train()
        for param in model.parameters():
            param.requires_grad_(True)

        outs = []
        for j in range(J):
            outs.append(torch.sigmoid(model(inp[:,j])))
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
    torch.save(model, f'model_rid{rid}.pt')
    torch.save(Hhat, f'Hhat_rid{rid}.pt')