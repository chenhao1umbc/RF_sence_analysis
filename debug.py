#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)


#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)
import itertools
from unet.unet_model import UNetHalf8to100_256_sig as UNetHalf
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)

I, J = 70, 3
d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
s_all = s.abs().permute(0,2,3,1) 
ratio = d.abs().amax(dim=(1,2,3))/3
xte = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]*3).permute(0,2,3,1)# [sample, N, F, channel]
data = Data.TensorDataset(awgn_batch(xte[:I], snr=0))
data_test = Data.DataLoader(data, batch_size=64, drop_last=True)
g = torch.load('../data/nem_ss/gtest_500.pt')
gte = g[:I]/g[:I].amax(dim=[1,2])[...,None,None]  #standardization 
gte = torch.cat([gte[:,None] for j in range(J)], dim=1)[:,:,None] # shape of [I_val,J,1,8,8]
l = torch.load('../data/nem_ss/140100_lb.pt')
lb = l.repeat(64, 1, 1, 1, 1).cuda()

def corr(vh, v):
    J = v.shape[-1]
    r = [] 
    permutes = list(itertools.permutations(list(range(J))))
    for jj in permutes:
        temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
        s = 0
        for j in range(J):
            s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[j].flatten())[0])
        r.append(s)
    r = sorted(r, reverse=True)
    return r[0]/J

def h_corr(h, hh):
    J = h.shape[-1]
    r = [] 
    permutes = list(itertools.permutations(list(range(J))))
    for p in permutes:
        temp = hh[:,torch.tensor(p)]
        s = 0
        for j in range(J):
            dino = h[:,j].norm() * temp[:, j].norm()
            nume = (temp[:, j].conj() @ h[:, j]).abs()
            s = s + nume/dino
        r.append(s/J)
    r = sorted(r, reverse=True)
    return r[0].item()

def nem_minibatch_test(data, ginit, model, lb, seed=1):
    torch.manual_seed(seed) 
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    EM_iters = 501
    M, N, F, J = 3, 100, 100, 3
    NF, I, batch_size = N*F, ginit.shape[0], 64

    vtr = torch.randn(N, F, J).abs().to(torch.cdouble).repeat(I, 1, 1, 1)
    Hhat = torch.randn(M, J).to(torch.cdouble).cuda()
    Rbtr = torch.ones(I, M).diag_embed().to(torch.cdouble)*100

    lv, s, h, v, ll_all = ([] for i in range(5)) 
    for i, (x,) in enumerate(data): # gamma [n_batch, 4, 4]
        #%% EM part
        vhat = vtr[i*batch_size:(i+1)*batch_size].cuda()        
        Rb = Rbtr[i*batch_size:(i+1)*batch_size].cuda()
        g = ginit[i*batch_size:(i+1)*batch_size].cuda().requires_grad_()

        x = x.cuda()
        optim_gamma = torch.optim.SGD([g], lr=0.001)
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((1,2))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        ll_traj = []

        for ii in range(EM_iters):
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
            Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
            Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb 
            Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
            l = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze()
            ll = l.sum().real
            Rx = Rxperm

            ll_traj.append(ll.item())
            if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
            if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
                print(f'EM early stop at iter {ii}')
                break
        ll_all.append(l.sum((-1, -2)).cpu().real)
        lv.append(ll.item())
        s.append(shat)
        h.append(Hhat)
        v.append(vhat)
        print(f'batch {i} is done')
    return sum(lv)/len(lv), torch.cat(ll_all), (s, h, v)

rid = 150000
model = torch.load(f'../data/nem_ss/models/rid{rid}/model_rid{rid}_35.pt')
meanl, l_all, shv = nem_minibatch_test(data_test, gte, model, lb, seed=1)
print('End date time ', datetime.now())

shat, hhat, vhat = shv
shat_all, hhat_all = torch.cat(shat).cpu(), torch.cat(hhat).cpu()
res_s = []
for i in range(64):
    res_s.append(corr(shat_all[i].squeeze().abs(), s_all[i]))
print(sum(res_s)/len(res_s))



#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)
from skimage.transform import resize
import itertools
import time
t = time.time()
d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
h = torch.tensor(h)
ratio = d.abs().amax(dim=(1,2,3))/3
x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
s_all = s.abs().permute(0,2,3,1) 

def corr(vh, v):
    J = v.shape[-1]
    r = [] 
    permutes = list(itertools.permutations(list(range(J))))
    for jj in permutes:
        temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
        s = 0
        for j in range(J):
            s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[j].flatten())[0])
        r.append(s)
    r = sorted(r, reverse=True)
    return r[0]/J

def h_corr(h, hh):
    J = h.shape[-1]
    r = [] 
    permutes = list(itertools.permutations(list(range(J))))
    for p in permutes:
        temp = hh[:,torch.tensor(p)]
        s = 0
        for j in range(J):
            dino = h[:,j].norm() * temp[:, j].norm()
            nume = (temp[:, j].conj() @ h[:, j]).abs()
            s = s + nume/dino
        r.append(s/J)
    r = sorted(r, reverse=True)
    return r[0].item()

def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=501, seed=1, model=''):
    torch.manual_seed(seed) 
    if model == '':
        print('A model is needed')

    model = torch.load(model)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    #%% EM part
    "initial"        
    N, F, M = x.shape
    NF= N*F
    graw = torch.tensor(resize(x[...,0].abs(), [8,8], order=1, preserve_range=True))
    graw = (graw/graw.max())[None,...]  #standardization shape of [1, 8, 8]
    g = torch.stack([graw[:,None] for j in range(J)], dim=1).cuda()  # shape of [1,J,8,8]
    lb = torch.load('../data/nem_ss/140100_lb.pt')
    lb = lb[None,...].cuda()
    x = x.cuda()

    vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
    outs = []
    for j in range(J):
        outs.append(torch.sigmoid(model(torch.cat((g[:,j], lb[:,j]), dim=1))))
    out = torch.cat(outs, dim=1).permute(0,2,3,1)
    vhat.real = threshold(out)
    Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
    Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
    Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
    Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
    Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
    g.requires_grad_()
    optim_gamma = torch.optim.SGD([g], lr= 0.001)
    ll_traj = []

    for ii in range(max_iter): # EM iterations
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
            outs.append(torch.sigmoid(model(torch.cat((g[:,j], lb[:,j]), dim=1))))
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
        if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
            print(f'EM early stop at iter {ii}')
            break

    return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu(), ll_traj

rid = 150000
location = f'../data/nem_ss/models/rid{rid}/model_rid{rid}_35.pt'
single_data = True
if single_data:
    ind = 9
    shat, Hhat, vhat, Rb, loss = nem_func(x_all[ind],seed=1,model=location)
    for i in range(3):
        plt.figure()
        plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
        plt.colorbar()
        plt.title(f'Estimated sources {i+1}')
        plt.show()
    print('h correlation:', h_corr(h, Hhat.squeeze()))
    print('s correlation:', corr(shat.squeeze().abs(), s_all[ind]))
    
    for i in range(3):
        plt.figure()
        plt.imshow(s_all[ind].squeeze().abs()[...,i])
        plt.colorbar()
        plt.title(f'GT sources {i+1}')
        plt.show()

    plt.figure()
    plt.plot(loss, '-x')
    plt.title('loss value')
    plt.show()

#%%
