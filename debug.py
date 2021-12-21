#%%
from PIL.Image import FASTOCTREE
from numpy import broadcast_arrays, dtype
from numpy.core.fromnumeric import nonzero
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)

#%% load data
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)
from skimage.transform import resize
import itertools
import time
t = time.time()
d, s, h = torch.load('../data/nem_ss/test500M6FT100_xsh.pt')
h, N, F, M = torch.tensor(h), s.shape[-1], s.shape[-2], 6
ratio = d.abs().amax(dim=(1,2,3))/3
x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
s_all = s.abs().permute(0,2,3,1) 

#%% NEM
def nem_func_less(x, J=6, Hscale=1, Rbscale=100, max_iter=501, seed=1, model=''):
    def log_likelihood(x, vhat, Hhat, Rb, ):
        """ Hhat shape of [I, M, J] # I is NO. of samples, M is NO. of antennas, J is NO. of sources
            vhat shape of [I, N, F, J]
            Rb shape of [I, M, M]
            x shape of [I, N, F, M]
        """
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rcj = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj()
        Rxperm = Rcj + Rb 
        Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
        l = -(np.pi*torch.linalg.det(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze() 
        return l.sum().real, Rs, Rxperm, Rcj

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
    lb = torch.load('../data/nem_ss/lb_c6_J188.pt')
    lb = lb[[0,5,2]]
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
        ll, Rs, Rx, Rcj = log_likelihood(x, vhat, Hhat, Rb)
        ll_traj.append(ll.item())
        if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
        if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <1e-4:
            print(f'EM early stop at iter {ii}')
            break

    return (shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze()), g.detach().cpu(), Rb.cpu(), ll_traj

rid = 160100
model = f'../data/nem_ss/models/rid{rid}/model_rid{rid}_33.pt'
t = time.time()
res = []
which_class, ind = [0,1,2], 0
for i, v in enumerate(which_class):
    if i == 0 : d = 0
    d = d + h[:M, v, None] @ s[ind, v].reshape(1, N*F)
d = d.reshape(M, N, F).permute(1,2,0)/d.abs().max()
shv, g, Rb, loss = nem_func_less(awgn(d, snr=30), J=3, seed=10, model=model, max_iter=301)
shat, Hhat, vhat = shv
# print('done', time.time()-t)

#%% EM
def em_func0(x, J=3, Hscale=1, Rbscale=100, max_iter=501, lamb=0, seed=0, show_plot=False):
    #  EM algorithm for one complex sample
    def calc_ll_cpx2(x, vhat, Rj, Rb):
        """ Rj shape of [J, M, M]
            vhat shape of [N, F, J]
            Rb shape of [M, M]
            x shape of [N, F, M]
        """
        _, M, M = Rj.shape
        N, F, J = vhat.shape
        Rcj = vhat.reshape(N*F, J) @ Rj.reshape(J, M*M)
        Rcj = Rcj.reshape(N, F, M, M)
        Rx = Rcj + Rb 
        l = -(np.pi*Rx.det()).log() - (x[..., None, :].conj()@Rx.inverse()@x[..., None]).squeeze()
        # l = -(np.pi*mydet(Rx)).log() - (x[..., None, :].conj()@Rx.inverse()@x[..., None]).squeeze()
        return l.sum()

    N, F, M = x.shape
    NF= N*F
    torch.torch.manual_seed(seed)
    vhat = torch.randn(N, F, J).abs().to(torch.cdouble)
    Hhat = torch.randn(M, J, dtype=torch.cdouble)*Hscale
    Rb = torch.eye(M).to(torch.cdouble)*Rbscale
    Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
    Rj = torch.zeros(J, M, M).to(torch.cdouble)
    ll_traj = []

    for i in range(max_iter):
        "E-step"
        Rs = vhat.diag_embed()
        Rx = Hhat @ Rs @ Hhat.t().conj() + Rb
        W = Rs @ Hhat.t().conj() @ Rx.inverse()
        shat = W @ x[...,None]
        Rsshatnf = shat @ shat.transpose(-1,-2).conj() + Rs - W@Hhat@Rs

        Rsshat = Rsshatnf.sum([0,1])/NF
        Rxshat = (x[..., None] @ shat.transpose(-1,-2).conj()).sum((0,1))/NF

        "M-step"
        if lamb<0:
            raise ValueError('lambda should be not negative')
        elif lamb >0:
            y = Rsshatnf.diagonal(dim1=-1, dim2=-2)
            vhat = -0.5/lamb + 0.5*(1+4*lamb*y)**0.5/lamb
        else:
            vhat = Rsshatnf.diagonal(dim1=-1, dim2=-2)
        vhat.imag = vhat.imag - vhat.imag
        vhat.real = threshold(vhat.real, ceiling=100)
        Hhat = Rxshat @ Rsshat.inverse()
        Rb = Rxxhat - Hhat@Rxshat.t().conj() - \
            Rxshat@Hhat.t().conj() + Hhat@Rsshat@Hhat.t().conj()
        Rb = Rb.diag().diag()
        Rb.imag = Rb.imag - Rb.imag

        "compute log-likelyhood"
        for j in range(J):
            Rj[j] = Hhat[:, j][..., None] @ Hhat[:, j][..., None].t().conj()
        ll_traj.append(calc_ll_cpx2(x, vhat, Rj, Rb).item())
        if i > 30 and abs((ll_traj[i] - ll_traj[i-3])/ll_traj[i-3]) <1e-4:
            print(f'EM early stop at iter {i}')
            break

    if show_plot:
        plt.figure(100)
        plt.plot(ll_traj,'o-')
        plt.show()
        "display results"
        for j in range(J):
            plt.figure()
            plt.imshow(vhat[:,:,j].real)
            plt.colorbar()

    return shat, Hhat, vhat, Rb, ll_traj

which_class, ind = [0,2,5], 10
for i, v in enumerate(which_class):
    if i == 0 : d = 0
    d = d + h[:M, v, None] @ s[ind, v].reshape(1, N*F)
d = d.reshape(M, N, F).permute(1,2,0)/d.abs().max()
shat, Hhat, vhat, Rb, loss = em_func0(awgn(d,snr=30), J=6)


#%%
plt.figure()
plt.plot(loss, '-x')
plt.show()

plt.figure()
for i in range(shat.squeeze().shape[-1]):
    plt.subplot(3,2,i+1)
    plt.imshow(shat.squeeze()[:,:,i].abs())
    a = plt.colorbar()
    a.ax.tick_params(labelsize=8)
    plt.tight_layout(pad=1.1)
plt.suptitle("Plots of estimated s", x=0.6)
plt.tight_layout(pad=1.5)

plt.figure()
for i in range(vhat.shape[-1]):
    plt.subplot(3,2,i+1)
    plt.imshow(vhat.squeeze()[:,:,i].abs())
    a = plt.colorbar()
    a.ax.tick_params(labelsize=8)
    
    plt.tight_layout(pad=1.1)
plt.suptitle("Plots of estimated v", x=0.6)
plt.tight_layout(pad=1.1)

plt.figure()
for i, v in enumerate(which_class):
    plt.subplot(3,2,i+1)
    plt.imshow(s[ind, v].abs())
    plt.title(f'Source {v+1} in sample {ind}', fontsize=10)
    a = plt.colorbar()
    a.ax.tick_params(labelsize=8)
    plt.tight_layout(pad=1.1)
plt.suptitle("Ground truth")
plt.tight_layout(pad=1.1)

# graw = torch.tensor(resize(d[...,0].abs(), [8,8], order=1, preserve_range=True))
# graw = (graw/graw.max())  #standardization shape of [1, 8, 8]
# # plt.imshow(graw.abs())
# g, cmin, cmax = g.squeeze(), (g-graw).min(), (g-graw).max()
# plt.figure()
# for i in range(6):
#     plt.subplot(3,2,i+1)
#     plt.imshow(g[i]-graw)
#     # plt.clim(cmin,cmax)
#     plt.colorbar()
#     plt.tight_layout(pad=1.1)