#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
# torch.set_default_dtype(torch.double)

#%%
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
h, N, F, M = torch.tensor(h), s.shape[-1], s.shape[-2], 3
ratio = d.abs().amax(dim=(1,2,3))/3
x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
s_all = s.abs().permute(0,2,3,1) 

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
    lb = lb[None,...].cuda()
    lb = torch.load('lb_140100.pt')
    lb = lb[:1,...].cuda()
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
        vhat.real = threshold(out, ceiling=1)
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
        if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
            print(f'EM early stop at iter {ii}')
            break

    return (shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze()), g.detach().cpu(), Rb.cpu(), ll_traj

rid = 171000
model = f'../data/nem_ss/models/rid{rid}/model_rid{rid}_45.pt'
# rid = 160100
# model = f'../data/nem_ss/models/rid{rid}/model_rid{rid}_33.pt'
which_class, ind = [0,5], 15
for i, v in enumerate(which_class):
    if i == 0 : d = 0
    d = d + h[:M, v, None] @ s[ind, v].reshape(1, N*F)
r = d.abs().max()
d = d.reshape(M, N, F).permute(1,2,0)/r

#%%
shv, g, Rb, loss = nem_func_less(awgn(d, 30), J=3, seed=10, model=model, max_iter=301)
shat, Hhat, vhat = shv

c = torch.rand(100,100,3,3).to(torch.cfloat)
for i in range(3):
    c[:,:,i] = shat.squeeze()[...,i][..., None] @ Hhat.squeeze()[None, :, i]
    print(c[:,:,i].norm())

for i in range(3):
    plt.figure()
    plt.imshow(c[:,:,i].squeeze().abs()[...,i]*r)
    plt.title('plot of c from NEM')
    plt.colorbar()

for i in range(3):
    plt.figure()
    plt.imshow(shat.squeeze().abs()[...,i])
    plt.colorbar()
    plt.title('plot of s from NEM')

# for i in range(3):
#     plt.figure()
#     plt.imshow(vhat.abs()[...,i])
#     plt.colorbar()

for i, v in enumerate(which_class):
    plt.figure()
    plt.imshow(s[ind, v].squeeze().abs())
    plt.colorbar()
print('done')

def hcorr(h, hh):
    n = h[None, :] @ hh[..., None].conj()
    d = h.norm() * hh.norm()
    return n.abs()/d

for v in which_class:
    print(f'class {v}')
    for i in range(3):
        print(hcorr(h[:M, v], Hhat.squeeze()[:,i]))

#%% EM
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)
from skimage.transform import resize
import itertools
import time

def em_func_(x, J=6, Hscale=1, Rbscale=100, max_iter=501, lamb=0, seed=0, show_plot=False):
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

    def mydet(x):
        """calc determinant of tensor for the last 2 dimensions,
        suppose x is postive definite hermitian matrix

        Args:
            x ([pytorch tensor]): [shape of [..., N, N]]
        """
        s = x.shape[:-2]
        N = x.shape[-1]
        l = torch.linalg.cholesky(x)
        ll = l.diagonal(dim1=-1, dim2=-2)
        res = torch.ones(s).to(x.device)
        for i in range(N):
            res = res * ll[..., i]**2
        return res

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
        Rcj = Hhat @ Rs @ Hhat.t().conj()
        Rx = Rcj + Rb
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
        vhat.real = threshold(vhat.real, floor=1e-10, ceiling=10)
        vhat.imag = vhat.imag - vhat.imag
        Hhat = Rxshat @ Rsshat.inverse()
        Rb = Rxxhat - Hhat@Rxshat.t().conj() - \
            Rxshat@Hhat.t().conj() + Hhat@Rsshat@Hhat.t().conj()
        Rb = threshold(Rb.diag().real, floor=1e-20).diag().to(torch.cdouble)
        # Rb = Rb.diag().real.diag().to(torch.cdouble)

        "compute log-likelyhood"
        for j in range(J):
            Rj[j] = Hhat[:, j][..., None] @ Hhat[:, j][..., None].t().conj()
        ll_traj.append(calc_ll_cpx2(x, vhat, Rj, Rb).item())
        if i > 30 and abs((ll_traj[i] - ll_traj[i-3])/ll_traj[i-3]) <1e-4:
            print(f'EM early stop at iter {i}')
            break

    return shat, Hhat, vhat, Rb, ll_traj, torch.linalg.matrix_rank(Rcj).double().mean()

shat, Hhat, vhat, Rb, ll_traj, rank = em_func_(awgn(d, 30), J=3, max_iter=300)

for i in range(3):
    plt.figure()
    plt.imshow(shat.squeeze().abs()[...,i])
    plt.title('plot of s from EM')
    plt.colorbar()

c = torch.rand(100,100,3,3).to(torch.cfloat)
for i in range(3):
    c[:,:,i] = shat.squeeze()[...,i][..., None] @ Hhat.squeeze()[None, :, i]
    print(c[:,:,i].norm())

for i in range(3):
    plt.figure()
    plt.imshow(c[:,:,i].squeeze().abs()[...,i]*r)
    plt.title('plot of c from EM')
    plt.colorbar()