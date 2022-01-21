#%% EM to see hierarchical initialization result
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
# torch.set_default_dtype(torch.double)
from skimage.transform import resize
import itertools
import time

#%%
class EM:
    def calc_ll_cpx2(self, x, vhat, Rj, Rb):
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
        return l.sum()
    
    def rand_init(self, x, J=6, Hscale=1, Rbscale=100, seed=0):
        N, F, M = x.shape
        torch.torch.manual_seed(seed)
        if x.dtype == torch.float:
            dtype = torch.cfloat
        if x.dtype == torch.double:
            dtype = torch.cdouble
        vhat = torch.randn(N, F, J).abs().to(dtype)
        Hhat = torch.randn(M, J, dtype=dtype)*Hscale
        Rb = torch.eye(M).to(dtype)*Rbscale
        Rj = torch.zeros(J, M, M).to(dtype)
        return vhat, Hhat, Rb, Rj
    
    def rand_init(self, x):
        N, F, M = x.shape
        if x.dtype == torch.float:
            dtype = torch.cfloat
        if x.dtype == torch.double:
            dtype = torch.cdouble

        x_bar = x/(x[..., None] @ x[:,:,None,:].conj())

        vhat, Hhat, Rb, Rj = [0]*4
        return vhat, Hhat, Rb, Rj

    def em_func_(self, x, J=6, max_iter=501, lamb=0, init=1):
        #  EM algorithm for one complex sample
        N, F, M = x.shape
        NF= N*F
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        if init == 0: # random init
            vhat, Hhat, Rb, Rj = self.rand_init(J)
        else:  #hierarchical initialization
            pass


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

shat, Hhat, vhat, Rb, ll_traj, rank = EM().em_func_(awgn(d, 30), J=3, max_iter=300)

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


#%%
d, s, h = torch.load('/home/chenhao1/Hpython/data/nem_ss/test500M6FT100_xsh.pt')
h, N, F = torch.tensor(h), s.shape[-1], s.shape[-2] # h is M*J matrix, here 6*6
ratio = d.abs().amax(dim=(1,2,3))/3
x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
s_all = s.abs().permute(0,2,3,1) 

which_class, ind, M = [0,2,5], 15, 6
for i, v in enumerate(which_class):
    if i == 0 : d = 0
    d = d + h[:M, v, None] @ s[ind, v].reshape(1, N*F)
r = d.abs().max()
x = d.reshape(M, N, F).permute(1,2,0)/r  # shape of [N,F,M]

N, F, M = x.shape
if x.dtype == torch.float:
    dtype = torch.cfloat
if x.dtype == torch.double:
    dtype = torch.cdouble

x_norm = ((x[:,:,None,:]@x[..., None].conj())**0.5)[:,:,0]
x_bar = x/x_norm  # shape of [N, F, M]
