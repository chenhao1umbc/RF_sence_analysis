#%% EM to see hierarchical initialization result
from dataclasses import astuple
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
    
    def cluster_init(self, x):
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

#%% test 
d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
h, N, F = torch.tensor(h), s.shape[-1], s.shape[-2] # h is M*J matrix, here 6*6
ratio = d.abs().amax(dim=(1,2,3))/3
x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
s_all = s.abs().permute(0,2,3,1) 

# which_class, ind, M = [0,2,5], 15, 3
# for i, v in enumerate(which_class):
#     if i == 0 : d = 0
#     d = d + h[:M, v, None] @ s[ind, v].reshape(1, N*F)
# r = d.abs().max()
# x = d.reshape(M, N, F).permute(1,2,0)/r  # shape of [N,F,M]
# x = x.to(torch.cfloat)

ind, J = 10, 3
x = x_all[ind].to(torch.cfloat)
dtype = x.dtype
N, F, M = x.shape

x_norm = ((x[:,:,None,:]@x[..., None].conj())**0.5)[:,:,0]
x_bar = x/x_norm * (-1j*x[...,0:1].angle()).exp()  # shape of [N, F, M]
x_tilde = x * (-1j*x[...,0:1].angle()).exp()
data = x_bar.reshape(N*F, M)

def get_clusters(data, J=3, showfig=False):
    """psudo code, https://www.saedsayad.com/clustering_hierarchical.htm
    Given : A set X of obejects{x1,...,xn}
            A cluster distance function dist(c1, c2)
    for i=1 to n
        ci = {xi}
    end for
    C = {c1, ..., cn}
    I = n+1
    While I>1 do
        (cmin1, cmin2) = minimum dist(ci, cj) for all ci, cj in C
        remove cmin1 and cmin2 from C
        add {cmin1, cmin2} to C
        I = I - 1
    end while

    However, this naive algorithm does not fit large samples. 

    J is how many clusters

    """        
    I = data.shape[0]
    C = [[i] for i in range(I)]  # initial cluster
    "first cluster"
    perms = torch.combinations(torch.arange(len(C)))
    d = data[perms]
    table = ((d[:,0] - d[:,1]).abs()**2).sum(dim=-1)**0.5
    from scipy.cluster.hierarchy import dendrogram, linkage
    z = linkage(table, method='average')
    if showfig: dn = dendrogram(z, p=3, truncate_mode='level')

    zind = torch.tensor(z).to(torch.int)
    flag = torch.cat((torch.ones(I), torch.zeros(I)))
    c = C + [[] for i in range(I)]
    for i in range(z.shape[0]-60):
        c[i+I] = c[zind[i][0]] + c[zind[i][1]]
        flag[i+I], flag[zind[i][0]], flag[zind[i][1]] = 1, 0, 0
    ind = (flag == 1).nonzero(as_tuple=True)[0]
    dict_c = {}  # which_cluster: how_many_nodes
    for i in range(ind.shape[0]):
        dict_c[ind[i].item()] = len(c[ind[i]])
    dict_c_sorted = {k:v for k,v in sorted(dict_c.items(), key=lambda x: -x[1])}
    res = []
    for i, (k,v) in enumerate(dict_c_sorted.items()):
        if i == J:
            break
        res.append(c[k])
    return res

cs = get_clusters(data)
hhat = torch.rand(M, J, dtype=dtype)
Rj = torch.rand(J, M, M)
for i in range(J):
    d = data[torch.tensor(cs[i])] # shape of [I_cj, M]
    hhat[:,i] = d.mean(0)
    Rj[i] = (d[..., None] @ d[:,None,:].conj()).mean(0).real
#%%
