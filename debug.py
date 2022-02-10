#%% EM to see hierarchical initialization result
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
# torch.set_default_dtype(torch.double)
from skimage.transform import resize
import itertools
import time

#%% test 
d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh_ang586264.pt')
h, N, F = torch.tensor(h), s.shape[-1], s.shape[-2] # h is M*J matrix, here 6*6
ratio = d.abs().amax(dim=(1,2,3))
x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
s_all = s.abs().permute(0,2,3,1) 

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
        dtype = x.dtype

        vhat = torch.randn(N, F, J).abs().to(dtype)
        Hhat = torch.randn(M, J, dtype=dtype)*Hscale
        Rb = torch.eye(M).to(dtype)*Rbscale
        Rj = torch.zeros(J, M, M).to(dtype)
        return vhat, Hhat, Rb, Rj
    
    def cluster_init(self, x, J=3, K=60, init=1, Rbscale=1e-3, showfig=False):
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
        Here we use scipy function for the linkage.
        J is how many clusters
        """   
        dtype = x.dtype
        N, F, M = x.shape

        "get data and clusters ready"
        x_norm = ((x[:,:,None,:]@x[..., None].conj())**0.5)[:,:,0]
        if init==1: x_ = x/x_norm * (-1j*x[...,0:1].angle()).exp() # shape of [N, F, M] x_bar
        else: x_ = x * (-1j*x[...,0:1].angle()).exp() # the x_tilde in Duong's paper
        data = x_.reshape(N*F, M)
        I = data.shape[0]
        C = [[i] for i in range(I)]  # initial cluster

        "calc. affinity matrix and linkage"
        perms = torch.combinations(torch.arange(len(C)))
        d = data[perms]
        table = ((d[:,0] - d[:,1]).abs()**2).sum(dim=-1)**0.5
        from scipy.cluster.hierarchy import dendrogram, linkage
        z = linkage(table, method='average')
        if showfig: dn = dendrogram(z, p=3, truncate_mode='level')

        "find the max J cluster and sample index"
        zind = torch.tensor(z).to(torch.int)
        flag = torch.cat((torch.ones(I), torch.zeros(I)))
        c = C + [[] for i in range(I)]
        for i in range(z.shape[0]-K):  # threshold of K level to stop
            c[i+I] = c[zind[i][0]] + c[zind[i][1]]
            flag[i+I], flag[zind[i][0]], flag[zind[i][1]] = 1, 0, 0
        ind = (flag == 1).nonzero(as_tuple=True)[0]
        dict_c = {}  # which_cluster: how_many_nodes
        for i in range(ind.shape[0]):
            dict_c[ind[i].item()] = len(c[ind[i]])
        dict_c_sorted = {k:v for k,v in sorted(dict_c.items(), key=lambda x: -x[1])}
        cs = []
        for i, (k,v) in enumerate(dict_c_sorted.items()):
            if i == J:
                break
            cs.append(c[k])

        "initil the EM variables"
        Hhat = torch.rand(M, J, dtype=dtype)
        Rj = torch.rand(J, M, M, dtype=dtype)
        for i in range(J):
            d = data[torch.tensor(cs[i])] # shape of [I_cj, M]
            Hhat[:,i] = d.mean(0)
            Rj[i] = (d[..., None] @ d[:,None,:].conj()).mean(0)
        vhat = torch.ones(N, F, J).abs().to(dtype)
        Rb = torch.eye(M).to(dtype)*Rbscale

        return vhat, Hhat, Rb, Rj

    def em_func_(self, x, J=3, max_iter=501, lamb=0, init=1):
        """init=0: random
            init=1: x_bar original
            init=else: x_tilde duong's paper
        """
        #  EM algorithm for one complex sample
        N, F, M = x.shape
        NF= N*F
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        if init == 0: # random init
            vhat, Hhat, Rb, Rj = self.rand_init(x, J=J)
        elif init == 1: #hierarchical initialization -- x_bar
            vhat, Hhat, Rb, Rj = self.cluster_init(x, J=J, init=init)
        else:  #hierarchical initialization -- x_tilde
            vhat, Hhat, Rb, Rj = self.cluster_init(x, J=J, init=init)

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
            elif lamb>0:
                y = Rsshatnf.diagonal(dim1=-1, dim2=-2)
                vhat = -0.5/lamb + 0.5*(1+4*lamb*y)**0.5/lamb
            else:
                vhat = Rsshatnf.diagonal(dim1=-1, dim2=-2)
            vhat.real = threshold(vhat.real, floor=1e-10, ceiling=10)
            vhat.imag = vhat.imag - vhat.imag
            Hhat = Rxshat @ Rsshat.inverse()
            Rb = Rxxhat - Hhat@Rxshat.t().conj() - \
                Rxshat@Hhat.t().conj() + Hhat@Rsshat@Hhat.t().conj()
            # Rb = threshold(Rb.diag().real, floor=1e-20).diag().to(x.dtype)
            Rb = Rb.diag().real.diag().to(x.dtype)

            "compute log-likelyhood"
            for j in range(J):
                Rj[j] = Hhat[:, j][..., None] @ Hhat[:, j][..., None].t().conj()
            ll_traj.append(self.calc_ll_cpx2(x, vhat, Rj, Rb).item())
            if i > 30 and abs((ll_traj[i] - ll_traj[i-3])/ll_traj[i-3]) <1e-4:
                print(f'EM early stop at iter {i}')
                break

        return shat, Hhat, vhat, Rb, ll_traj, torch.linalg.matrix_rank(Rcj).double().mean()

def h_corr(h, hh):
    "hh and h are in the shape of [M, J]"
    J = h.shape[-1]
    r = [] 
    permutes = list(itertools.permutations(list(range(J))))
    for p in permutes:
        temp = hh[:,torch.tensor(p)]
        s = 0
        for j in range(J):
            dino = h[:,j].norm() * temp[:, j].norm()
            nume = (temp[:, j].conj() * h[:, j]).sum().abs()
            s = s + nume/dino
        r.append(s/J)
    r = sorted(r, reverse=True)
    return r[0].item()

#%%
ind = 10
vhat, Hhat, Rb, Rj = EM().cluster_init(x_all[ind])
print(h_corr(Hhat, h[ind]))
# print(Hhat.angle()/np.pi*180)
# print(h[ind].angle()/np.pi*180)

shat, Hhat, vhat, Rb, ll_traj, rank = EM().em_func_(x_all[0], max_iter=300, init=1)
print(h_corr(Hhat, h[ind]))

plt.figure()
plt.plot(ll_traj, '-x')

for i in range(3):
    plt.figure()
    plt.imshow(shat.squeeze().abs()[...,i])
    plt.title('plot of s from EM')
    plt.colorbar()

#%%
NF, J, N, F, M = 100, 3, 10, 10, 6
s_hat = torch.rand(J,10,10).to(torch.complex64)
s = torch.rand(J,10,10).to(torch.complex64)
noise = torch.rand(M, J,10,10).to(torch.complex64)
PsTimessj_hat = s.clone()

"get s_target"
s_target = ((s*s_hat.conj()).sum(dim=(-1,-2), keepdim=True) *s)/(s.abs()**2).sum(dim=(-1,-2), keepdim=True)
"get e_interf"
Rss = s.reshape(J, NF) @ s.reshape(3, NF).conj().t()# Rss is the Gram matrix of the sources [J,J]
for j in range(J):
    temp = (s_hat[j] * s.conj()).sum(dim=(-1,-2)) #shape of [J]
    c = Rss.inv()@ temp[None,:].conj().t()
    PsTimessj_hat[j] = (c.t().conj()@s.reshape(J, NF)).reshape(N,F)
e_interf = PsTimessj_hat - s_target
"get e_noise"
e_noise = 0
"get e_artif"
e_artif = s_hat - e_noise -PsTimessj_hat

SDR = (s_target.norm()/(e_interf+e_noise+e_artif).norm()).log10()*20
SIR = (s_target.norm()/e_interf.norm()).log10()*20
SNR = ((s_target+e_interf).norm()/e_noise.norm()).log10()*20
SAR = ((s_target+e_interf+e_noise).norm()/e_artif.norm()).log10()*20