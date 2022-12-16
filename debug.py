#%% 3-class EM
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())

"make the result reproducible"
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)       # current GPU seed
torch.cuda.manual_seed_all(seed)   # all GPUs seed
torch.backends.cudnn.deterministic = True  #True uses deterministic alg. for cuda
torch.backends.cudnn.benchmark = False  #False cuda use the fixed alg. for conv, may slower

#%%
d, s, h = torch.load('../data/nem_ss/val1kM3FT64_xsh_data5.pt')
N, F = s.shape[-1], s.shape[-2] # h is M*J matrix, here 6*6
ratio = d.abs().amax(dim=(1,2,3))
x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
s_all = s.abs().permute(0,2,3,1) 

class EM:
    def calc_ll(self, x, vhat, Rj):
        """ Rj shape of [J, M, M]
            vhat shape of [N, F, J]
            x shape of [N, F, M]
        """
        Rcj = vhat[...,None, None]*Rj[:,None,None] #shape of[J,N,F,M,M]
        Rx = Rcj.sum(0) #shape of[N,F,M,M]
        l = -(np.pi*Rx.det()).log() - (x[..., None, :].conj()@Rx.inverse()@x[..., None]).squeeze()
        return Rcj, Rx, l.sum()

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
        for i in range(z.shape[0]-K): # threshold of K level to stop
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
        vhat = torch.ones(J, N, F).abs().to(dtype)
        Rb = torch.eye(M).to(dtype)*Rbscale

        return vhat, Hhat, Rb, Rj

    def em_func_(self, x, J=3, max_iter=501, lamb=0, thresh_K=60, init=1):
        """init=0: random
            init=1: x_bar original
            init=else: x_tilde duong's paper
        """
        #  EM algorithm for one complex sample
        N, F, M = x.shape
        eye = torch.eye(M).to(x.dtype)
        if init == 0: # random init
            vhat, Hhat, Rb, Rj = self.rand_init(x, J=J)
        elif init == 1: #hierarchical initialization -- x_bar
            vhat, Hhat, Rb, Rj = self.cluster_init(x, J=J, K=thresh_K, init=init)
        else:  #hierarchical initialization -- x_tilde
            vhat, Hhat, Rb, Rj = self.cluster_init(x, J=J, K=thresh_K, init=init)
        
        Rcj, Rx, ll = self.calc_ll(x, vhat, Rj)
        ll_traj = []
        for i in range(max_iter):
            "E-step"
            W = Rcj @ Rx.inverse() #shape of[J,N,F,M,M]
            chat = W @ x[...,None] #shape of[J,N,F,M,1]
            Rcjhat = chat @ chat.transpose(-1,-2).conj() + (eye- W)@Rcj #shape of[J,N,F,M,M]

            "M-step"
            vhat = (Rj.inverse()[:,None,None]@Rcjhat).diagonal(dim1=-1, dim2=-2).mean(-1)
            vhat.real = threshold(vhat.real, floor=1e-10, ceiling=10)
            vhat.imag = vhat.imag - vhat.imag
            Rj = (Rcjhat/vhat[...,None, None]).mean(dim=(1,2)) #shape of[J,M,M]

            "compute log-likelyhood"
            Rcj, Rx, ll = self.calc_ll(x, vhat, Rj)
            ll_traj.append(ll.item())
            if i > 30 and abs((ll_traj[i] - ll_traj[i-3])/ll_traj[i-3]) <1e-4:
                print(f'EM early stop at iter {i}')
                break

        return chat, vhat, ll_traj, torch.linalg.matrix_rank(Rcj).double().mean()

#%%
EMs, EMh = [], []
for snr in ['inf']:#, 20, 10, 5, 0]:
    ems, emh = [], []
    for ind in range(1):
        if snr != 'inf':
            data = awgn(x_all[ind], snr)
        else:
            data = x_all[ind]
        chat, vhat, ll_traj, rank = \
                EM().em_func_(data, J=3, max_iter=301, thresh_K=10, init=1)
        shat = chat.permute(3,4,1,2,0).abs()[0]
        hhat = chat[...,0].permute(1,2,3,0).mean(dim=(0,1))
        plt.figure()
        plt.plot(ll_traj)
        plt.title(f'{ind}')
        plt.show()

        temp_s = s_corr_cuda(shat, s_all[ind:ind+1].abs()).item()
        temp = h_corr(hhat, h[ind])
        if ind %1 == 0 :
            print(f'At epoch {ind}', ' h corr: ', temp, ' s corr:', temp_s)

        ems.append(temp_s)
        emh.append(temp)

    EMs.append(sum(ems)/len(ems))
    EMh.append(sum(emh)/len(emh))

    print(f'done with one snr {snr}')
    print('EMs, EMh', EMs, EMh)

print('End date time ', datetime.now())