#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())

#%%
d, s, h = torch.load('../data/nem_ss/val500M3FT64_xsh_data0.pt')
h, N, F = torch.tensor(h), s.shape[-1], s.shape[-2] # h is M*J matrix, here 6*6
ratio = d.abs().amax(dim=(1,2,3))
x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
s_all = s.abs().permute(0,2,3,1) 

def cluster_init(x, J=3, K=60, init=1, Rbscale=1e-3, showfig=False):
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
    vhat = torch.ones(N, F, J).abs().to(dtype)
    Rb = torch.eye(M).to(dtype)*Rbscale

    return vhat, Hhat, Rb, Rj

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

    "initial"        
    N, F, M = x.shape
    NF= N*F
    g = torch.rand(1,J,1,8,8).cuda()  # shape of [1,J,8,8]
    vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
    outs = []
    for j in range(J):
        outs.append(model(g[:,j]))
    out = torch.cat(outs, dim=1).permute(0,2,3,1)
    vhat.real = threshold(out)
    # Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
    # _, _ , h = d = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    _, Hhat, _, _ = cluster_init(x)
    x = x.cuda()
    Hhat = Hhat.to(torch.cdouble).cuda()
    Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)

    Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
    Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
    Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
    g.requires_grad_()
    optim_gamma = torch.optim.SGD([g], lr= 0.01)
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
            outs.append(model(g[:,j]))
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

#%%
EMs, EMh = [], []
s_raw, h_raw = [], []
for snr in ['inf']:
    ss, emh = [], []
    hh, ems = [], []
    for ind in range(100):
        if snr != 'inf':
            data = awgn(x_all[ind], snr)
        else:
            data = x_all[ind]
        shat, Hhat, vhat, Rb, ll_traj, rank = EM().em_func_(data, J=3, max_iter=301, init=0)
        temp_s = s_corr(shat.squeeze().abs(), s_all[ind].abs())
        temp = h_corr(Hhat.squeeze(), h[ind])
        print('h corr: ', temp)
        print('s corr:', temp_s)

        noise = (x_all[ind] - (Hhat@shat).squeeze()).permute(2,0,1)
        # rs0 = get_metric(s[ind], shat.squeeze().permute(2,0,1), noise)
        # ems.append(rs0[0].item())
        ems.append(temp_s)
        emh.append(temp)

        ss.append(shat)
        hh.append(Hhat)

    EMs.append(sum(ems)/100)
    EMh.append(sum(emh)/100)
    s_raw.append(ss)
    h_raw.append(hh)
    print('done with one snr')
    print('EMs, EMh', EMs, EMh)

print('End date time ', datetime.now())
torch.save((s_raw, h_raw), 'EMs_EMh_raw_init1.pt')