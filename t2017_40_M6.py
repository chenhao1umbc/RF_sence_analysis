#%% t2017_40_M6
"""Key varibales are 
M - how many channels
which_class - list of source index, which sources are in the mixture
J - the algorithm presumes how many classes in the mixture
ind - from 0 to 99, index of test sample
max_iter - how many EM iterations
seed - random seed number
rid - training model index
model - saved trained model
"""
from utils import *

import matplotlib
matplotlib.rc('font', size=16)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
from skimage.transform import resize
import itertools
import time
t = time.time()
d, s, h = torch.load('../data/nem_ss/test500M6FT100_xsh.pt')
h, N, F = torch.tensor(h), s.shape[-1], s.shape[-2] # h is M*J matrix, here 6*6
ratio = d.abs().amax(dim=(1,2,3))
x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
s_all = s.abs().permute(0,2,3,1) 
glr = 0.005

#%% loading data and functions
from unet.unet_model import *
#%% define model
from unet.unet_model import *
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        n_channels = 1
        n_classes = 1
        bilinear = False
        n_ch = 384
        self.inc = DoubleConv(n_channels, n_ch)
        self.up1 = MyUp(n_ch, n_ch//2)
        self.up2 = MyUp(n_ch//2, n_ch//4)
        self.up3 = MyUp(n_ch//4, n_ch//8)
        self.up4 = MyUp(n_ch//8, n_ch//16)
        self.reshape = nn.Sequential(
            nn.Conv2d(n_ch//16, n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_ch//16, n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_ch//16, n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        x = self.sig(x)
        out = x/x.detach().amax(keepdim=True, dim=(-1,-2))
        return out

def cluster_init(x, J=3, K=14, init=1, Rbscale=1e-3, showfig=False):
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

def nem_hci(x, J=6, Hscale=1, Rbscale=1, max_iter=501, seed=1, model=''):

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
    graw = torch.tensor(resize(x[...,0].abs().numpy(), [8,8], order=1, preserve_range=True))
    graw = (graw/graw.max())[None,...]  #standardization shape of [1, 8, 8]
    g = torch.stack([graw[:,None] for j in range(J)], dim=1)  # shape of [1,J,8,8] 
    noise = torch.rand(1, J, 1, 8, 8)
    g = (g + noise/10).to(torch.float).cuda() 
    x = x.cuda()

    vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
    outs = []
    for j in range(J):
        outs.append(model(g[:,j]))
    out = torch.cat(outs, dim=1).permute(0,2,3,1)
    vhat.real = threshold(out)
    # Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
    Hhat = cluster_init(x.cpu(), J=J)[1].cuda()
    Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
    Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
    Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
    Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
    g.requires_grad_()
    optim_gamma = optim.RAdam([g],
            lr= glr,
            betas=(0.9, 0.999), 
            eps=1e-8,
            weight_decay=0)
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

#%%
for dm in range(1,41):
    res_s, res_h = [], []
    for ii in range(20):
        which_class, ind = [0,1,2,3,4,5], ii
        M, J = 6, len(which_class)
        for i, v in enumerate(which_class):
            if i == 0 : d = 0
            d = d + h[:M, v, None] @ s[ind, v].reshape(1, N*F)
        r = d.abs().max()
        d = d.reshape(M, N, F).permute(1,2,0)/r
        data = awgn(d, 30, seed=0)

        # rid = 160100
        # model = f'../data/nem_ss/models/rid1+/rid{rid}/model_rid{rid}_33.pt'
        # shv, g, Rb, loss = nem_func_less(data, J=J, seed=10, model=model, max_iter=301)

        rid = 2017
        model = f'../data/nem_ss/models/rid{rid}/model_rid{rid}_{dm}.pt'
        shv, g, Rb, loss = nem_hci(data, J=J, seed=10, model=model, max_iter=501)

        shat, Hhat, vhat = shv
        res_s.append(s_corr(shat.squeeze().abs(), s_all[ind].abs()))
        res_h.append(h_corr(Hhat.squeeze(), h[:M, which_class]))
        print(ii)
        print('s_corr', res_s[-1])
        print('h_corr', res_h[-1])
    print(f's mean at db{dm}', sum(res_s)/20)
    print(f'h mean at db{dm}', sum(res_h)/20)
print('done')


