#%%
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
d, s, h = torch.load('../data/nem_ss/test1kM3FT64_xsh_data3.pt')
N, F = s.shape[-1], s.shape[-2] # h is M*J matrix, here 6*6
ratio = d.abs().amax(dim=(1,2,3))
x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
s_all = s.abs().permute(0,2,3,1) 
glr = 0.01

from unet.unet_model import *
class UNetHalf(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
        """
        super().__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.n_ch = 128

        self.up = nn.Sequential(DoubleConv(n_channels, self.n_ch),
                    MyUp(self.n_ch, self.n_ch//2),
                    MyUp(self.n_ch//2, self.n_ch//4),
                    MyUp(self.n_ch//4, self.n_ch//4))
        self.reshape = nn.Sequential(
                    nn.Conv2d(self.n_ch//4, self.n_ch//4, kernel_size=3, padding=(1,2)),
                    nn.BatchNorm2d(self.n_ch//4),
                    nn.LeakyReLU(inplace=True),
                    DoubleConv(self.n_ch//4, self.n_ch//4),
                    DoubleConv(self.n_ch//4, self.n_ch//4))
        self.outc = OutConv(32, n_classes)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        x = self.up(x)
        x = self.reshape(x) 
        x = self.outc(x)
        x = self.sig(x)
        out = x/x.detach().amax(keepdim=True, dim=(-1,-2))
        return out

def cluster_init(x, J=3, K=10, init=1, Rbscale=1e-3, showfig=False):
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
    for i in range(J):
        d = data[torch.tensor(cs[i])] # shape of [I_cj, M]
        Hhat[:,i] = d.mean(0)

    return Hhat

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

    # EM part
    "initial"        
    N, F, M = x.shape
    NF= N*F
    gamma = torch.rand(1,J,1,8,8).cuda()
    x = x.cuda()

    vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
    outs = []
    for j in range(J):
        outs.append(model(gamma[:,j]))
    out = torch.cat(outs, dim=1).permute(0,2,3,1).to(torch.double)
    vhat.real = threshold(out)
    Hhat = cluster_init(x.cpu(), J=J).cuda().to(torch.cdouble)
    Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
    Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
    Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
    Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
    gamma.requires_grad_()
    optim_gamma = torch.optim.RAdam([gamma],
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
            outs.append(model(gamma[:,j]))
        out = torch.cat(outs, dim=1).permute(0,2,3,1).to(torch.double)
        vhat.real = threshold(out, ceiling=1)
        loss = loss_func(vhat, Rsshatnf.cuda())
        optim_gamma.zero_grad()   
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gamma], max_norm=1)
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

    return (shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze()), gamma.detach().cpu(), Rb.cpu(), ll_traj

#%%
rid = 'n4'
model = f'../data/nem_ss/models/{rid}/model_rid{rid}_39.pt'

EMs, EMh = [], []
for snr in ['inf', 20, 10, 5, 0]:
    ems, emh = [], []
    for ind in range(1000):
        if snr != 'inf':
            data = awgn(x_all[ind], snr).to(torch.cdouble)
        else:
            data = x_all[ind].to(torch.cdouble)

        shv, g, Rb, loss = nem_hci(data, J=3, seed=10, model=model, max_iter=301)
        shat, Hhat, vhat = shv
        temp_s = s_corr(shat.squeeze().abs(), s_all[ind].abs())
        temp = h_corr(Hhat.squeeze(), h[ind])
        if ind %20 == 0 :
            print(f'At epoch {ind}', ' h corr: ', temp, ' s corr:', temp_s)
        ems.append(temp_s)
        emh.append(temp)

    EMs.append(sum(ems)/len(ems))
    EMh.append(sum(emh)/len(emh))

    print(f'snr{snr}, EMs, EMh', EMs, EMh)

print('End date time ', datetime.now())

