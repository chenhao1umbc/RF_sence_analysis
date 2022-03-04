#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
from skimage.transform import resize

d, s, h = torch.load('/home/chenhao1/Hpython/data/nem_ss/test500M3FT100_xsh.pt')
h, N, F = torch.tensor(h), s.shape[-1], s.shape[-2] # h is M*J matrix, here 6*6
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
    g = torch.rand(1,J,1,8,8).cuda()  # shape of [1,J,8,8]
    x = x.cuda()

    vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
    outs = []
    for j in range(J):
        outs.append(model(g[:,j]))
    out = torch.cat(outs, dim=1).permute(0,2,3,1)
    vhat.real = threshold(out)
    # Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
    _, _ , h = d = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    Hhat = torch.tensor(h).to(torch.cdouble).cuda()
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

ind = 0
rid = 182340
model = f'/home/chenhao1/Hpython/data/nem_ss/models/rid{rid}/model_rid{rid}_41.pt'
shv, g, Rb, loss = nem_func_less(awgn(x_all[ind], 30), J=3, seed=10, model=model, max_iter=301)
shat, Hhat, vhat = shv

c = torch.rand(100,100,3,3).to(torch.cfloat)
for i in range(3):
    c[:,:,i] = shat.squeeze()[...,i][..., None] @ Hhat.squeeze()[None, :, i]
    print(c[:,:,i].norm())

for i in range(3):
    plt.figure()
    plt.imshow(c[:,:,i].squeeze().abs()[...,i]*ratio[ind])
    plt.title('plot of c from NEM')
    plt.colorbar()

for i in range(3):
    plt.figure()
    plt.imshow(shat.squeeze().abs()[...,i])
    plt.colorbar()
    plt.title('plot of s from NEM')

print(h_corr(Hhat.squeeze(), h))