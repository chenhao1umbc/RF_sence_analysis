#%% loading dependency
import os
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import stft 
from scipy import stats
import itertools

import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
import torch_optimizer as optim

# from torch.utils.tensorboard import SummaryWriter
"make the result reproducible"
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('done loading')


#%% functions
def loss_func(vhat, Rsshatnf, lamb=0):
    """This function is only the Q1 part, which is related to vj
        Q= Q1 + Q2, Q1 = \sum_nf -log(|Rs(n,f)|) - tr{Rsshat_old(n,f)Rs^-1(n,f))}
        loss = -Q1

    Args:
        vhat ([tensor]): [shape of [batch, N, F, J]]
        Rsshatnf ([tensor]): [shape of [batch, N, F, J, J]]

    Returns:
        [scalar]: [-Q1]
    """
    J = vhat.shape[-1]
    shape = vhat.shape[:-1]
    det_Rs = torch.ones(shape).to(vhat.device)
    for j in range(J):
        det_Rs = det_Rs * vhat[..., j]
    p1 = det_Rs.log().sum() 
    p2 = Rsshatnf.diagonal(dim1=-1, dim2=-2)/vhat
    if lamb == 0:
        loss = p1 + p2.sum() 
    else:
        loss = p1 + p2.sum() - lamb*vhat.abs().sum()
    return loss.sum()

def log_lh(x, vhat, Hhat, Rb):
    """ Hhat shape of [I, M, J] # I is NO. of samples, M is NO. of antennas, J is NO. of sources
        vhat shape of [I, N, F, J]
        Rb shape of [I, M, M]
        x shape of [I, N, F, M]
    """
    Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
    Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb 
    Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
    try:
        l = -(np.pi*Rx.det()).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze()
        return l.sum().real, Rs, Rxperm
    except:
        eps = torch.ones(Rx.shape[-1], device=Rx.device).diag_embed()*1e-10
        Rx = Rx + eps
        l = -(np.pi*Rx.det()).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze()
        return l.sum().real, Rs, Rxperm+eps   

def log_likelihood(x, vhat, Hhat, Rb, lamb=0):
    """ Hhat shape of [I, M, J] # I is NO. of samples, M is NO. of antennas, J is NO. of sources
        vhat shape of [I, N, F, J]
        Rb shape of [I, M, M]
        x shape of [I, N, F, M]
    """
    Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
    Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb 
    Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
    if lamb == 0:
        l = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze() 
    else:
        l = lamb*vhat.abs().sum() -(np.pi*mydet(Rx)).log() - \
        (x[..., None, :].conj()@Rx.inverse()@x[..., None]).squeeze() 
    return l.sum().real, Rs, Rxperm

def calc_ll_cpx2(x, vhat, Rj, Rb):
    """ Rj shape of [I, J, M, M]
        vhat shape of [I, N, F, J]
        Rb shape of [I, M, M]
        x shape of [I, N, F, M]
    """
    _, _, M, M = Rj.shape
    I, N, F, J = vhat.shape
    Rcj = vhat.reshape(I, N*F, J) @ Rj.reshape(I, J, M*M)
    Rcj = Rcj.reshape(I, N, F, M, M).permute(1,2,0,3,4)
    Rx = (Rcj + Rb).permute(2,0,1,3,4) # shape of [I, N, F, M, M]
    l = -(np.pi*mydet(Rx)).log() - (x[..., None, :].conj()@Rx.inverse()@x[..., None]).squeeze()
    return l.sum().real

def mydet(x):
    """calc determinant of tensor for the last 2 dimensions,
    suppose x is postive definite hermitian matrix

    Args:
        x ([pytorch tensor]): [shape of [..., N, N]]
    """
    s = x.shape[:-2]
    N = x.shape[-1]
    try:
        l = torch.linalg.cholesky(x)
    except:
        eps = eps = x.abs().max().requires_grad_(False)
        l = torch.linalg.cholesky(x + eps*1e-5*torch.ones(x.shape[:-1], device=x.device).diag_embed())
        print('low rank happend')
    ll = l.diagonal(dim1=-1, dim2=-2)
    res = torch.ones(s).to(x.device)
    for i in range(N):
        res = res * ll[..., i]**2
    return res

def threshold(x, floor=1e-20, ceiling=1e3):
    y = torch.min(torch.max(x, torch.tensor(floor)), torch.tensor(ceiling))
    return y

def em_func(x, J=3, Hscale=1, Rbscale=100, max_iter=501, lamb=0, seed=0, show_plot=False):
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
        # l = -(np.pi*Rx.det()).log() - (x[..., None, :].conj()@Rx.inverse()@x[..., None]).squeeze()
        l = -(np.pi*mydet(Rx)).log() - (x[..., None, :].conj()@Rx.inverse()@x[..., None]).squeeze()
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

    return shat, Hhat, vhat, Rb

def awgn(xx, snr=20, seed=0):
    """
    This function is adding white guassian noise to the given complex signal
    :param x: the given signal with shape of [N, F, Channel]
    :param snr: a float number
    :return:
    """
    SNR = 10 ** (snr / 10.0)
    x = xx.clone()
    np.random.seed(seed)
    if len(x.shape) == 2:        
        Esym = x.norm()**2/ x.numel()
        N0 = (Esym / SNR).item()
        noise = torch.tensor(np.sqrt(N0) * np.random.normal(0, 1, x.shape), device=x.device)
        return x+noise.to(x.dtype)
    else: #len(x.shape) == 3
        N, F, J = x.shape
        for j in range(J):
            Esym = x[:,:,j].norm()**2/ x[:,:,j].numel()
            N0 = (Esym / SNR).item()
            z = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(N*F, 2)).view(np.complex128)
            noise = torch.tensor(np.sqrt(N0)*z, device=x.device).reshape(N, F)
            x[:,:,j] = x[:,:,j] + noise       
        return  x

def awgn_batch(xx, snr, seed=0):
    """
    This function is adding white guassian noise to the given complex signal
    :param x: the given signal with shape of [I, N, F, J(Channel, M actually)]
    :param snr: a float number
    :return:
    """
    SNR = 10 ** (snr / 10.0)
    x = xx.clone()
    np.random.seed(seed)
    I, N, F, J = x.shape
    for i in range(I):
        for j in range(J):
            Esym = x[i,:,:,j].norm()**2/ x[i,:,:,j].numel()
            N0 = (Esym / SNR).item()
            z = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(N*F, 2)).view(np.complex128)
            noise = torch.tensor(np.sqrt(N0)*z, device=x.device).reshape(N, F)
            x[i,:,:,j] = x[i,:,:,j] + noise       
    return  x

def val_run(data, ginit, model, lb, MJbs=(3,3,64), seed=1):
    torch.manual_seed(seed) 
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    EM_iters, N, F = 201, 100, 100
    M, J, bs = MJbs
    NF, I = N*F, ginit.shape[0]

    vtr = torch.randn(N, F, J).abs().to(torch.cdouble).repeat(I, 1, 1, 1)
    Hhat = torch.randn(M, J).to(torch.cdouble).cuda()
    Rbtr = torch.ones(I, M).diag_embed().to(torch.cdouble)*100

    lv = []
    for i, (x,) in enumerate(data): # gamma [n_batch, 4, 4]
        #%% EM part
        vhat = vtr[i*bs:(i+1)*bs].cuda()        
        Rb = Rbtr[i*bs:(i+1)*bs].cuda()
        g = ginit[i*bs:(i+1)*bs].cuda().requires_grad_()

        x = x.cuda()
        optim_gamma = torch.optim.SGD([g], lr=0.001)
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((1,2))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        ll_traj = []

        for ii in range(EM_iters):
            "E-step"
            W = Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() @ Rx.inverse()  # shape of [N, F, I, J, M]
            shat = W.permute(2,0,1,3,4) @ x[...,None]
            Rsshatnf = shat @ shat.transpose(-1,-2).conj() + Rs - (W@Hhat@Rs.permute(1,2,0,3,4)).permute(2,0,1,3,4)
            Rsshat = Rsshatnf.sum([1,2])/NF # shape of [I, J, J]
            Rxshat = (x[..., None] @ shat.transpose(-1,-2).conj()).sum((1,2))/NF # shape of [I, M, J]

            "M-step"
            Hhat = Rxshat @ Rsshat.inverse() # shape of [I, M, J]
            # Hhat = (Rxshat @ Rsshat.inverse()).mean(0) # shape of [M, J]
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
            ll, Rs, Rx = log_likelihood(x, vhat, Hhat, Rb)
            ll_traj.append(ll.item())
            if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
            if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
                print(f'EM early stop at iter {ii}')
                break
        lv.append(ll.item())
        print(f'val batch {i} is done')
    return sum(lv)/len(lv)


def metrics(s, s_hat, noise):
    """This funciton calculates sdr, sir, snr, sar definded in the paper
    Performance Measurement in Blind Audio Source Separation

    Args:
        s (tensor): shape of [J, N,F]
        s_hat ([tensor]): shape of [J, N,F]
        noise ([tensor]): shape of [M, N,F]

    Returns:
        [list of tensors]: sdr, sir, snr, sar
    """
    J, N, F = s.shape
    NF = N*F

    "get s_target"
    s_target = ((s*s_hat.conj()).sum(dim=(-1,-2), keepdim=True) *s)/(s.abs()**2).sum(dim=(-1,-2), keepdim=True)
    "get e_interf"
    PsTimessj_hat = s.clone() # init. Ps times sj_hat
    Rss = s.reshape(J, NF) @ s.reshape(J, NF).conj().t()# Rss is the Gram matrix of the sources [J,J]
    for j in range(J):
        temp = (s_hat[j] * s.conj()).sum(dim=(-1,-2)) #shape of [J]
        c = Rss.inverse()@ temp[None,:].conj().t()
        PsTimessj_hat[j] = (c.t().conj()@s.reshape(J, NF)).reshape(N,F)
    e_interf = PsTimessj_hat - s_target
    "get e_noise"
    e_noise = s.clone() # init. e_noise 
    for j in range(J):
        inner = (s_hat[j] * noise.conj()).sum(dim=(-1,-2))
        temp = inner[:,None,None]*noise / (noise.abs()**2).sum(dim=(-1,-2), keepdim=True)
        e_noise[j] = temp.sum(0)
    "get e_artif"
    e_artif = s_hat - e_noise -PsTimessj_hat

    sdr = (s_target.norm()/(e_interf+e_noise+e_artif).norm()).log10()*20
    sir = (s_target.norm()/e_interf.norm()).log10()*20
    snr = ((s_target+e_interf).norm()/e_noise.norm()).log10()*20
    sar = ((s_target+e_interf+e_noise).norm()/e_artif.norm()).log10()*20
    
    return sdr, sir, snr, sar
#%%
if __name__ == '__main__':
    a, b = torch.rand(3,1).double(), torch.rand(3,1).double()
    x = (a@a.t() + b@b.t()).cuda()

    eps = x.abs().max().requires_grad_(False)
    ll = torch.linalg.cholesky(x + eps*1e-5*torch.ones(x.shape[:-1], device=x.device).diag_embed())
    l = torch.linalg.cholesky(x)
#%%
