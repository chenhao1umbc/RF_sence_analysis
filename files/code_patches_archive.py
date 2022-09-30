"""This is file is coded based on cell mode, 
if True gives each cell an indent, so that each cell could be folded in vs code
"""
#%% load dependency 
if True:
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)

################################################ data #########################################
#%% Prepare real data Jclasses 18ktr, with rangdom AOA, 1000 val, 1000 te
    from utils import *
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    from skimage.transform import resize
    import itertools
    import time

    "raw data processing"
    FT = 64  #48, 64, 80, 100, 128, 200, 256
    var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
    data = {}

    def get_ftdata(data_pool):
        *_, Z = stft(data_pool, fs=4e7, nperseg=FT, boundary=None)
        x = torch.tensor(np.roll(Z, FT//2, axis=1))  # roll nperseg//2
        # x =  x/((x.abs()**2).sum(dim=(1,2),keepdim=True)**0.5)# normalize
        return x.to(torch.cfloat)

    for i in range(6):
        # if i == 2 or i == 3:
        #     temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k_resize2.mat')
        # else:
        #     temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        x = torch.tensor(temp['x'])
        x =  x/((x.abs()**2).sum(dim=(1),keepdim=True)**0.5)# normalize
        data[i] = x
    s1 = get_ftdata(data[0]) # ble [2000,F,T]
    s2 = get_ftdata(data[2]) # fhss1
    s3 = get_ftdata(data[5]) # wifi2
    s = [s1, s2, s3]

    torch.manual_seed(1)
    M, J, I = 3, 3, 20000
    aoa = torch.rand(J ,int(2.5e4))*180 # get more then remove diff_angle<10
    for i in range(J):
        if i == 0:
            id = (aoa[i]- aoa[i-1]).abs() > 10 # ang diff >10
        else:
            id += (aoa[i]- aoa[i-1]).abs() > 10
    aoa = aoa[:, id][:,:I].to(torch.cfloat)/180*np.pi  # [J, I] to radius angle 
    ch = torch.arange(M)[:,None].to(torch.cfloat)*np.pi #[M,1]
    H = (ch@aoa.t().sin()[:,None]*1j).exp() #[I, M, J]

    hall = H.reshape(10,2000,M,J) # this is easier for later processing
    "training data"
    x = []
    for i in range(9):
        temp = 0
        for j in range(J):
            idx = torch.randperm(2000)
            temp += hall[i,:,:,j:j+1]@s[j][idx].reshape(2000,1,-1)
        x.append(temp)
    x = torch.cat(x, dim=0).reshape(-1,M,FT,FT)
    x = awgn_batch(x, snr=40, seed=1) # added white noise
    plt.figure()
    plt.imshow(x[0,0].abs(), aspect='auto', interpolation='None')
    plt.title('One example of 3-component mixture')
    torch.save(x[:18000], f'tr18kM3FT{FT}_data0.pt')

    "val and test data"
    temp = 0
    svaltest = []
    for j in range(J):
        idx = torch.randperm(2000)
        temp += hall[9,:,:,j:j+1]@s[j][idx].reshape(2000,1,-1)
        svaltest.append(s[j][idx])
    valtest = temp.reshape(-1,M,FT,FT)
    valtest = awgn_batch(valtest, snr=40, seed=1) # added white noise
    svaltest = torch.tensor(np.stack(svaltest, axis=1))  #[2000, J, F, T]
    torch.save((valtest[:1000], svaltest[:1000], hall[9,:1000]), f'val1kM3FT{FT}_xsh_data0.pt')
    torch.save((valtest[1000:], svaltest[1000:], hall[9,1000:]), f'test1kM3FT{FT}_xsh_data0.pt')
    print('done')

#%% Prepare data2, fhss1, fhss2 is compressed in other ways to see more clearly
    "Prepare real data Jclasses 18ktr, with rangdom AOA, 1000 val, 1000 te"
    from utils import *
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    from skimage.transform import resize
    import itertools
    import time

    "raw data processing"
    FT = 64  #48, 64, 80, 100, 128, 200, 256
    var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
    data = {}
    for i in range(6):
        if i == 2 or i == 3:
            temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k_resize2.mat')
        else:
            temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        data[i] = temp['x']

    def get_ftdata(data_pool):
        *_, Z = stft(data_pool, fs=4e7, nperseg=FT, boundary=None)
        x = torch.tensor(np.roll(Z, FT//2, axis=1))  # roll nperseg//2
        x =  x/((x.abs()**2).sum(dim=(1,2),keepdim=True)**0.5)# normalize
        return x.to(torch.cfloat)

    s1 = get_ftdata(data[0]) # ble [2000,F,T]
    s2 = get_ftdata(data[2]) # fhss1
    s3 = get_ftdata(data[5]) # wifi2
    s = [s1, s2, s3]

    torch.manual_seed(1)
    M, J, I = 3, 3, 20000
    aoa = torch.rand(J ,int(2.5e4))*180 # get more then remove diff_angle<10
    for i in range(J):
        if i == 0:
            id = (aoa[i]- aoa[i-1]).abs() > 10 # ang diff >10
        else:
            id += (aoa[i]- aoa[i-1]).abs() > 10
    aoa = aoa[:, id][:,:I].to(torch.cfloat)/180*np.pi  # [J, I] to radius angle 
    ch = torch.arange(M)[:,None].to(torch.cfloat)*np.pi #[M,1]
    H = (ch@aoa.t().sin()[:,None]*1j).exp() #[I, M, J]

    hall = H.reshape(10,2000,M,J) # this is easier for later processing
    "training data"
    x = []
    for i in range(9):
        temp = 0
        for j in range(J):
            idx = torch.randperm(2000)
            temp += hall[i,:,:,j:j+1]@s[j][idx].reshape(2000,1,-1)
        x.append(temp)
    x = torch.cat(x, dim=0).reshape(-1,M,FT,FT)
    x = awgn_batch(x, snr=40, seed=1) # added white noise
    plt.figure()
    plt.imshow(x[0,0].abs(), aspect='auto', interpolation='None')
    plt.title('One example of 3-component mixture')
    torch.save(x[:9000], f'tr18kM3FT{FT}_data2.pt')

    "val and test data"
    temp = 0
    svaltest = []
    for j in range(J):
        idx = torch.randperm(2000)
        temp += hall[9,:,:,j:j+1]@s[j][idx].reshape(2000,1,-1)
        svaltest.append(s[j][idx])
    valtest = temp.reshape(-1,M,FT,FT)
    valtest = awgn_batch(valtest, snr=40, seed=1) # added white noise
    svaltest = torch.tensor(np.stack(svaltest, axis=1))  #[2000, J, F, T]
    torch.save((valtest[:1000], svaltest[:1000], hall[9,:1000]), f'val1kM3FT{FT}_xsh_data2.pt')
    torch.save((valtest[1000:], svaltest[1000:], hall[9,1000:]), f'test1kM3FT{FT}_xsh_data2.pt')
    print('done')

#%% Prepare real data3 J=3 classes 18ktr, with rangdom AOA, 1000 val, 1000 te
    from utils import *
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    from skimage.transform import resize
    import itertools
    import time

    "raw data processing"
    FT = 64  #48, 64, 80, 100, 128, 200, 256
    var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
    data = {}

    def get_ftdata(data_pool):
        *_, Z = stft(data_pool, fs=4e7, nperseg=FT, boundary='zeros')
        x = torch.tensor(np.roll(Z, FT//2, axis=1))  # roll nperseg//2
        # x =  x/((x.abs()**2).sum(dim=(1,2),keepdim=True)**0.5)# normalize
        return x.to(torch.cfloat)

    for i in range(6):
        # if i == 2 or i == 3:
        #     temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k_resize2.mat')
        # else:
        #     temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        x = torch.tensor(temp['x'])
        x =  x/((x.abs()**2).sum(dim=(1),keepdim=True)**0.5)# normalize
        data[i] = x
    s1 = get_ftdata(data[0]) # ble [2000,F,T]
    s2 = get_ftdata(data[2]) # fhss1
    s3 = get_ftdata(data[5]) # wifi2
    s = [s1, s2, s3]

    torch.manual_seed(1)
    M, J, I = 3, 3, 20000
    aoa = torch.rand(J ,int(2.5e4))*180 # get more then remove diff_angle<10
    for i in range(J):
        if i == 0:
            id = (aoa[i]- aoa[i-1]).abs() > 10 # ang diff >10
        else:
            id += (aoa[i]- aoa[i-1]).abs() > 10
    aoa = aoa[:, id][:,:I].to(torch.cfloat)/180*np.pi  # [J, I] to radius angle 
    ch = torch.arange(M)[:,None].to(torch.cfloat)*np.pi #[M,1]
    H = (ch@aoa.t().sin()[:,None]*1j).exp() #[I, M, J]

    hall = H.reshape(10,2000,M,J) # this is easier for later processing
    "training data"
    x = []
    for i in range(9):
        temp = 0
        for j in range(J):
            idx = torch.randperm(2000)
            temp += hall[i,:,:,j:j+1]@s[j][idx].reshape(2000,1,-1)
        x.append(temp)
    x = torch.cat(x, dim=0).reshape(-1,M,FT,FT+2)
    x = awgn_batch(x, snr=40, seed=1) # added white noise
    plt.figure()
    plt.imshow(x[0,0].abs(), aspect='auto', interpolation='None')
    plt.title('One example of 3-component mixture')
    torch.save(x[:18000], f'tr18kM3FT{FT}_data3.pt')

    "val and test data"
    temp = 0
    svaltest = []
    for j in range(J):
        idx = torch.randperm(2000)
        temp += hall[9,:,:,j:j+1]@s[j][idx].reshape(2000,1,-1)
        svaltest.append(s[j][idx])
    valtest = temp.reshape(-1,M,FT,FT+2)
    valtest = awgn_batch(valtest, snr=40, seed=1) # added white noise
    svaltest = torch.tensor(np.stack(svaltest, axis=1))  #[2000, J, F, T]
    torch.save((valtest[:1000], svaltest[:1000], hall[9,:1000]), f'val1kM3FT{FT}_xsh_data3.pt')
    torch.save((valtest[1000:], svaltest[1000:], hall[9,1000:]), f'test1kM3FT{FT}_xsh_data3.pt')
    print('done')

#%% Prepare real data4 J=6 classes 18ktr, with rangdom AOA, 1000 val, 1000 te
    from utils import *
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    from skimage.transform import resize
    import itertools
    import time

    "raw data processing"
    FT = 64  #48, 64, 80, 100, 128, 200, 256
    var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
    data = {}

    def get_ftdata(data_pool):
        *_, Z = stft(data_pool, fs=4e7, nperseg=FT, boundary='zeros')
        x = torch.tensor(np.roll(Z, FT//2, axis=1))  # roll nperseg//2
        # x =  x/((x.abs()**2).sum(dim=(1,2),keepdim=True)**0.5)# normalize
        return x.to(torch.cfloat)

    for i in range(6):
        # if i == 2 or i == 3:
        #     temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k_resize2.mat')
        # else:
        #     temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        x = torch.tensor(temp['x'])
        x =  x/((x.abs()**2).sum(dim=(1),keepdim=True)**0.5)# normalize
        data[i] = x

    s = []
    for i in range(6):
        s.append(get_ftdata(data[i])) # ble [2000,F,T]

    torch.manual_seed(1)
    M, J, I = 6, 6, 20000
    aoa = torch.rand(5) # get more then remove diff_angle<10
    ln = 0
    res = []
    combs = torch.combinations(torch.tensor([i for i in range(J)]))
    while ln < I:
        aoa = torch.rand(J ,int(2e4))*180 # get more then remove diff_angle<10
        for i, c in enumerate(combs):
            if i == 0:
                id = (aoa[c[0]]- aoa[c[1]]).abs() > 10 # ang diff >10
            else:
                id = torch.logical_and(id, (aoa[c[0]]- aoa[c[1]]).abs() > 10)
        ln += id.sum()
        res.append(aoa[:,id])
        print('one loop')
    aoa = torch.cat(res, dim=1)
    aoa = aoa[:,:I].to(torch.cfloat)/180*np.pi  # [J, I] to radius angle 
    ch = torch.arange(M)[:,None].to(torch.cfloat)*np.pi #[M,1]
    H = (ch@aoa.t().sin()[:,None]*1j).exp() #[I, M, J]

    hall = H.reshape(10,2000,M,J) # this is easier for later processing
    "training data"
    x = []
    for i in range(9):
        temp = 0
        for j in range(J):
            idx = torch.randperm(2000)
            temp += hall[i,:,:,j:j+1]@s[j][idx].reshape(2000,1,-1)
        x.append(temp)
    x = torch.cat(x, dim=0).reshape(-1,M,FT,FT+2)
    x = awgn_batch(x, snr=40, seed=1) # added white noise
    plt.figure()
    plt.imshow(x[0,0].abs(), aspect='auto', interpolation='None')
    plt.title('One example of 3-component mixture')
    torch.save(x[:18000], f'tr18kM6FT{FT}_data4.pt')

    "val and test data"
    temp = 0
    svaltest = []
    for j in range(J):
        idx = torch.randperm(2000)
        temp += hall[9,:,:,j:j+1]@s[j][idx].reshape(2000,1,-1)
        svaltest.append(s[j][idx])
    valtest = temp.reshape(-1,M,FT,FT+2)
    valtest = awgn_batch(valtest, snr=40, seed=1) # added white noise
    svaltest = torch.tensor(np.stack(svaltest, axis=1))  #[2000, J, F, T]
    torch.save((valtest[:1000], svaltest[:1000], hall[9,:1000]), f'val1kM6FT{FT}_xsh_data4.pt')
    torch.save((valtest[1000:], svaltest[1000:], hall[9,1000:]), f'test1kM6FT{FT}_xsh_data4.pt')
    print('done')

############################################## Testing ########################################
#%% test CNN nem
    import itertools, time
    d = sio.loadmat('../data/nem_ss/100_test_all.mat') 
    "x shape of [I,M,N,F], c [I,M,N,F,J], h [I,M,J]"
    x_all, c_all, h_all = d['x'], d['c_all'], d['h_all']
    d = sio.loadmat('../data/nem_ss/v.mat')
    v = torch.tensor(d['v'], dtype=torch.cdouble) # shape of [N,F,J]

    def mse(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for jj in permutes:
            temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
            s = 0
            for j in range(J):
                s = s + (v[...,j] -temp[j]).norm()**2
            r.append(s.item())
        r = sorted(r)
        return r[0]/J

    def corr(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for jj in permutes:
            temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
            s = 0
            for j in range(J):
                s = s + stats.pearsonr(v[...,j].flatten(), temp[j].flatten())[0]
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

    def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=150, lamb=0, seed=0, model='', show_plot=False):
        if model == '':
            models = torch.load('../../Hpython/data/nem_ss/models/model_4to50_em20_151epoch_1H_100Rb_v1.pt')
        models = torch.load(model)
        for j in range(J):
            models[j].eval()
            for param in models[j].parameters():
                    param.requires_grad_(False)

        #%% EM part
        "initial"        
        N, F, M = x.shape
        NF= N*F
        x = x.cuda()

        torch.torch.manual_seed(seed)        
        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        g = torch.rand(J, 1, 4, 4).cuda().requires_grad_()
        optim_gamma = torch.optim.SGD([g], lr= 0.05)
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
            out = torch.randn(vhat.shape, device='cuda', dtype=torch.double)
            for j in range(J):
                out[..., j] = models[j](g[None,j]).exp().squeeze()
            vhat.real = threshold(out)
            loss = loss_func(vhat, Rsshatnf.cuda(), lamb=lamb)
            optim_gamma.zero_grad()   
            loss.backward()
            torch.nn.utils.clip_grad_norm_([g], max_norm=1)
            optim_gamma.step()
            torch.cuda.empty_cache()
            
            "compute log-likelyhood"
            vhat = vhat.detach()
            ll, Rs, Rx = log_likelihood(x, vhat, Hhat, Rb)
            ll_traj.append(ll.item())
            if torch.isnan(torch.tensor(ll_traj[-1])) : inp('nan happened')
            if ii > 3 and abs((ll_traj[ii] - ll_traj[ii-1])/ll_traj[ii-1]) <1e-3:
                # print(f'EM early stop at iter {ii}')
                break

        if show_plot:
            plt.figure(100)
            plt.plot(ll_traj,'o-')
            plt.show()
            "display results"
            for j in range(J):
                plt.figure(j)
                plt.subplot(1,2,1)
                plt.imshow(vhat[...,j].cpu().squeeze().real)
                plt.colorbar()
        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu()

    I = x_all.shape[0]
    res_mse, res_corr = [], []
    for id in range(1,6):
        location = f'../../Hpython/data/nem_ss/models/model_4to50_21epoch_1H_100Rb_cold_same_M3_v{id}.pt'
        for i in range(I):
            x = torch.from_numpy(x_all[i]).permute(1,2,0)
            MSE, CORR = [], []
            for ii in range(20):  # for diff initializations
                shat, Hhat, vhat, Rb = nem_func(x, seed=ii, model=location, show_plot=False)
                MSE.append(mse(vhat, v))
                CORR.append(corr(vhat.real, v.real))
            res_mse.append(MSE)
            res_corr.append(CORR)
            print(f'finished {i} samples')
        torch.save((res_mse, res_corr), f'nem_CNN_v{id}.pt')

#%% Test FCN nem on static toy
    import itertools
    class Fcn(nn.Module):
        def __init__(self, input, output):
            super().__init__()
            self.fc = nn.Linear(input , output)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x1 = self.fc(x)
            x2 = self.sigmoid(x1)
            return x2

    def mse(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for jj in permutes:
            temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
            s = 0
            for j in range(J):
                s = s + (v[...,j] -temp[j]).norm()**2
            r.append(s.item())
        r = sorted(r)
        return r[0]/J

    def corr(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for jj in permutes:
            temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
            s = 0
            for j in range(J):
                s = s + stats.pearsonr(v[...,j].flatten(), temp[j].flatten())[0]
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

    d = sio.loadmat('../data/nem_ss/100_test_all.mat') 
    "x shape of [I,M,N,F], c [I,M,N,F,J], h [I,M,J]"
    x_all, c_all, h_all = d['x'], d['c_all'], d['h_all']
    d = sio.loadmat('../data/nem_ss/v.mat')
    v = torch.tensor(d['v'], dtype=torch.cdouble) # shape of [N,F,J]

    def nem_fcn(x, J=3, Hscale=1, Rbscale=100, max_iter=150, lamb=0, seed=0, model='', show_plot=False):
        if model == '':
            print('A FCN model is needed')
            return None
        models = torch.load(model)
        for param in models.parameters():
            param.requires_grad_(False)

        #%% EM part
        N, F, M = x.shape
        NF= N*F
        x = x.cuda()

        torch.torch.manual_seed(seed)        
        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        g = torch.rand(1, J, 250).cuda().requires_grad_()
        optim_gamma = torch.optim.SGD([g], lr= 0.01)
        ll_traj = []

        for ii in range(max_iter):
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
            out = torch.randn(vhat.shape, device='cuda', dtype=torch.double)
            out = models(g)
            vhat.real = threshold(out.permute(0,2,1).reshape(1,N,F,J))
            loss = loss_func(vhat, Rsshatnf.cuda())
            optim_gamma.zero_grad()   
            loss.backward()
            torch.nn.utils.clip_grad_norm_([g], max_norm=10)
            optim_gamma.step()
            torch.cuda.empty_cache()
            
            "compute log-likelyhood"
            vhat = vhat.detach()
            ll, Rs, Rx = log_likelihood(x, vhat, Hhat, Rb)
            ll_traj.append(ll.item())
            if ii > 3 and abs((ll_traj[ii] - ll_traj[ii-1])/ll_traj[ii-1]) <1e-3:
                # print(f'EM early stop at iter {ii}')
                break
            if torch.isnan(torch.tensor(ll_traj[-1])) : inp('nan happened')

        if show_plot:
            plt.figure(100)
            plt.plot(ll_traj,'o-')
            plt.show()
            "display results"
            for j in range(J):
                plt.figure(j)
                plt.subplot(1,2,1)
                plt.imshow(vhat[...,j].cpu().squeeze().real)
                plt.colorbar()
                plt.show()
        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu()

    I = x_all.shape[0]
    res_mse, res_corr = [], []
    for id in range(1,6):
        location = f'../../Hpython/data/nem_ss/models/model_FCN_21epoch_1H_100Rb_cold_same_M3_v{id}.pt'
        for i in range(I):
            x = torch.from_numpy(x_all[i]).permute(1,2,0)
            MSE, CORR = [], []
            for ii in range(20):  # for diff initializations
                shat, Hhat, vhat, Rb = nem_fcn(x, seed=ii, model=location, show_plot=False)
                MSE.append(mse(vhat, v))
                CORR.append(corr(vhat.real, v.real))
            res_mse.append(MSE)
            res_corr.append(CORR)
            print(f'finished {i} samples')
        torch.save((res_mse, res_corr), f'nem_FCN_v{id}.pt')

#%% Test EM on dynamic toy
    import itertools
    d = sio.loadmat('../data/nem_ss/test100M3_shift.mat')
    vj_all = torch.tensor(d['vj']).to(torch.cdouble)  # shape of [I, N, F, J]
    x_all = torch.tensor(d['x']).permute(0,2,3,1)  # shape of [I, M, N, F]
    cj = torch.tensor(d['cj'])  # shape of [I, M, N, F, J]

    def corr(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for jj in permutes:
            temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
            s = 0
            for j in range(J):
                s = s + stats.pearsonr(v[...,j].flatten(), temp[j].flatten())[0]
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

    import time
    t = time.time()
    I = x_all.shape[0]
    res_corr = []
    for i in range(I):
        CORR = []
        for ii in range(20):
            shat, Hhat, vhat, Rb = em_func(x_all[i], seed=ii, show_plot=False)
            CORR.append(corr(vhat.real, vj_all[i].real))
            # plt.figure()
            # plt.imshow(vhat[...,0].real)
            # plt.show()
        res_corr.append(CORR)
        print(f'finished {i} samples')
    print('Time used is ', time.time()-t)
    # torch.save(res_corr, 'res_toy100shift.pt')

#%% Test NEM on dynamic toy
    import itertools
    from skimage.transform import resize
    d = sio.loadmat('../data/nem_ss/test100M3_shift.mat')
    vj_all = torch.tensor(d['vj']).to(torch.cdouble)  # shape of [I, N, F, J]
    x_all = torch.tensor(d['x']).permute(0,2,3,1)  # shape of [I, M, N, F]
    cj = torch.tensor(d['cj'])  # shape of [I, M, N, F, J]

    def corr(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for jj in permutes:
            temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

    def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=151, lamb=0, seed=0, model='', show_plot=False):
        if model == '':
            print('A model is needed')
        models = torch.load(model)
        for j in range(J):
            models[j].eval()
            for param in models[j].parameters():
                    param.requires_grad_(False)

        #%% EM part
        "initial"        
        N, F, M = x.shape
        NF= N*F      
        gtr = torch.tensor(resize(x[...,0].abs(), [8,8], order=1, preserve_range=True))
        gtr = (gtr/gtr.max())[None,...]  #standardization shape of [1, 8, 8]
        g = torch.stack([gtr[:,None] for j in range(J)], dim=1).cuda().requires_grad_()
        x = x.cuda()

        torch.manual_seed(seed)        
        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        optim_gamma = torch.optim.SGD([g], lr= 0.05)
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
            out = torch.randn(vhat.shape, device='cuda', dtype=torch.double)
            for j in range(J):
                out[..., j] = torch.sigmoid(models[j](g[:,j]).squeeze())
            vhat.real = threshold(out)
            loss = loss_func(vhat, Rsshatnf.cuda(), lamb=lamb)
            optim_gamma.zero_grad()   
            loss.backward()
            torch.nn.utils.clip_grad_norm_([g], max_norm=1)
            optim_gamma.step()
            torch.cuda.empty_cache()
            
            "compute log-likelyhood"
            vhat = vhat.detach()
            ll, Rs, Rx = log_likelihood(x, vhat, Hhat, Rb)
            ll_traj.append(ll.item())
            if torch.isnan(torch.tensor(ll_traj[-1])) : inp('nan happened')
            if ii > 5 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <1e-3:
                print(f'EM early stop at iter {ii}')
                break

        if show_plot:
            plt.figure(100)
            plt.plot(ll_traj,'o-')
            plt.show()
            "display results"
            for j in range(J):
                plt.figure(j)
                plt.subplot(1,2,1)
                plt.imshow(vhat[...,j].cpu().squeeze().real)
                plt.colorbar()
        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu()

    I = x_all.shape[0]
    res_corr = []
    location = f'../data/nem_ss/models/model_rid8300.pt'
    for i in range(3):
        c = []
        for ii in range(3):
            shat, Hhat, vhat, Rb = nem_func(x_all[i], seed=ii,model=location)
            c.append(corr(vhat.real, vj_all[i].real))
        res_corr.append(c)
        print(f'finished {i} samples')
    # torch.save(res_corr, f'nem_toy_shift.pt')

#%% Test EM on real data
    import itertools
    d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    h = torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    x = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1)

    def corr(vh, v):
        "vh and v are the real value with shape of [N,F,J]"
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = vh[..., torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

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

    single_data = True
    if single_data:
        ind = 0
        shat, Hhat, vhat, Rb = em_func(x[ind])
        for i in range(3):
            plt.figure()
            plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
            plt.colorbar()
            plt.show()
        # sio.savemat('x0s_em.mat', {'x':x[ind].numpy(),'s_em':(shat.squeeze().abs()*ratio[ind]).numpy()})
    else: # run a lot of samples
        res, res2 = [], []
        for i in range(100):
            c, cc = [], []
            for ii in range(20):
                shat, Hhat, vhat, Rb = em_func(awgn(x[i], snr=20), seed=ii)
                c.append(corr(shat.squeeze().abs(), s_all[i]))
                cc.append(h_corr(h, Hhat))
            res.append(c)
            res2.append(cc)
            print(f'finished {i} samples')

#%% Test NEM on real data
    from unet.unet_model import UNetHalf8to100 as UNetHalf
    from skimage.transform import resize
    import itertools
    import time
    t = time.time()
    d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    h = torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1) 

    def corr(vh, v):
        "vh and v are the real value with shape of [N,F,J]"
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = vh[..., torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

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
    
    def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=151, lamb=0, seed=1, model='', show_plot=False):
        torch.manual_seed(seed) 
        if model == '':
            print('A model is needed')
        models = {}  # the following 3 lines are matching training initials
        for j in range(J): #---- see above ---
            models[j] = UNetHalf(1, 1) #---- see above ---
        del models #---- see above ---

        models = torch.load(model)
        for j in range(J):
            models[j].eval()
            for param in models[j].parameters():
                    param.requires_grad_(False)

        #%% EM part
        "initial"        
        N, F, M = x.shape
        NF= N*F
        gtr = torch.tensor(resize(x[...,0].abs(), [8,8], order=1, preserve_range=True))
        gtr = (gtr/gtr.max())[None,...]  #standardization shape of [1, 8, 8]
        g = torch.stack([gtr[:,None] for j in range(J)], dim=1).cuda().requires_grad_()
        x = x.cuda()

        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
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
            out = torch.randn(vhat.shape, device='cuda', dtype=torch.double)
            for j in range(J):
                out[..., j] = torch.sigmoid(models[j](g[:,j]).squeeze())
            vhat.real = threshold(out)
            loss = loss_func(vhat, Rsshatnf.cuda(), lamb=lamb)
            optim_gamma.zero_grad()   
            loss.backward()
            torch.nn.utils.clip_grad_norm_([g], max_norm=1)
            optim_gamma.step()
            torch.cuda.empty_cache()
            
            "compute log-likelyhood"
            vhat = vhat.detach()
            ll, Rs, Rx = log_likelihood(x, vhat, Hhat, Rb)
            ll_traj.append(ll.item())
            if torch.isnan(torch.tensor(ll_traj[-1])) : inp('nan happened')
            if ii > 5 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <1e-3:
                print(f'EM early stop at iter {ii}')
                break

        if show_plot:
            plt.figure(100)
            plt.plot(ll_traj,'o-')
            plt.show()
            "display results"
            for j in range(J):
                plt.figure(j)
                plt.subplot(1,2,1)
                plt.imshow(vhat[...,j].cpu().squeeze().real)
                plt.colorbar()
        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu()
    
    location = f'../data/nem_ss/models/model_rid5200.pt'
    single_data = True
    if single_data:
        ind = 0
        shat, Hhat, vhat, Rb = nem_func(awgn(x_all[ind], snr=0),seed=1,model=location)
        for i in range(3):
            plt.figure()
            plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
            plt.colorbar()
            # plt.title(f'Estimated sources {i+1}')
            plt.show()
        print(h_corr(h, Hhat.squeeze()))
        
        for i in range(3):
            plt.figure()
            plt.imshow(s_all[ind].squeeze().abs()[...,i])
            plt.colorbar()
            plt.title(f'GT sources {i+1}')
            plt.show()
        # sio.savemat('sshat_nem.mat', {'s':s_all[ind].squeeze().abs().numpy(),'s_nem':(shat.squeeze().abs()*ratio[ind]).numpy()})
    else: # run a lot of samples
        res, res2 = [], []
        for i in range(100):
            c, cc = [], []
            for ii in range(20):
                shat, Hhat, vhat, Rb = nem_func(x_all[i],seed=ii,model=location)
                c.append(corr(shat.squeeze().abs(), s_all[i]))
                cc.append(h_corr(h, Hhat.squeeze()))
            res.append(c)
            res2.append(cc)
            print(f'finished {i} samples')
        print('Time used is ', time.time()-t)
        torch.save([res, res2], 'res_nem_shat_hhat.pt')

#%% Test 1 channel 1 model NEM
    "best ones rid 135100/125240/135110"
    from skimage.transform import resize
    import itertools
    import time
    t = time.time()
    d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    h = torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1) 

    def corr(vh, v):
        "vh and v are the real value with shape of [N,F,J]"
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = vh[..., torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

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

    def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=501, seed=1, model=''):
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
        graw = torch.stack([graw[:,None] for j in range(J)], dim=1)  # shape of [1,J,8,8]
        noise = torch.rand(J,1,8,8)
        for j in range(J):
            noise[j,0] = awgn(graw[0,j,0], snr=10, seed=j) - graw[0,j,0]
        g = (graw + noise).cuda().requires_grad_()
        x = x.cuda()

        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        out = torch.randn(vhat.shape, device='cuda', dtype=torch.double)
        for j in range(J):
            out[..., j] = torch.sigmoid(model(g[:,j]).squeeze())
        vhat.real = threshold(out)
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
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
            out = torch.randn(vhat.shape, device='cuda', dtype=torch.double)
            for j in range(J):
                out[..., j] = torch.sigmoid(model(g[:,j]).squeeze())
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

        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu()
    
    location = f'../data/nem_ss/models/model_rid135110.pt'
    single_data = False
    if single_data:
        ind = 70
        shat, Hhat, vhat, Rb = nem_func(x_all[ind],seed=10,model=location)
        for i in range(3):
            plt.figure()
            plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
            plt.colorbar()
            plt.title(f'Estimated sources {i+1}')
            plt.show()
        print('h correlation:', h_corr(h, Hhat.squeeze()))
        print('s correlation:', corr(shat.squeeze().abs(), s_all[ind]))
        
        for i in range(3):
            plt.figure()
            plt.imshow(s_all[ind].squeeze().abs()[...,i])
            plt.colorbar()
            plt.title(f'GT sources {i+1}')
            plt.show()
        # sio.savemat('sshat_nem.mat', {'s':s_all[ind].squeeze().abs().numpy(),'s_nem':(shat.squeeze().abs()*ratio[ind]).numpy()})
    else: # run a lot of samples
        res, res2 = [], []
        for i in range(100):
            c, cc = [], []
            for ii in range(20):
                shat, Hhat, vhat, Rb = nem_func(x_all[i],seed=ii,model=location)
                c.append(corr(shat.squeeze().abs(), s_all[i]))
                cc.append(h_corr(h, Hhat.squeeze()))
            res.append(c)
            res2.append(cc)
            print(f'finished {i} samples')
        print('Time used is ', time.time()-t)
        torch.save([res, res2], 'res_nem_shat_hhat_rid135110.pt')

#%% Test 2 channel model 1 model NEM
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    from skimage.transform import resize
    import itertools
    import time
    t = time.time()
    d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    h = torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1) 

    def corr(vh, v):
        "vh and v are the real value with shape of [N,F,J]"
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = vh[..., torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

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

    def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=501, seed=1, model=''):
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
        lb = torch.load('../data/nem_ss/140100_lb.pt')
        lb = lb[None,...].cuda()
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

        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu(), ll_traj

    rid = 149000
    location = f'../data/nem_ss/models/rid{rid}/model_rid{rid}_38.pt'
    single_data = False
    if single_data:
        ind = 43
        shat, Hhat, vhat, Rb, loss = nem_func(x_all[ind],seed=10,model=location)
        for i in range(3):
            plt.figure()
            plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
            plt.colorbar()
            plt.title(f'Estimated sources {i+1}')
            plt.show()
        print('h correlation:', h_corr(h, Hhat.squeeze()))
        print('s correlation:', corr(shat.squeeze().abs(), s_all[ind]))
        
        for i in range(3):
            plt.figure()
            plt.imshow(s_all[ind].squeeze().abs()[...,i])
            plt.colorbar()
            plt.title(f'GT sources {i+1}')
            plt.show()

        plt.figure()
        plt.plot(loss, '-x')
        plt.title('loss value')
        plt.show()
    else: # run a lot of samples
        res, res2 = [], []
        for i in range(100):
            c, cc = [], []
            for ii in range(10):
                shat, Hhat, *_ = nem_func(awgn(x_all[i], snr=20),seed=ii,model=location)
                c.append(corr(shat.squeeze().abs(), s_all[i]))
                cc.append(h_corr(h, Hhat.squeeze()))
            res.append(c)
            res2.append(cc)
            print(f'finished {i} samples')
        print('Time used is ', time.time()-t)
        torch.save([res, res2], f'res_10seed_rid{rid}_snr20.pt')

#%% Test 2 channel model 1 model NEM, with mini batch
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 100
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    import itertools
    from unet.unet_model import UNetHalf8to100_256_sig as UNetHalf
    from datetime import datetime
    print('starting date time ', datetime.now())
    torch.manual_seed(1)

    I, J, bs = 150, 3, 64 # I should be larger than bs
    d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    s_all, h = s.abs().permute(0,2,3,1), torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    xte = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]*3).permute(0,2,3,1)# [sample, N, F, channel]
    xte = awgn_batch(xte[:I], snr=1000)
    data = Data.TensorDataset(xte)
    data_test = Data.DataLoader(data, batch_size=bs, drop_last=True)
    from skimage.transform import resize
    gte = torch.tensor(resize(xte[...,0].abs(), [I,8,8], order=1, preserve_range=True ))
    # g = torch.load('../data/nem_ss/gtest_500.pt') # preload g only works for no noise case
    gte = gte[:I]/gte[:I].amax(dim=[1,2])[...,None,None]  #standardization 
    gte = torch.cat([gte[:,None] for j in range(J)], dim=1)[:,:,None] # shape of [I_val,J,1,8,8]
    
    l = torch.load('../data/nem_ss/140100_lb.pt')
    lb = l.repeat(bs, 1, 1, 1, 1).cuda()

    def corr(vh, v):
        "vh and v are the real value with shape of [N,F,J]"
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = vh[..., torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

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
    
    def nem_minibatch_test(data, ginit, model, lb, bs, seed=1):
        torch.manual_seed(seed) 
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        EM_iters = 501
        M, N, F, J = 3, 100, 100, 3
        NF, I = N*F, ginit.shape[0]

        vtr = torch.randn(N, F, J).abs().to(torch.cdouble).repeat(I, 1, 1, 1)
        Hhat = torch.randn(M, J).to(torch.cdouble).cuda()
        Rbtr = torch.ones(I, M).diag_embed().to(torch.cdouble)*100

        lv, s, h, v, ll_all = ([] for i in range(5)) 
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
                Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
                Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb 
                Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
                l = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze()
                ll = l.sum().real
                Rx = Rxperm

                ll_traj.append(ll.item())
                if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
                if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
                    print(f'EM early stop at iter {ii}')
                    break
            ll_all.append(l.sum((-1, -2)).cpu().real)
            lv.append(ll.item())
            s.append(shat)
            h.append(Hhat)
            v.append(vhat)
            print(f'batch {i} is done')
        return sum(lv)/len(lv), torch.cat(ll_all), (s, h, v)

    rid = 150000
    model = torch.load(f'../data/nem_ss/models/rid{rid}/model_rid{rid}_35.pt')
    meanl, l_all, shv = nem_minibatch_test(data_test, gte, model, lb, bs, seed=1)
    print('End date time ', datetime.now())

    shat, hhat, vhat = shv
    shat_all, hhat_all = torch.cat(shat).cpu(), torch.cat(hhat).cpu()
    res_s, res_h = [], []
    for i in range(100):
        res_s.append(corr(shat_all[i].squeeze().abs(), s_all[i]))
        res_h.append(h_corr(h, hhat_all[i].squeeze()))
    print(sum(res_s)/len(res_s))
    print(sum(res_h)/len(res_h))

#%% Test 2 channel model 1 model NEM, with mini batch for 6 classes
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 100
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    import itertools
    from datetime import datetime
    print('starting date time ', datetime.now())
    torch.manual_seed(1)

    I, J, bs = 130, 6, 32 # I should be larger than bs
    d, s, h = torch.load('../data/nem_ss/test500M6FT100_xsh.pt')
    s_all, h = s.abs().permute(0,2,3,1), torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    xte = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]*3).permute(0,2,3,1)# [sample, N, F, channel]
    xte = awgn_batch(xte[:I], snr=1000)
    data = Data.TensorDataset(xte)
    data_test = Data.DataLoader(data, batch_size=bs, drop_last=True)

    from skimage.transform import resize
    gte = torch.tensor(resize(xte[...,0].abs(), [I,8,8], order=1, preserve_range=True ))
    gte = gte[:I]/gte[:I].amax(dim=[1,2])[...,None,None]  #standardization 
    gte = torch.cat([gte[:,None] for j in range(J)], dim=1)[:,:,None] # shape of [I_val,J,1,8,8]
    l = torch.load('../data/nem_ss/lb_c6_J188.pt')
    lb = l.repeat(bs, 1, 1, 1, 1).cuda()
    
    def corr(vh, v):
        "vh and v are the real value with shape of [N,F,J]"
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = vh[..., torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

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

    def nem_minibatch_test(data, ginit, model, lb, bs, seed=1):
        torch.manual_seed(seed) 
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        EM_iters = 501
        M, N, F, J = 6, 100, 100, 6
        NF, I = N*F, ginit.shape[0]

        vtr = torch.randn(N, F, J).abs().to(torch.cdouble).repeat(I, 1, 1, 1)
        Hhat = torch.randn(M, J).to(torch.cdouble).cuda()
        Rbtr = torch.ones(I, M).diag_embed().to(torch.cdouble)*100

        lv, s, h, v, ll_all = ([] for i in range(5)) 
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
                Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
                Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb 
                Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
                l = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze()
                ll = l.sum().real
                Rx = Rxperm

                ll_traj.append(ll.item())
                if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
                if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
                    print(f'EM early stop at iter {ii}')
                    break
            ll_all.append(l.sum((-1, -2)).cpu().real)
            lv.append(ll.item())
            s.append(shat)
            h.append(Hhat)
            v.append(vhat)
            print(f'batch {i} is done')
        return sum(lv)/len(lv), torch.cat(ll_all), (s, h, v)

    rid = 160001
    model = torch.load(f'../data/nem_ss/models/rid{rid}/model_rid{rid}_41.pt')
    meanl, l_all, shv = nem_minibatch_test(data_test, gte, model, lb, bs, seed=1)
    print('End date time ', datetime.now())

    shat, hhat, vhat = shv
    shat_all, hhat_all = torch.cat(shat).cpu(), torch.cat(hhat).cpu()
    res_s, res_h = [], []
    for i in range(100):
        res_s.append(corr(shat_all[i].squeeze().abs(), s_all[i]))
        res_h.append(h_corr(h, hhat_all[i].squeeze()))
        print(f'{i}-th sample is done')
    print(sum(res_s)/len(res_s))
    print(sum(res_h)/len(res_h))

#%% Test 2 channel model 1 model NEM, gamma=noise as label -- abandoned
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    from skimage.transform import resize
    import itertools
    import time
    t = time.time()
    d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    h = torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1) 

    def corr(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for jj in permutes:
            temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

    def h_corr(h, hh):
        J = h.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = hh[:,torch.tensor(p)]
            s = 0
            for j in range(J):
                dino = h[:,j].norm() * temp[:, j].norm()
                nume = (temp[:, j].conj() @ h[:, j]).abs()
                s = s + nume/dino
            r.append(s/J)
        r = sorted(r, reverse=True)
        return r[0].item()

    def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=501, seed=1, model=''):
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
        xx = torch.tensor(resize(x[...,0].abs(), [8,8], order=1, preserve_range=True))
        xx = (xx/xx.max())[None,None,...].cuda()  #standardization shape of [1, 1, 8, 8]
        g = torch.load('../data/nem_ss/g_init.pt')  # shape of [J, 1, 8, 8]
        g = g[None,...].cuda() # do not put requires_grad_ here
        x = x.cuda()

        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        outs = []
        for j in range(J):
            outs.append(torch.sigmoid(model(torch.cat((g[:,j], xx), dim=1))))
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
                outs.append(torch.sigmoid(model(torch.cat((g[:,j], xx), dim=1))))
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
        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu()

    location = f'../data/nem_ss/models/model_rid141101.pt'
    # location = f'../data/nem_ss/models/rid141103/model_rid141103_58.pt'
    single_data = True
    if single_data:
        ind = 43
        shat, Hhat, vhat, Rb = nem_func(x_all[ind],seed=10,model=location)
        for i in range(3):
            plt.figure()
            plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
            plt.colorbar()
            plt.title(f'Estimated sources {i+1}')
            plt.show()
        print('h correlation:', h_corr(h, Hhat.squeeze()))
        print('s correlation:', corr(shat.squeeze().abs(), s_all[ind]))
        
        for i in range(3):
            plt.figure()
            plt.imshow(s_all[ind].squeeze().abs()[...,i])
            plt.colorbar()
            plt.title(f'GT sources {i+1}')
            plt.show()
        # sio.savemat('sshat_nem.mat', {'s':s_all[ind].squeeze().abs().numpy(),'s_nem':(shat.squeeze().abs()*ratio[ind]).numpy()})
    else: # run a lot of samples
        res, res2 = [], []
        for i in range(100):
            c, cc = [], []
            for ii in range(10):
                shat, Hhat, vhat, Rb = nem_func(awgn(x_all[i], snr=20),seed=ii,model=location)
                c.append(corr(shat.squeeze().abs(), s_all[i]))
                cc.append(h_corr(h, Hhat.squeeze()))
            res.append(c)
            res2.append(cc)
            print(f'finished {i} samples')
        print('Time used is ', time.time()-t)
        torch.save([res, res2], 'res_10seed_rid141103_58_snr20.pt')

#%% NEM test 3-mixture from 6 mixture model
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    from skimage.transform import resize
    import itertools
    import time
    t = time.time()
    d, s, h = torch.load('../data/nem_ss/test500M6FT100_xsh.pt')
    h, N, F, M= torch.tensor(h), s.shape[-1], s.shape[-2], d.shape[1]
    ratio = d.abs().amax(dim=(1,2,3))/3
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1) 

    def nem_func_less(x, J=6, Hscale=1, Rbscale=100, max_iter=501, seed=1, model=''):
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

        return (shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze()), g.detach().cpu(), Rb.cpu(), ll_traj

    rid = 160100
    location = f'model_rid{rid}_33.pt'
    ind = 10 # which sample to test
    J, which_class = 6, [0, 2, 5]  # J is NO. of class you guess, which_class is really there

    "prep data"
    for i, v in enumerate(which_class):
        if i == 0 : d = 0
        d = d + h[:, v, None] @ s[ind, v].reshape(1, N*F)
    d = d.reshape(M, N, F).permute(1,2,0)
    
    shv, g, Rb, loss = nem_func_less(d, J=J, seed=10, model=location)
    shat, Hhat, vhat = shv
    for i in range(6):
        plt.figure()
        plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
        plt.colorbar()
        plt.title(f'Estimated sources {i+1}')
        plt.show()

    for i, v in enumerate(which_class):
        plt.figure()
        plt.imshow(s[ind, v].abs())
        plt.colorbar()
        plt.title(f'GT sources {i+1}')
        plt.show()

    plt.figure()
    plt.plot(loss, '-x')
    plt.title('loss value')
    plt.show()

    plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(g[0,i,0].abs())
        plt.colorbar()
        plt.tight_layout(pad=1.2)
        plt.title(f'gamma of source {i+1}',y=1.2)

    graw = torch.tensor(resize(d[...,0].abs(), [8,8], order=1, preserve_range=True))
    graw = (graw/graw.max())[None,...]  #standardization shape of [1, 8, 8]
    plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(g[0,i,0].abs() - graw[0].abs())
        plt.colorbar(fraction=0.046)
        plt.tight_layout(pad=1.7)    
        # plt.title(f'gamma diff of source {i+1}',y=1.2)

#%% EM for M=6 > real J, to get correct J
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    from skimage.transform import resize
    import itertools
    import time
    data, s, h = torch.load('../data/nem_ss/test500M6FT100_xsh.pt')
    h, N, F, M = torch.tensor(h), s.shape[-1], s.shape[-2], 6

    def em_func_(x, J=3, Hscale=1, Rbscale=100, max_iter=501, lamb=0, seed=0, show_plot=False):
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

    t = time.time()
    res = []
    for JJ in range(1, 7):
        r = []
        # ind = 0 # which sample to test
        # J, which_class = 6, [0,2,5]  # J is NO. of class you guess, which_class is really there
        comb = list(itertools.combinations(range(6), JJ))
        for which_class in comb:
            for ind in range(100): 
                for i, v in enumerate(which_class):
                    if i == 0 : d = 0
                    d = d + h[:M, v, None] @ s[ind, v].reshape(1, N*F)
                d = d.reshape(M, N, F).permute(1,2,0)/d.abs().max()

                shat, Hhat, vhat, Rb, ll_traj, rank = em_func_(d, J=6, max_iter=10)
                r.append(rank)
            print('one comb is done', which_class)
        res.append(r)
        torch.save(res, 'res.pt')
    print('done', time.time()-t)

#%% NEM M=6 > real J, get hhat
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
    h, N, F, M = torch.tensor(h), s.shape[-1], s.shape[-2], 6
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
            vhat.real = threshold(out)
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

    rid = 160100
    model = f'model_rid{rid}_33.pt'
    t = time.time()
    res = []
    for JJ in range(2, 7):
        r = []
        comb = list(itertools.combinations(range(6), JJ))
        for which_class in comb:
            for ind in range(100): 
                for i, v in enumerate(which_class):
                    if i == 0 : d = 0
                    d = d + h[:M, v, None] @ s[ind, v].reshape(1, N*F)
                d = d.reshape(M, N, F).permute(1,2,0)/d.abs().max()

                shv, g, Rb, loss = nem_func_less(d, J=6, seed=10, model=model, max_iter=301)
                shat, Hhat, vhat = shv
                r.append(Hhat)
            print('one comb is done', which_class)
        res.append(r)
        torch.save(res, 'Hhat_2-6comb_res.pt')
    print('done', time.time()-t)


#%% NEM train 6 test on 2,3,4,5 mixture
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
    h, N, F, M = torch.tensor(h), s.shape[-1], s.shape[-2], 6
    ratio = d.abs().amax(dim=(1,2,3))/3
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1) 

    def s_corr(vh, v):
        "vh and v are the real value with shape of [N,F,J]"
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = vh[..., torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

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
            vhat.real = threshold(out)
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

    rid = 160100
    model = f'../data/nem_ss/models/rid{rid}/model_rid{rid}_33.pt'
    t = time.time()
    res_s, res_h = [], []
    JJ =2
    comb = list(itertools.combinations(range(6), JJ))
    for which_class in comb:
        h_r, s_r = [], []
        for ind in range(10): 
            for i, v in enumerate(which_class):
                if i == 0 : d = 0
                d = d + h[:M, v, None] @ s[ind, v].reshape(1, N*F)
            d = d.reshape(M, N, F).permute(1,2,0)/d.abs().max()

            shv, g, Rb, loss = nem_func_less(d, J=JJ, seed=10, model=model, max_iter=301)
            shat, Hhat, vhat = shv
            s_r.append(s_corr(s[ind, which_class].permute(1,2,0).abs(), shat.squeeze().abs()))
            h_r.append(h_corr(h[:, which_class], Hhat.squeeze()))
        res_s.append(sum(s_r)/len(s_r))
        res_h.append(sum(h_r)/len(h_r))
        print('one comb is done', which_class)
    torch.save([res_s, res_h], f'nem_sh_J{JJ}.pt')
    print('done', time.time()-t)

#%% EM train 6 test on 2,3,4,5 mixture
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
    h, N, F, M = torch.tensor(h), s.shape[-1], s.shape[-2], 6
    ratio = d.abs().amax(dim=(1,2,3))/3
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1) 

    def s_corr(vh, v):
        "vh and v are the real value with shape of [N,F,J]"
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = vh[..., torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

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

    rid = 160100
    model = f'../data/nem_ss/models/rid{rid}/model_rid{rid}_33.pt'
    t = time.time()
    res_s, res_h = [], []
    JJ = 2
    comb = list(itertools.combinations(range(6), JJ))
    for which_class in comb:
        h_r, s_r = [], []
        for ind in range(10): 
            for i, v in enumerate(which_class):
                if i == 0 : d = 0
                d = d + h[:M, v, None] @ s[ind, v].reshape(1, N*F)
            d = d.reshape(M, N, F).permute(1,2,0)/d.abs().max()

            shat, Hhat, vhat, Rb, ll_traj, rank = em_func_(awgn(d, 30), J=JJ, max_iter=300)

            s_r.append(s_corr(s[ind, which_class].permute(1,2,0).abs(), shat.squeeze().abs()))
            h_r.append(h_corr(h[:, which_class], Hhat.squeeze()))
        res_s.append(sum(s_r)/len(s_r))
        res_h.append(sum(h_r)/len(h_r))
        print('one comb is done', which_class)
    torch.save([res_s, res_h], f'em_sh_J{JJ}.pt')
    print(sum(res_s)/len(res_s), sum(res_h)/len(res_h))
    print('done', time.time()-t)

    "raw data processing"
    FT = 100
    var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
    data = {}
    for i in range(6):
        temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        dd = (np.sum((abs(temp['x'])**2), 1)**0.5).reshape(2000, 1)
        data[i] = temp['x'] / dd  # normalized very sample to 1

    np.set_printoptions(linewidth=150)
    "To generate 5000 mixture samples"
    M, I_comb = 6, 50# M is number of Channel, I_comb is how many samples per combination
    theta = np.array([15, 45, 75, -15, -45, -75])*np.pi/180  #len=J, signal AOAs  
    h = np.exp(-1j*np.pi*np.arange(0, M)[:,None]@np.sin(theta)[None, :])  # shape of [M, J]

    data_pool, lbs = [], []
    for J in range(2, 7):
        combs = list(itertools.combinations(range(6), J))
        for i, lb in enumerate(combs):
            np.random.seed(i+10)  # val i+5, te i+10, run from scratch
            d = 0
            for ii in range(J):
                np.random.shuffle(data[lb[ii]])
            for ii in range(J):
                d = d + h[:,lb[ii]][:,None]@data[lb[ii]][:I_comb,None,:] 
            data_pool.append(d)
            lbs.append(lb)
        print(J)
    data_all = np.concatenate(data_pool, axis=0)  #[I,M,time_len]
    *_, Z = stft(data_all, fs=4e7, nperseg=FT, boundary=None)
    x = torch.tensor(np.roll(Z, FT//2, axis=2))  # roll nperseg//2
    plt.figure()
    plt.imshow(x[0,0].abs().log(), aspect='auto', interpolation='None')
    plt.title(f'One example of {J}-component mixture')
    # torch.save((x,lbs), f'weakly50percomb_tr3kM{M}FT{FT}_xlb.pt')

    #"get s and h for the val and test data"
    import itertools
    "raw data processing"
    data = {}
    for i in range(6):
        temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        dd = (np.sum((abs(temp['x'])**2), 1)**0.5).reshape(2000, 1)
        data[i] = temp['x'] / dd  # normalized very sample to 1

    s_pool, lbs = [], []
    for J in range(2, 7):
        combs = list(itertools.combinations(range(6), J))
        for i, lb in enumerate(combs):
            np.random.seed(i+10)  # val i+5, te i+10, run from scratch
            for ii in range(J):
                np.random.shuffle(data[lb[ii]])
            for ii in range(J):
                d = data[lb[ii]][:I_comb,:] 
                s_pool.append(d.copy())
            lbs.append(lb)
        print(J)
    data_all = np.concatenate(s_pool, axis=0)  #[I,M,time_len]
    *_, Z = stft(data_all, fs=4e7, nperseg=FT, boundary=None)
    s = torch.tensor(np.roll(Z, FT//2, axis=1))  # roll nperseg//2
    plt.figure()
    plt.imshow(s[0].abs().log(), aspect='auto', interpolation='None')
    plt.title(f'One example of {J}-component mixture')
    # torch.save((x,s,lbs), f'weakly50percomb_te3kM{M}FT{FT}_xslb.pt')
    #%% check s 
    "x.shape is [2850, 6, 100, 100], which means 2850=57*50, 57=15+20+15+6+1"
    "s.shape is [9300, 100, 100], which means 9300=186*50, 186=15*2+20*3+15*4+6*6+1*6"
    J = 2
    combs = list(itertools.combinations(range(6), J))
    print(combs)
    xr = x.reshape(57,50,6,100,100)
    sr = s.reshape(186,50,100,100)

    wc = 2
    ind = 10
    plt.figure()
    plt.imshow(xr[wc,ind,0].abs())

    plt.figure()
    if wc<15:
        for i in range(2):
            plt.figure()
            plt.imshow(sr[wc*2+i,ind].abs())


############################################## Others #########################################
#%% plot the EM vs EM_l1 results
    all_lamb = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for lamb in all_lamb:
        mse, corr = torch.load(f'../data/nem_ss/lamb/lamb_{lamb}.pt')
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(range(1, 101), torch.tensor(mse).mean(dim=1))
        plt.boxplot(mse, showfliers=True)        
        plt.legend(['Mean is blue'])
        # plt.ylim([340, 360])
        plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
        plt.xlabel('Sample index')
        plt.title(f'MSE result for lambda={lamb}')

        plt.subplot(2,1,2)
        plt.plot(range(1, 101), torch.tensor(corr).mean(dim=1))
        plt.boxplot(corr, showfliers=False)        
        plt.legend(['Mean is blue'])
        plt.ylim([0.5, 0.8])
        plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
        plt.xlabel('Sample index')
        plt.title(f'Correlation result for lambda={lamb}')

        plt.subplots_adjust(hspace=0.7)
        plt.savefig(f'lambda{lamb}.png')
        plt.show()

    m, c = {'mean':[], 'std':[]}, {'mean':[], 'std':[]}
    for lamb in all_lamb:
        mse, corr = torch.load(f'../data/nem_ss/lamb/lamb_{lamb}.pt')
        m['mean'].append(torch.tensor(mse).mean())
        m['std'].append(torch.tensor(mse).var()**0.5)
        c['mean'].append(torch.tensor(corr).mean())
        c['std'].append(torch.tensor(corr).var()**0.5)
    plt.figure()
    plt.plot(range(len(all_lamb)),np.log(m['mean']), '-x')
    plt.xticks(ticks=range(len(all_lamb)), \
        labels=('0','1e-3','0.01','0.1','1','10','100','1e3'))
    plt.xlabel('Lambda')
    plt.title('Mean of MSE')
    plt.savefig('Mean of MSE.png')

    plt.figure()
    plt.xticks(ticks=range(len(all_lamb)), \
    labels=('0','1e-3','0.01','0.1','1','10','100','1e3'))
    plt.xlabel('Lambda')
    plt.plot(np.log(m['std']), '-x')
    plt.title('STD of MSE')
    plt.savefig('STD of MSE.png')

    plt.figure()
    plt.errorbar(range(len(all_lamb)),np.log(m['mean']), abs(np.log(m['std'])), capsize=4)
    plt.xticks(ticks=range(len(all_lamb)), \
        labels=('0','1e-3','0.01','0.1','1','10','100','1e3'))
    plt.xlabel('Lambda')
    plt.title('Mean of MSE with std')
    plt.savefig('Mean of MSE with std.png')


    plt.figure()
    plt.plot(c['mean'],'-x')
    plt.xticks(ticks=range(len(all_lamb)), \
    labels=('0','1e-3','0.01','0.1','1','10','100','1e3'))
    plt.xlabel('Lambda')
    plt.title('Mean of Corr.')
    plt.savefig('Mean of Corr.png')

    plt.figure()
    plt.plot(c['std'], '-x')
    plt.xticks(ticks=range(len(all_lamb)), \
    labels=('0','1e-3','0.01','0.1','1','10','100','1e3'))
    plt.xlabel('Lambda')
    plt.title('STD of Corr.')
    plt.savefig('STD of Corr.png')

    plt.figure()
    plt.errorbar(range(len(all_lamb)),c['mean'], c['std'], capsize=4)
    plt.xticks(ticks=range(len(all_lamb)), \
        labels=('0','1e-3','0.01','0.1','1','10','100','1e3'))
    plt.xlabel('Lambda')
    plt.title('Mean of Corr with std')
    plt.savefig('Mean of Corr with std.png')

#%% plot nem results
    for i in range(1,6):
        mse, corr = torch.load(f'../data/nem_ss/nem_res/nem_20iter_v{i}.pt')
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(range(1, 101), torch.tensor(mse[-100:]).mean(dim=1))
        plt.boxplot(mse[-100:], showfliers=True)        
        plt.legend(['Mean is blue'])
        # plt.ylim([340, 360])
        plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
        plt.xlabel('Sample index')
        plt.title(f'MSE result for NEM-{i}')

        plt.subplot(2,1,2)
        plt.plot(range(1, 101), torch.tensor(corr[-100:]).mean(dim=1))
        plt.boxplot(corr[-100:], showfliers=True)        
        plt.legend(['Mean is blue'])
        # plt.ylim([0.5, 0.8])
        plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
        plt.xlabel('Sample index')
        plt.title(f'Correlation result for NEM-{i}')

        plt.subplots_adjust(hspace=0.7)
        plt.savefig(f'NEM-{i}.png')
        plt.show()

    # m, c = {'mean':[], 'std':[]}, {'mean':[], 'std':[]}
    # for i in range(1,6):
    #     mse, corr = torch.load(f'nem_v{i}.pt')
    #     m['mean'].append(torch.tensor(mse).mean())
    #     m['std'].append(torch.tensor(mse).var()**0.5)
    #     c['mean'].append(torch.tensor(corr).mean())
    #     c['std'].append(torch.tensor(corr).var()**0.5)
    # plt.figure()
    # plt.plot(range(5),np.log(m['mean']), '-x')
    # plt.xticks(ticks=range(5), \
    #     labels=('model-1','model-2','model-3','model-4','model-5'))
    # plt.xlabel('Lambda')
    # plt.title('Mean of MSE')
    # plt.savefig('Mean of MSE.png')

    # plt.figure()
    # plt.xticks(ticks=range(5), \
    # labels=('model-1','model-2','model-3','model-4','model-5'))
    # plt.xlabel('Lambda')
    # plt.plot(np.log(m['std']), '-x')
    # plt.title('STD of MSE')
    # plt.savefig('STD of MSE.png')

    # plt.figure()
    # plt.errorbar(range(5),np.log(m['mean']), abs(np.log(m['std'])), capsize=4)
    # plt.xticks(ticks=range(5), \
    #     labels=('model-1','model-2','model-3','model-4','model-5'))
    # plt.xlabel('Lambda')
    # plt.title('Mean of MSE with std')
    # plt.savefig('Mean of MSE with std.png')


    # plt.figure()
    # plt.plot(c['mean'],'-x')
    # plt.xticks(ticks=range(5), \
    # labels=('model-1','model-2','model-3','model-4','model-5'))
    # plt.xlabel('Lambda')
    # plt.title('Mean of Corr.')
    # plt.savefig('Mean of Corr.png')

    # plt.figure()
    # plt.plot(c['std'], '-x')
    # plt.xticks(ticks=range(5), \
    # labels=('model-1','model-2','model-3','model-4','model-5'))
    # plt.xlabel('Lambda')
    # plt.title('STD of Corr.')
    # plt.savefig('STD of Corr.png')

    # plt.figure()
    # plt.errorbar(range(len(all_lamb)),c['mean'], c['std'], capsize=4)
    # plt.xticks(ticks=range(5), \
    #     labels=('model-1','model-2','model-3','model-4','model-5'))
    # plt.xlabel('Lambda')
    # plt.title('Mean of Corr with std')
    # plt.savefig('Mean of Corr with std.png')

#%% plot EM dynamic toy results 
    res = torch.load('../data/nem_ss/nem_res/res_toy100shift.pt')
    corr = torch.tensor(res)
    plt.figure()
    plt.plot(range(1, 101), torch.tensor(corr[-100:]).mean(dim=1))
    plt.boxplot(corr[-100:], showfliers=True)        
    plt.legend(['Mean is blue'])
    # plt.ylim([0.5, 0.8])
    plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
    plt.xlabel('Sample index')
    plt.title('Correlation result for EM')
    plt.show()

#%% MUSIC algorithm for DOA -- it need J<M
    for i in range(20):
        x = torch.from_numpy(x_all[i]).permute(1,2,0)
        Rx = (x[..., None] @ x[:,:,None,:].conj()).sum([0,1])/2500
        l, v = torch.linalg.eigh(Rx)
        un = v[:,:2]
        res = []
        for i in torch.arange(0, np.pi, np.pi/100):
            omeg = i
            e = torch.tensor([1, np.exp(1*1j*omeg), np.exp(2*1j*omeg), np.exp(3*1j*omeg), np.exp(4*1j*omeg)])
            P = 1/(e[None, :]@un@un.conj().t()@e[:,None])
            res.append(abs(P))
        plt.figure()
        plt.plot(res)

#%% show 20db, 10db, 5db, 0db result
    "This code shows EM is boosted by a little bit noise"
    # res, _ = torch.load('../data/nem_ss/nem_res/res_nem_shat_hhat_snr5.pt') # s,h
    # _, res = torch.load('../data/nem_ss/nem_res/res_nem_shat_hhat_snr5.pt') # _,h
    # res, _ = torch.load('../data/nem_ss/nem_res/res_shat_hhat_snrinf.pt') # s,_ EM
    # _, res = torch.load('../data/nem_ss/nem_res/res_shat_hhat_snr20.pt') # _,h

    # plt.figure()
    # plt.plot(range(1, 101), torch.tensor(res).mean(dim=1))
    # plt.boxplot(res, showfliers=True)        
    # plt.legend(['Mean is blue'])
    # plt.ylim([0.5, 1])
    # plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
    # plt.xlabel('Sample index')
    # plt.title('NEM correlation result for h')
    # plt.show()

    ss = []
    for i in [0, 5, 10, 20, 'inf']:
        res, _ = torch.load(f'../data/nem_ss/nem_res/res_em_sh_init10db_snr{i}.pt') # s, h NEM
        s = 0
        for i in range(100):
            for ii in range(20):
                s = s + res[i][ii]
        print(s/2000)
        ss.append(s/2000)
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')

    ss = []
    for i in [0, 5, 10, 20, 'inf']:
        res, _ = torch.load(f'../data/nem_ss/nem_res/EM_NEM_snr/res_nem_shat_hhatsnr_{i}.pt') # s, h NEM
        s = 0
        for i in range(100):
            for ii in range(20):
                s = s + res[i][ii]
        print(s/2000)
        ss.append(s/2000)
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-o')


    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['EM', 'NEM'])
    plt.title('Correlation result for s')

#%% 10db result as the initial to do the EM
    import itertools, time
    t = time.time()
    d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    h = torch.tensor(h)
    J = h.shape[-1]
    ratio = d.abs().amax(dim=(1,2,3))/3
    x = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1)

    def corr(vh, v):
        "vh and v are the real value with shape of [N,F,J]"
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = vh[..., torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

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

    def em_func_mod(x, J=3, Hscale=1, Rbscale=100, max_iter=501, v_init=False, h_init=False, \
        lamb=0, seed=0, show_plot=False):
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
        if v_init is False: 
            vhat = torch.randn(N, F, J).abs().to(torch.cdouble) 
        else :
            vhat = v_init
        if h_init is False: 
            Hhat = torch.randn(M, J, dtype=torch.cdouble)*Hscale
        else:
            Hhat = h_init
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

    # samples = 100
    # seeds = 20
    # hh_all = torch.rand(samples, seeds, J, J, dtype=torch.cdouble)
    # vh_all = torch.rand(samples, seeds, 100, 100, J, dtype=torch.cdouble)
    # for i in range(samples):
    #     for ii in range(seeds):
    #         shat, hh_all[i, ii], vh_all[i, ii], Rb = em_func(awgn(x[i], snr=10), seed=ii)
    #     print(f'finished {i} sample')
    # print(f'used {time.time()-t}s')
    # torch.save([hh_all, vh_all], 'hh_all-vh_all_resforinit.py')
    hh_all, vh_all = torch.load('hh_all-vh_all_resforinit.py')

    res, res2 = [], []
    for i in range(100):
        c, cc = [], []
        for ii in range(10):
            shat, Hhat, vhat, Rb = em_func_mod(awgn(x[i], snr=20), J=6,\
                v_init=vh_all[i,ii], h_init=hh_all[i,ii], seed=ii)
            c.append(corr(shat.squeeze().abs(), s_all[i]))
            cc.append(h_corr(h, Hhat))
        res.append(c)
        res2.append(cc)
        print(f'finished {i} samples')
    # torch.save([res, res2], 'res_em_shat_hhat_snr20.pt')

    s = 0 
    for i in range(100):
        for ii in range(20):
            s = s + res[i][ii]
    print(s/2000)

#%% check loss function values
    l = torch.load('../data/nem_ss/models/rid141103/loss_rid141103.pt')
    l = torch.tensor(l)
    n = 3
    c = []
    for epoch in range(len(l)):
        if epoch > 10 :
            ll = l[:epoch]
            s1, s2 = sum(ll[epoch-2*n:epoch-n])/n, sum(ll[epoch-n:])/n
            c.append( abs((s1-s2)/s1))
            print(f'current epcoch-{epoch}: ', abs((s1-s2)/s1), s1, s2)
            if s1 - s2 < 0 :
                print('break-1')
                break
            if abs((s1-s2)/s1) < 5e-4 :
                print(epoch)
                print('break-2')
                break
    plt.plot(c, '-x')

#%% validation value check
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 100
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    import itertools
    from unet.unet_model import UNetHalf8to100_256_sig as UNetHalf
    from datetime import datetime
    print('starting date time ', datetime.now())
    torch.manual_seed(1)

    I, J, bs = 130, 6, 32 # I should be larger than bs
    d, s, h = torch.load('../data/nem_ss/val500M6FT100_xsh.pt')
    s_all, h = s.abs().permute(0,2,3,1), torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    xte = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]*3).permute(0,2,3,1)# [sample, N, F, channel]
    xte = awgn_batch(xte[:I], snr=1000)
    data = Data.TensorDataset(xte)
    data_test = Data.DataLoader(data, batch_size=bs, drop_last=True)

    from skimage.transform import resize
    gte = torch.tensor(resize(xte[...,0].abs(), [I,8,8], order=1, preserve_range=True ))
    gte = gte[:I]/gte[:I].amax(dim=[1,2])[...,None,None]  #standardization 
    gte = torch.cat([gte[:,None] for j in range(J)], dim=1)[:,:,None] # shape of [I_val,J,1,8,8]
    l = torch.load('../data/nem_ss/lb_c6_J188.pt')
    lb = l.repeat(bs, 1, 1, 1, 1).cuda()

    rid = 160001
    ll_all = []
    for i in range(42):
        model = torch.load(f'../data/nem_ss/models/rid{rid}/model_rid{rid}_{i}.pt')
        ll = val_run(data_test, gte, model, lb, [6,6,32], seed=1)
        ll_all.append(ll)
        print(ll)
        plt.figure()
        plt.title(f'validation likelihood till epoch {i}')
        plt.plot(ll_all, '-or')
        plt.savefig(f'id{rid} validation likelihood')
        plt.close()
    print('End date time ', datetime.now())

#%% Use H to do classification, if ground truth h is available
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    import itertools
    import time
    t = time.time()
    d, s, h = torch.load('../data/nem_ss/test500M6FT100_xsh.pt')
    h, N, F, M = torch.tensor(h), s.shape[-1], s.shape[-2], 6

    def hcorr(hi, hh):
        r = []
        for i in range(6):
            n = hi @ hh[:,i].conj()
            d = hi.norm() * hh[:,i].norm()
            r.append(n.abs()/d)
        return max(r)

    n_comb = 2
    comb = list(itertools.combinations(range(6), n_comb))
    lb = torch.tensor(comb)
    lbs = lb.unsqueeze(1).repeat(1,100,1).reshape(lb.shape[0]*100, n_comb)
    hall = torch.load(f'../data/nem_ss/nem_res/Hhat_{n_comb}comb_res.pt')
    acc = 0
    for i in range(len(hall)):
        hh = hall[i].squeeze()
        res = []
        for wc in range(6):
            res.append(hcorr(h[:,wc], hh))
        _, lb_hat = torch.topk(torch.tensor(res), n_comb)

        for ii in range(n_comb):
            if lb_hat[ii] in lbs[i]:
                acc += 1
    print(acc/len(hall)/n_comb)
    "if ground truth h is konwn, 2~0.97, 3~0.92983, 4~0.937833, 5~0.95766, 6~1.00 and 1~1.00 of coursce"

#%% Use H to do classification, using training h 
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    import itertools
    import time
    t = time.time()
    s_all, h_all, v_all= torch.load('../data/nem_ss/shv_tr3k_M6_rid160100.pt')
    s0, h0 = s_all[0].squeeze(), h_all[0]
    "In s[0] from 0~31: 3,4,8,13,14,15,29,31 are class4 then class5, others are 5,4"
    "also, 5,6,7,8, 11, 12,23, 26, 30 really bad"
    # idx1 = torch.tensor([0,5,2,1,4,3]) # for others -- 0,3,2,5,4,1
    # idx2 = torch.tensor([0,5,2,1,3,4]) # for 3,4,8,13,14,15,29,31 -- 0,3,2,4,5,1

    def get_lb(h0, hh):
        res = torch.zeros(30,6,6)
        for i in range(3):
            if i in [3,4,8,13,14,15,29,31]:
                idx = torch.tensor([0,5,2,1,3,4])
            else:
                idx = torch.tensor([0,5,2,1,4,3])
            h = h0[0][:, idx]
            for d1 in range(6):
                hi = h[:, d1]
                for d2 in range(6):
                    n = hi@ hh[:,d2].conj()
                    d = hi.norm() * hh[:,d2].norm()
                    res[i, d1, d2] = n.abs()/d
        return res.sum(0).amax(dim=1)

    n_comb = 2
    comb = list(itertools.combinations(range(6), n_comb))
    lb = torch.tensor(comb)
    lbs = lb.unsqueeze(1).repeat(1,100,1).reshape(lb.shape[0]*100, n_comb)
    hall = torch.load(f'../data/nem_ss/nem_res/Hhat_{n_comb}comb_res.pt')
    acc = 0
    for i in range(len(hall)):
        hh = hall[i].squeeze()
        res = get_lb(h0, hh)
        _, lb_hat = torch.topk(res, n_comb)

        for ii in range(n_comb):
            if lb_hat[ii] in lbs[i]:
                acc += 1
    print(acc/len(hall)/n_comb)

#%% HCI10 for NEM 18k training M=3
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

    d= torch.load('../data/nem_ss/tr18kM3FT64_data0.pt')
    ratio = d.abs().amax(dim=(1,2,3))
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)

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

    H = []
    for i in range(18_000):
        H.append(cluster_init(x_all[i]))
        if i % 10 == 0:
            print(f'done with {i}')

    Hhat = torch.stack(H, dim=0)
    torch.save(Hhat, 'tr18kHCI10.pt')
    print('done')

#%% HCI10 for NEM 18k training, M=6
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

    d= torch.load('../data/nem_ss/tr18kM6FT64_data3.pt')
    ratio = d.abs().amax(dim=(1,2,3))
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)

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

    H = []
    for i in range(18_000):
        H.append(cluster_init(x_all[i], J=6, K=7))
        if i % 10 == 0:
            print(f'done with {i}')

    Hhat = torch.stack(H, dim=0)
    torch.save(Hhat, 'tr18kHCI10M6_data3.pt')
    print('done')