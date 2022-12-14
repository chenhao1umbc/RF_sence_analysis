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
