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

    #%% Prepare full rank data5 J=6 classes 18ktr, with rangdom AOA, 1000 val, 1000 te
    from utils import *
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)

    "raw data processing"
    FT = 64  #48, 64, 80, 100, 128, 200, 256
    var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
    data = {}
    def get_ftdata(data_pool):
        *_, Z = stft(data_pool, fs=4e7, nperseg=FT, boundary='zeros')
        x = torch.tensor(np.roll(Z, FT//2, axis=1))  # roll nperseg//2
        return x.to(torch.cfloat)

    for i in range(6):
        temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        x = torch.tensor(temp['x'])
        x =  x/((x.abs()**2).sum(dim=(1),keepdim=True)**0.5)# normalize
        data[i] = x
    s = []
    for i in range(6):
        s.append(get_ftdata(data[i])) # ble [2000,F,T]

    torch.manual_seed(0)
    M, J, I = 6, 6, 20000
    ln = 0
    res = []
    combs = torch.combinations(torch.tensor([i for i in range(J)]))
    while ln < I:
        aoa = torch.rand(J ,int(2e4))*160 +10# get more then remove diff_angle<10
        for i, c in enumerate(combs):
            if i == 0:
                id = (aoa[c[0]]- aoa[c[1]]).abs() > 10 # ang diff >10
            else:
                id = torch.logical_and(id, (aoa[c[0]]- aoa[c[1]]).abs() > 10)
        ln += id.sum() # id contains True and False
        res.append(aoa[:,id])
        print('one loop')
    aoa = torch.cat(res, dim=1)
    aoa = aoa[:,:I].to(torch.cfloat)/180*np.pi  # [J, I] to radius angle 
    def get_channel(aoa):
        los = torch.arange(M)[:,None].to(torch.cfloat)*np.pi #[M,1]
        h_los = (los@aoa.t().sin()[:,None]*1j).exp() #[I, M, J]
        p_aoa = torch.tensor([30, -30, 45, -45, 60])
        att = torch.tensor([0.5**0.5, 0.5, 0.5**1.5, 0.5**2, 0.5**2.5])
        ch = [h_los]
        for ii in range(M-1): # multi-path
            ang = p_aoa[ii] + aoa
            ch.append(att[ii]*(los@ang.t().sin()[:,None]*1j).exp())
        ch = torch.stack(ch)
        mix = ch.sum(0)
        return ch, mix
    ch, mix_ch = get_channel(aoa=aoa)

    #%%
    hall = mix_ch.reshape(10,2000,M,J) # this is easier for later processing
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
    torch.save(x[:18000], f'tr18kM6FT{FT}_data5.pt')

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
    torch.save((valtest[:1000], svaltest[:1000], hall[9,:1000]), f'val1kM6FT{FT}_xsh_data5.pt')
    torch.save((valtest[1000:], svaltest[1000:], hall[9,1000:]), f'test1kM6FT{FT}_xsh_data5.pt')
    print('done')
