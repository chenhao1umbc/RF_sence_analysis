#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)

#%%
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


location = '../data/nem_ss/nem_res/'

ss = []
for i in [0, 5, 10, 20, 'inf']:
    res, _ = torch.load(location + f'res_nem_shat_hhat_rid135110_snr_{i}db.pt') # s, h NEM
    s = 0
    for i in range(100):
        for ii in range(10):
            s = s + res[i][ii]
    print(s/1000)
    ss.append(s/1000)
plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')

ss = []
for i in [0, 5, 10, 20, 'inf']:
    _, res = torch.load(location + f'res_nem_shat_hhat_rid135110_snr_{i}db.pt') # s, h NEM
    s = 0
    for i in range(100):
        for ii in range(10):
            s = s + res[i][ii]
    print(s/1000)
    ss.append(s/1000)
plt.plot([0, 5, 10, 20, 'inf'], ss, '-o')


plt.plot([0, 5, 10, 20, 'inf'], [0.794507, 0.904, 0.950276, 0.950312, 0.951212], '-x')

plt.ylabel('Averaged correlation result')
plt.xlabel('SNR')
plt.legend(['NEM Correlation for s', 'NEM Correlation for h', 'EM Correlation for s'])
plt.title('1-Channel 1 neural network')
#%%

plt.figure()
ss = []
for i in [0, 5, 10, 20, 'inf']:
    res, _ = torch.load(location + f'res_nem_shat_hhat_rid135110_snr_{i}db.pt') # s, h NEM
    s = 0
    for i in range(100):
        for ii in range(10):
            s = s + res[i][ii]
    print(s/1000)
    ss.append(s/1000)
plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')


plt.plot([0, 5, 10, 20, 'inf'], [0.794507, 0.904, 0.950276, 0.950312, 0.951212], '-x')

plt.ylabel('Averaged correlation result')
plt.xlabel('SNR')
plt.legend(['NEM Correlation for s', 'EM Correlation for s'])
plt.title('1-Channel one model')
#%%
