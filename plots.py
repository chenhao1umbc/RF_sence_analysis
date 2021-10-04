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
res, _ = torch.load('../data/nem_ss/nem_res/res_nem_10seed_rid140100_snrinf.pt') # s,_ EM
# _, res = torch.load('../data/nem_ss/nem_res/res_nem_10seed_rid140200_snrinf.pt') # _,h

plt.figure()
plt.plot(range(1, 101), torch.tensor(res).mean(dim=1))
plt.boxplot(res, showfliers=True)        
plt.legend(['Mean is blue'])
plt.ylim([0.5, 1])
plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
plt.xlabel('Sample index')
plt.title('NEM correlation result for s')
plt.show()

#%% plot s
location = '../data/nem_ss/nem_res/'
plt.figure()
#res, _ = torch.load(location + f'res_sh_10seed_snr{i}.pt') # s, h NEM
plt.plot([0, 5, 10, 20, 'inf'], [0.7955, 0.9053, 0.9521, 0.9544, 0.9513], '-x')
ss = []
for i in [0, 5, 10, 20, 'inf']:
    res, _ = torch.load(location + f'res_nem_10seed_rid140100_snr{i}.pt') # s, h NEM
    s = 0
    for i in range(100):
        for ii in range(10):
            s = s + res[i][ii]
    print(s/1000)
    ss.append(s/1000)
plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')

plt.ylabel('Averaged correlation result')
plt.xlabel('SNR')
plt.legend(['EM Correlation for s', 'NEM Correlation for s'])
plt.title('Correlation for s')

#%% plot h
location = '../data/nem_ss/nem_res/'
plt.figure()
ss = []
#_, res = torch.load(location + f'res_sh_10seed_snr{i}.pt') # s, h NEM
plt.plot([0, 5, 10, 20, 'inf'], [0.967135, 0.971392, 0.978568, 0.976789, 0.980658], '-x')

ss = []
for i in [0, 5, 10, 20, 'inf']:
    _, res = torch.load(location + f'res_nem_10seed_rid140100_snr{i}.pt') # s, h NEM
    s = 0
    for i in range(100):
        for ii in range(10):
            s = s + res[i][ii]
    print(s/1000)
    ss.append(s/1000)
plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')

plt.ylabel('Averaged correlation result')
plt.xlabel('SNR')
plt.legend(['EM Correlation for h', 'NEM Correlation for h'])
plt.title('Correlation for h')

#%%
