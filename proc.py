#%%
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
for name in ['s40', 's41', 's42', 's43', 's44', 's45']:
# for name in ['s35', 's35a', 's35b', 's36', 's37', 's38']:
    hall, sall = [], []
    file1 = open('../data/0ut_files/'+name+'.out', 'r')
    lines = file1.readlines()
    for line in lines:
        if line[:len('first 3 h_corr')] == 'first 3 h_corr':
            hval = line.split()[-1]
            hall.append(float(hval[:6]))
        if line[:len('first 3 s_corr')] == 'first 3 s_corr':
            sval = line.split()[-1]
            sall.append(float(sval[:6]))
    plt.figure()
    plt.plot(hall, '--x')
    plt.plot(sall, '-o')
    plt.ylim([0.5, 0.9])
    plt.grid()
    plt.legend(['h corr.', 's corr.'])
    plt.title(name)