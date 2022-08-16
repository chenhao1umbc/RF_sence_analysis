#%%
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
for name in ['v78', 'v78c', 'v79', 'v79c', 'v80', 'v81', 'v82', 'v82a', 'v83']:
    hall, sall = [], []
    file1 = open(name+'.out', 'r')
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
    plt.grid()
    plt.legend(['h corr.', 's corr.'])
    plt.title(name)