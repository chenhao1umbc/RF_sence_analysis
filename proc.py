#%%
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
for name in [f'fv3a{i}' for i in range(0,12)]:
    hall, sall = [], []
    file1 = open('../data/0ut_files/'+name+'.out', 'r')
    lines = file1.readlines()
    max_s, max_h = 0, 0
    for line in lines:
        if line[:len('first 3 h_corr')] == 'first 3 h_corr':
            hval = line.split()[-1]
            if hval == 'nan':
                continue
            hall.append(float(hval[:6]))
            max_h = max(float(hval[:6]), max_h)
        if line[:len('first 3 s_corr')] == 'first 3 s_corr':
            sval = line.split()[-1]
            if sval == 'nan':
                continue
            sall.append(float(sval[:6]))
            max_s = max(float(sval[:6]), max_s)
    print(name+' max_h, max_s', max_h, max_s)
    plt.figure()
    plt.plot(hall, '--x')
    plt.plot(sall, '-o')
    plt.ylim([0.55, 1])
    plt.grid()
    plt.legend(['h corr.', 's corr.'])
    plt.title(name)