#%%
if True:
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    
    import matplotlib
    matplotlib.rc('font', size=22)

#%% plot rid140100 details
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

#%% rid 140100 results
    #%% plot s -- 10 seed
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

    #%% plot h -- 10 seed
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

#%%  3 model vs 1 model 
    plt.figure()
    em1 = [0.78254, 0.90830, 0.96578, 0.99330, 0.99593]
    nem1 = [0.85142, 0.95064, 0.99069, 0.99865, 0.99991]
    em2 = [0.7955, 0.9053, 0.9521, 0.9544, 0.9513]
    # nem2 = [0.877197, 0.958912, 0.975978, 0.970579, 0.978831]  # 140100_52
    nem2 = [0.904499, 0.972585, 0.990800, 0.998605, 0.999293]  # 140100_48
    plt.plot([0, 5, 10, 20, 'inf'], em1, '--o', color='royalblue')
    plt.plot([0, 5, 10, 20, 'inf'], nem1, '--o', color='orange')
    plt.plot([0, 5, 10, 20, 'inf'], em2, '-x', color='royalblue')
    plt.plot([0, 5, 10, 20, 'inf'], nem2, '-x', color='orange')
    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['Old EM', 'Old NEM', 'New EM', 'New NEM'])
    plt.title('Merged plots')

#%% These results are not all correct, the snr0 should be worse than these
    #%%  rid150000_35
    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'], [0.7955, 0.9053, 0.9521, 0.9544, 0.9513], '-x')
    ss = [0.9151661591587187, 0.9641487953572089, 0.9856960356248871, 0.9977277628251509, 0.9997574770952526]
    # ss = [0.8769 ,...] #snr0 should be
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')

    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['EM Correlation for s', 'NEM Correlation for s'])
    plt.title('Correlation for s')

    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'], [0.967135, 0.971392, 0.978568, 0.976789, 0.980658], '-x')
    ss = [0.9996597122402164, 0.9996303066482091, 0.9994293296731978, 0.9995672830187705, 0.9998204659383773]
    # ss = [0.9980, ...] # snr0 should be
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')
    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['EM Correlation for h', 'NEM Correlation for h'])
    plt.title('Correlation for h')

    #%%  rid141104_58  low snr is bad
    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'], [0.7955, 0.9053, 0.9521, 0.9544, 0.9513], '-x')
    ss = [0.631, 0.916, 0.9903, 0.995, 0.997]
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')

    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['EM Correlation for s', 'NEM Correlation for s'])
    plt.title('Correlation for s')

    #%%  rid145002_57 -- not as good as rid150000_35
    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'], [0.7955, 0.9053, 0.9521, 0.9544, 0.9513], '-x')
    ss = [0.8783880123971055, 0.9338809976629762, 0.9663077043350395, 0.9852282607157516, 0.9871739072054928]
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')

    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['EM Correlation for s', 'NEM Correlation for s'])
    plt.title('Correlation for s')

    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'], [0.967135, 0.971392, 0.978568, 0.976789, 0.980658], '-x')
    ss = [0.9933938175362463, 0.9980773181036969, 0.9998076659451002, 0.9999453892450755, 0.9998789322102434]
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')
    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['EM Correlation for h', 'NEM Correlation for h'])
    plt.title('Correlation for h')

#%% rid160000_41 6 classes vs EM raw random
    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'], [0.6008424959295375, 0.7248948191915779, 0.7780298439919326, 0.7898476635307224, 0.7458360596326318], '-x')
    ss = [0.618957818305481, 0.7868766088411929, 0.8562572809329362, 0.9007027679035039, 0.9072986763470671]
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')

    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['EM Correlation for s', 'NEM Correlation for s'])
    plt.title('Correlation for s')

    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'], [0.8520199621647806, 0.8916706811833388,0.9001839690460522,0.8782347365683133,0.865501992043354], '-x')
    ss = [0.8691834507788601, 0.8673765321742478, 0.8590790125706347, 0.8711583053818234, 0.8698747989019353]
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')
    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['EM Correlation for h', 'NEM Correlation for h'])
    plt.title('Correlation for h')

#%% rid160100_33 6 classes vs EM boost random
    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'], [0.6351, 0.7451, 0.7928, 0.8059, 0.8109 ], '-x')
    ss = [0.6708, 0.8032, 0.8690, 0.9151, 0.9218]
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')
    # plt.ylim([0.5, 1])
    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['EM Correlation for s', 'NEM Correlation for s'])
    plt.title('Correlation for s')

    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'], [0.8520, 0.8616, 0.8671, 0.8682, 0.8755], '-x')
    ss = [0.8721, 0.8910, 0.9007, 0.9092, 0.9099]
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')
    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    # plt.ylim([0.5, 1])
    plt.legend(['EM Correlation for h', 'NEM Correlation for h'])
    plt.title('Correlation for h')

    matplotlib.rc('font', size=16)
    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'],  [0.6351, 0.7451, 0.7928, 0.8059, 0.8109 ], '--x')
    ss = [0.6708, 0.8032, 0.8690, 0.9151, 0.9218]
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-o')
    plt.plot([0, 5, 10, 20, 'inf'],[0.8520, 0.8616, 0.8671, 0.8682, 0.8755], '--x')
    hh = [0.8721, 0.8910, 0.9007, 0.9092, 0.9099]
    plt.plot([0, 5, 10, 20, 'inf'], hh, '-o')
    plt.ylabel('Corr.', fontsize=16)
    plt.xlabel('SNR', fontsize=16)
    plt.legend(['EM for s', 'NEM for s', 'EM for h', 'NEM for h'], prop={'size': 16})
    plt.savefig('6hs.eps', bbox_inches = 'tight')


#%% 182340
    matplotlib.rc('font', size=16)
    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'], [0.7099, 0.8650, 0.9360, 0.9580, 0.9499 ], '-x')
    ss = [0.7999, 0.9250, 0.9630, 0.9980, 0.9999]
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-o')
    # plt.ylim([0.5, 1])
    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['EM ', 'NEM'])
    # plt.title('Correlation for s')
    plt.savefig('3s.eps')

    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'], [0.8550, 0.9216, 0.9571, 0.9682, 0.9675], '-x')
    ss = [0.941, 0.970, 0.9717, 0.991, 0.9999]
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-o')
    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    # plt.ylim([0.5, 1])
    plt.legend(['EM ', 'NEM '])
    # plt.title('Correlation for h')
    plt.savefig('3h.eps')

    matplotlib.rc('font', size=16)
    plt.figure()
    plt.plot([0, 5, 10, 20, 'inf'], [0.7099, 0.8650, 0.9360, 0.9580, 0.9499 ], '--x')
    ss = [0.7999, 0.9250, 0.9630, 0.9980, 0.9999]
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-o')
    plt.plot([0, 5, 10, 20, 'inf'], [0.8550, 0.9216, 0.9571, 0.9682, 0.9675], '--x')
    hh = [0.941, 0.970, 0.9717, 0.991, 0.9999]
    plt.plot([0, 5, 10, 20, 'inf'], hh, '-o')
    plt.ylabel('Corr.', fontsize=16)
    plt.xlabel('SNR', fontsize=16)
    plt.legend(['EM for s', 'NEM for s', 'EM for h', 'NEM for h'], prop={'size': 16})
    plt.savefig('3hs.eps', bbox_inches = 'tight')

#%% plot mixture
    d, s, h = torch.load('/home/chenhao1/Hpython/data/nem_ss/test500M3FT100_xsh.pt')
    h, N, F = torch.tensor(h), s.shape[-1], s.shape[-2] # h is M*J matrix, here 6*6
    ratio = d.abs().amax(dim=(1,2,3))
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    
    plt.imshow(x_all[0,...,0].abs())
    plt.colorbar()