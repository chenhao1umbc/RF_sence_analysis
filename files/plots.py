#%%
if True:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)

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
    
#%% NEM, EM, VAE 3-classes
    em_h = [0.9551496576997164, 0.9360595836205491, 0.7878769764356256, 0.7296894052324431, 0.7227296148806387]
    em_s = [0.86208436247627, 0.8225587648460572, 0.6706913897637541, 0.5984397914593004, 0.5115356154467433]

    nem_h = [0.9691816622106057, 0.9688851905589388, 0.9734216309412388, 0.9532163183589794, 0.8837493564754592]
    nem_s = [0.8948252103660488, 0.8553565386062452, 0.802421071484698, 0.7227045673753786, 0.5473333896372122] 

    vae_h = [0.9984038119316101, 0.9983815302848816, 0.9960109205245972, 0.980860576748848, 0.9167653774619102] 
    vae_s = [0.9600338533520698, 0.9418429788947106, 0.8837822783589363, 0.793323303937912, 0.640838300049305]

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot([0, 5, 10, 20, 'inf'], em_h[::-1], '-x', color="#1f77b4") 
    plt.plot([0, 5, 10, 20, 'inf'], nem_h[::-1], '--o',color="#2ca02c")
    plt.plot([0, 5, 10, 20, 'inf'], vae_h[::-1], '-.v', color="#ff7f0e") 
    plt.ylim([0.7, 1])
    plt.ylabel('Channel correlation')
    plt.xlabel('SNR (dB)')
    plt.legend(['EM', 'NEM', 'VAE'])

    plt.subplot(1,2,2)
    plt.plot([0, 5, 10, 20, 'inf'], em_s[::-1], '-x', color="#1f77b4") 
    plt.plot([0, 5, 10, 20, 'inf'], nem_s[::-1], '--o',color="#2ca02c")
    plt.plot([0, 5, 10, 20, 'inf'], vae_s[::-1], '-.v', color="#ff7f0e") 
    plt.ylim([0.5, 1])
    plt.legend(['EM', 'NEM', 'VAE'])
    plt.ylabel('STFT correlation')
    plt.xlabel('SNR (dB)')
    plt.tight_layout(pad=1)

#%% EM, VAE 6-classes
    em_s = [0.7445756667554378, 0.7094151854638325, 0.5843308384998702, 0.5246096028979378, 0.4447607653506068]
    em_h = [0.8522483021616936, 0.8460731382025013, 0.7403606974878257, 0.6926773731185453, 0.6571403841287834]

    vae_s = [0.8535560958385467, 0.8231705112457275, 0.7537818377017975, 0.6681900478005409, 0.5212438029050827]
    vae_h = [0.9512082036733628, 0.9428754450082779, 0.9035894940495491, 0.8645251979231834, 0.7836784112453461]

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot([0, 5, 10, 20, 'inf'], em_h[::-1], '-x', color="#1f77b4") 
    plt.plot([0, 5, 10, 20, 'inf'], vae_h[::-1], '-.v', color="#ff7f0e")  
    plt.legend(['EM', 'VAE'])
    plt.ylabel('Channel correlation')
    plt.xlabel('SNR (dB)')
    plt.legend(['EM', 'NEM', 'VAE'])

    plt.subplot(1,2,2)
    plt.plot([0, 5, 10, 20, 'inf'], em_s[::-1], '-x', color="#1f77b4") 
    plt.plot([0, 5, 10, 20, 'inf'], vae_s[::-1], '-.v', color="#ff7f0e")
    plt.legend(['EM', 'VAE'])
    plt.ylabel('STFT correlation')
    plt.xlabel('SNR (dB)')
    plt.tight_layout(pad=1)

    "VAE, NEM 6-class small dataset, using s1201k, s1202k..."
    vae = [0.6853250654935836,  0.7350429352819919, 0.7588296358287334, 0.7619795973300933, 0.7876128880381584]
    nem = [0.7686689290784765,0.770921271815709, 0.7710523081942222, 0.7756224503521928, 0.7829264613947694]
    plt.figure()
    plt.plot(torch.tensor([1e3, 2e3, 3e3, 4e3, 6e3]).log(), nem, '--o', color="#2ca02c") 
    plt.plot(torch.tensor([1e3, 2e3, 3e3, 4e3, 6e3]).log(), vae, '-.v', color="#ff7f0e") 
    plt.xticks(torch.tensor([1e3, 2e3, 3e3, 4e3, 6e3]).log(),['1k', '2k', '3k', '4k', '6k'])
    plt.legend(['NEM', 'VAE'])
    plt.ylabel('STFT correlation')
    plt.xlabel('Training set size')
