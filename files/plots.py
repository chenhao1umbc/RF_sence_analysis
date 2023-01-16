#%%
if True:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 500
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)

#%% plot full-rank vae added to ICC results
# fr_vae_s = [0.8029057860374451, 0.7901700229644776, 0.7399093695878982, 0.6615210013091565, 0.5074138607978821]
# fr_vae_h = [0.9037600067853928, 0.8975545323491096, 0.8950184407830238, 0.8810217734575272, 0.8265930427312851]
fr_vae_h = [0.8984494034647942, 0.8976094114780426, 0.8950949864983558, 0.8712245292067528, 0.8400674899816513] 
fr_vae_s = [0.7951915324926376, 0.7789489619135856, 0.7537819658517837, 0.7017759349346161, 0.5418391361534596]

em_s = [0.7445756667554378, 0.7094151854638325, 0.5843308384998702, 0.5246096028979378, 0.4447607653506068]
nem_s = [0.7869621894432611, 0.7704570677869884, 0.720015995038842, 0.6592119414151044, 0.4793672238778914]
vae_s = [0.8535560958385467, 0.8231705112457275, 0.7537818377017975, 0.6681900478005409, 0.5212438029050827]

em_h = [0.8522483021616936, 0.8460731382025013, 0.7403606974878257, 0.6926773731185453, 0.6571403841287834]
nem_h = [0.878959828681557, 0.8747711389403388, 0.8708698675967644, 0.8493816832457324, 0.7786746796365122]    
vae_h = [0.9512082036733628, 0.9428754450082779, 0.9035894940495491, 0.8645251979231834, 0.7837165744900704]

plt.figure()
plt.subplot(1,2,1)
plt.plot([0, 5, 10, 20, 'inf'], em_h[::-1], '-x', color="#1f77b4") 
plt.plot([0, 5, 10, 20, 'inf'], nem_h[::-1], '--o', color="#2ca02c")
plt.plot([0, 5, 10, 20, 'inf'], vae_h[::-1], '-.v', color="#ff7f0e")  
plt.plot([0, 5, 10, 20, 'inf'], fr_vae_h[::-1], '-.^',color="r") 
plt.legend(['EM', 'NEM', 'VAE'])
plt.ylabel('Channel correlation')
plt.xlabel('SNR (dB)')
plt.legend(['EM', 'NEM', 'VAE','VAE-new'])

plt.subplot(1,2,2)
plt.plot([0, 5, 10, 20, 'inf'], em_s[::-1], '-x', color="#1f77b4") 
plt.plot([0, 5, 10, 20, 'inf'], nem_s[::-1], '--o', color="#2ca02c")
plt.plot([0, 5, 10, 20, 'inf'], vae_s[::-1], '-.v', color="#ff7f0e")
plt.plot([0, 5, 10, 20, 'inf'], fr_vae_s[::-1], '-.^',color="r") 
plt.legend(['EM', 'NEM', 'VAE','VAE-new'])
plt.ylabel('STFT correlation')
plt.xlabel('SNR (dB)')
plt.tight_layout(pad=1)