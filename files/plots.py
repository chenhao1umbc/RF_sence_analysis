#%%
if True:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 500
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)

#%% plot rid140100 details
    