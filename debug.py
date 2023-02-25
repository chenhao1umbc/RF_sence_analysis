#%% 3-class EM
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())

"make the result reproducible"
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)       # current GPU seed
torch.cuda.manual_seed_all(seed)   # all GPUs seed
torch.backends.cudnn.deterministic = True  #True uses deterministic alg. for cuda
torch.backends.cudnn.benchmark = False  #False cuda use the fixed alg. for conv, may slower

#%% 3gpp channel model
"""check the file generate_bulk_par.m
only 1 link which means k=1
only care about AOA, which means AOD is fine, just let TX antenna number = 1
only care about 1 path and 20 subpaths
The subpaths are equal powered
"""

aod_2deg = torch.tensor([0.0894, 0.2826, 0.4984, 0.7431, 1.0257, 1.3594,\
                          1.7688, 2.2961, 3.0389, 4.3101])
aod_5deg = torch.tensor([0.2236, 0.7064, 1.2461, 1.8578, 2.5642, 3.3986, \
                          4.4220, 5.7403, 7.5974, 10.7753])
aoa_35deg = torch.tensor([1.5679, 4.9447, 8.7224, 13.0045, 17.9492,\
                           23.7899, 30.9538, 40.1824, 53.1816, 75.4274]) # [Table 5.2]
aoa = aod_2deg  
L = 20 # L is the number of sub-paths
ThetaMs = 360*(torch.rand(1)-0.5) # -180 to 180                       
delta_nm_aoa = torch.cat((aoa[:L//2], -aoa[:L//2]))
inds = torch.randperm(L)
delta_nm_aoa_paired = delta_nm_aoa[inds]
theta_nm_aoa = ThetaMs + delta_nm_aoa_paired