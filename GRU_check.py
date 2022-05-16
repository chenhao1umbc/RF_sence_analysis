
#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
from datetime import datetime
print('starting date time ', datetime.now())
torch.manual_seed(1)

if torch.__version__[:5] != '1.8.1':
    def mydet(x):
        return x.det()
    RAdam = torch.optim.RAdam
else:
    RAdam = optim.RAdam

torch.autograd.set_detect_anomaly(True)
from vae_model import *

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True,)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class Matrix_inverse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
            )
    
    def forward(self, x):
        xx = self.fcn1(x)
        return xx
