import torch
import torch.nn as nn
from unet.unet_parts import *
from utils import threshold
torch.pi = torch.acos(torch.zeros(1)).item()*2

class Up_(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, larger=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if larger:
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//2, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class LinearBlock(nn.Module):
    def __init__(self, in_channels,out_channels,activation=True):
        super(LinearBlock, self).__init__()
        if activation is True:
            self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                )
        else:
            self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                )

    def forward(self, x):
        return self.block(x)


class VAE1(nn.Module):
    """This is convolutional version -- ref UNet
    Input shape [I,M,N,F], e.g.[32,3,100,100]"""
    def __init__(self, M=3, K=3):
        super(VAE1, self).__init__()
        dz = 32
        self.K = K
        self.encoder = nn.Sequential(
            Down(in_channels=M, out_channels=64),
            Down(in_channels=64, out_channels=K),
            )
        self.fc1 = nn.Linear(25*25*K, 2*dz*K)
        self.fc2 = nn.Linear(dz*K, 25*25*K)
        self.decoder = nn.Sequential(
            Up_(in_channels=K, out_channels=64),
            Up_(in_channels=64, out_channels=M),
            ) 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        "Encoder"
        x = self.encoder(x)
        sizes = x.shape
        "Get latent variable"
        dz = self.fc1(x.reshape(sizes[0],-1))
        mu = dz[:,::2]
        logvar = dz[:,1::2]
        z = self.reparameterize(mu, logvar)
        "Decoder"
        x = self.fc2(z)
        xr = x.reshape(sizes[0],self.K,25,25)
        x_hat = self.decoder(xr)

        return x_hat, z, mu, logvar


class VAE2(nn.Module):
    """This is MLP version  -- ref VAESS
    Input shape [I,MNF], e.g.[32, 3*100*100]"""
    def __init__(self, dimx=30000, K=3):
        super(VAE2, self).__init__()

        self.K = K
        self.dz = 32
        # chans = (700, 600, 500, 400, 300)
        chans = (2560, 2048, 1536, 1024, 512)
        self.encoder = nn.Sequential(
            LinearBlock(dimx, chans[0]),
            LinearBlock(chans[0],chans[1]),
            LinearBlock(chans[1],chans[2]),
            LinearBlock(chans[2],chans[3]),
            LinearBlock(chans[3],chans[4]),
            nn.Linear(chans[4], 2*self.dz*K)
            )
        self.decoder = nn.Sequential(
            LinearBlock(self.dz, chans[4]),
            LinearBlock(chans[4],chans[3]),
            LinearBlock(chans[3],chans[2]),
            LinearBlock(chans[2],chans[1]),
            LinearBlock(chans[1],chans[0]),
            LinearBlock(chans[0],dimx,activation=False),
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        "Encoder and Get latent variable"
        zz = self.encoder(x)
        mu = zz[:,::2]
        logvar = zz[:,1::2]
        z = self.reparameterize(mu, logvar)
        "Decoder"
        sources = self.decoder(z.view(-1,self.dz))
        s = sources.view(-1,self.K, x.shape[-1])
        x_hat = s.sum(1)

        return x_hat, z, mu, logvar, s


class VAE3(nn.Module):
    """This is spatial broadcast decoder (SBD) version
    Input shape [I,M,N,F], e.g.[32,3,100,100]"""
    def __init__(self, M=3, K=3, im_size=100):
        super().__init__()
        self.dz = 32
        self.K, self.M = K, M
        self.encoder = nn.Sequential(
            Down(in_channels=M, out_channels=64),
            Down(in_channels=64, out_channels=K),
            )
        self.fc1 = nn.Linear(25*25*K, 2*self.dz*K)
        self.fc2 = nn.Linear(self.dz*K, 25*25*K)
        self.decoder = nn.Sequential(
            DoubleConv(in_channels=self.dz+2, out_channels=64),
            DoubleConv(in_channels=64, out_channels=M),
            ) 

        self.im_size = im_size
        x = torch.linspace(-1, 1, im_size)
        y = torch.linspace(-1, 1, im_size)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        "Encoder"
        x = self.encoder(x)
        batch_size = x.shape[0]
        "Get latent variable"
        zz = self.fc1(x.reshape(batch_size,-1))
        mu = zz[:,::2]
        logvar = zz[:,1::2]
        z = self.reparameterize(mu, logvar)

        "Decoder"
        rz = z.reshape(batch_size,self.K, self.dz)
        ss_all = []
        for i in range(self.K):
            # View z as 4D tensor to be tiled across new N and F dimensions            
            zr = rz[:,i].view(rz[:,i].shape + (1, 1))  #Shape: IxDxNxF
            # Tile across to match image size
            zr = zr.expand(-1, -1, self.im_size, self.im_size)  #Shape: IxDx64x64
            # Expand grids to batches and concatenate on the channel dimension
            zbd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                        self.y_grid.expand(batch_size, -1, -1, -1), zr), dim=1) # Shape: Ix(dz*K+2)xNxF
            ss = self.decoder(zbd)
            ss_all.append(ss)
        s = torch.stack(ss_all, 2) # shape of [I, M, K, N, F]
        x_hat = s.sum(2)
        return x_hat, z, mu, logvar, s


class NN(nn.Module):
    """This is spatial broadcast decoder (SBD) version
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M=3, K=3, im_size=100):
        super().__init__()

        # Estimate V
        self.dz = 32
        self.K, self.M = K, M
        self.encoder = nn.Sequential(
            Down(in_channels=M, out_channels=64),
            Down(in_channels=64, out_channels=K+1),
            )
        self.fc1 = nn.Linear(25*25*(K+1), 2*self.dz*(K+1))
        self.decoder = nn.Sequential(
            DoubleConv(in_channels=self.dz+2, out_channels=64),
            DoubleConv(in_channels=64, out_channels=1),
            ) 

        self.im_size = im_size
        x = torch.linspace(-1, 1, im_size)
        y = torch.linspace(-1, 1, im_size)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

        # Estimate H
        self.fc_h = nn.Sequential(
            LinearBlock(self.dz, 64),
            nn.Linear(64, 1),
            nn.Tanh()
            )   
        
        # Estimate Rb
        self.fc_b = nn.Sequential(
            LinearBlock(self.dz, 64),
            nn.Linear(64, 1),
            )   
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        "Encoder"
        x = self.encoder(x.abs())
        batch_size = x.shape[0]
        "Get latent variable"
        zz = self.fc1(x.reshape(batch_size,-1))
        mu = zz[:,::2]
        logvar = zz[:,1::2]
        z = self.reparameterize(mu, logvar)

        "Decoders"
        rz = z.reshape(batch_size, self.K+1, self.dz)
        v_all, h_all = [], []
        for i in range(self.K):
            "Decoder1 get V"
            # View z as 4D tensor to be tiled across new N and F dimensions            
            zr = rz[:,i].view(rz[:,i].shape + (1, 1))  #Shape: IxDxNxF
            # Tile across to match image size
            zr = zr.expand(-1, -1, self.im_size, self.im_size)  #Shape: IxDx64x64
            # Expand grids to batches and concatenate on the channel dimension
            zbd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                        self.y_grid.expand(batch_size, -1, -1, -1), zr), dim=1) # Shape: Ix(dz*K+2)xNxF
            v = self.decoder(zbd).exp()
            v_all.append(threshold(v, ceiling=1e4)) # 1e-3 to 1e4
            "Decoder2 get H"
            ang = self.fc_h(rz[:, i])
            h_all.append((ang*torch.pi*1j*torch.arange(self.M, device=ang.device)).exp())
        "Decoder3 get sig_b"
        sig_b = self.fc_b(rz[:, -1]).exp()

        vhat = torch.stack(v_all, 4).squeeze() # shape:[I, N, F, K], float32
        Hhat = torch.stack(h_all, 2) # shape:[I, M, K], cfloat
        Rb = sig_b[:,:,None]**2 * torch.ones(batch_size, \
            self.M, device=sig_b.device).diag_embed() # shape:[I, M, M], float32

        return vhat, Hhat, Rb, mu, logvar


class NN0(nn.Module):
    """This is spatial broadcast decoder (SBD) version
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M=3, K=3, im_size=100):
        super().__init__()

        # Estimate V
        self.dz = 32
        self.K, self.M = K, M//2
        self.encoder = nn.Sequential(
            Down(in_channels=M, out_channels=64),
            Down(in_channels=64, out_channels=K+1),
            )
        self.fc1 = nn.Linear(25*25*(K+1), 2*self.dz*(K+1))
        self.decoder = nn.Sequential(
            DoubleConv(in_channels=self.dz+2, out_channels=64),
            DoubleConv(in_channels=64, out_channels=1),
            ) 

        self.im_size = im_size
        x = torch.linspace(-1, 1, im_size)
        y = torch.linspace(-1, 1, im_size)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

        # Estimate H
        self.fc_h = nn.Sequential(
            LinearBlock(self.dz, 64),
            nn.Linear(64, 1),
            nn.Tanh()
            )   
        
        # Estimate Rb
        self.fc_b = nn.Sequential(
            LinearBlock(self.dz, 64),
            nn.Linear(64, 1),
            )   
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        "Encoder"
        x = self.encoder(torch.cat((x.real, x.imag), dim=1))
        batch_size = x.shape[0]
        "Get latent variable"
        zz = self.fc1(x.reshape(batch_size,-1))
        mu = zz[:,::2]
        logvar = zz[:,1::2]
        z = self.reparameterize(mu, logvar)

        "Decoders"
        rz = z.reshape(batch_size, self.K+1, self.dz)
        v_all, h_all = [], []
        for i in range(self.K):
            "Decoder1 get V"
            # View z as 4D tensor to be tiled across new N and F dimensions            
            zr = rz[:,i].view(rz[:,i].shape + (1, 1))  #Shape: IxDxNxF
            # Tile across to match image size
            zr = zr.expand(-1, -1, self.im_size, self.im_size)  #Shape: IxDx64x64
            # Expand grids to batches and concatenate on the channel dimension
            zbd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                        self.y_grid.expand(batch_size, -1, -1, -1), zr), dim=1) # Shape: Ix(dz*K+2)xNxF
            v = self.decoder(zbd).exp()
            v_all.append(threshold(v, ceiling=1e4)) # 1e-3 to 1e4
            "Decoder2 get H"
            ang = self.fc_h(rz[:, i])
            h_all.append((ang*torch.pi*1j*torch.arange(self.M, device=ang.device)).exp())
        "Decoder3 get sig_b"
        sig_b = self.fc_b(rz[:, -1]).exp()

        vhat = torch.stack(v_all, 4).squeeze() # shape:[I, N, F, K], float32
        Hhat = torch.stack(h_all, 2) # shape:[I, M, K], cfloat
        Rb = sig_b[:,:,None]**2 * torch.ones(batch_size, \
            self.M, device=sig_b.device).diag_embed() # shape:[I, M, M], float32

        return vhat, Hhat, Rb, mu, logvar