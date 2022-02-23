import torch
import torch.nn as nn
from unet.unet_parts import *


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
        chans = (700, 600, 500, 400, 300)
        # chans = (2560, 2048, 1536, 1024, 512)
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
            Down(in_channels=dz*K, out_channels=64),
            Down(in_channels=64, out_channels=M),
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
        sizes = x.shape
        "Get latent variable"
        dz = self.fc1(x.reshape(sizes[0],-1))
        mu = dz[:,::2]
        logvar = dz[:,1::2]
        z = self.reparameterize(mu, logvar)

        "Decoder"
        batch_size = z.size(0)
        # View z as 4D tensor to be tiled across new H and W dimensions
        # Shape: NxDx1x1
        z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.im_size, self.im_size)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        zbd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                       self.y_grid.expand(batch_size, -1, -1, -1), z), dim=1)
        x_hat = self.decoder(zbd)

        return x_hat, z, mu, logvar