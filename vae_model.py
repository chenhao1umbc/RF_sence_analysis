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

        dz = 32
        self.encoder = nn.Sequential(
            LinearBlock(dimx, 2560),
            LinearBlock(2560, 2048),
            LinearBlock(2048, 1536),
            LinearBlock(1536, 1024),
            LinearBlock(1024, 512),
            nn.Linear(512, 2*dz*K)
            )
        self.decoder = nn.Sequential(
            LinearBlock(dz*K, 512),
            LinearBlock(512, 1024),
            LinearBlock(1024, 1536),
            LinearBlock(1536, 2048),
            LinearBlock(2048, 2560),
            nn.Linear(2560, dimx)            
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        "Encoder and Get latent variable"
        dz = self.encoder(x)
        mu = dz[:,::2]
        logvar = dz[:,1::2]
        z = self.reparameterize(mu, logvar)
        "Decoder"
        xr = self.decoder(z)
        x_hat = xr.reshape(x.shape)

        return x_hat, z, mu, logvar
