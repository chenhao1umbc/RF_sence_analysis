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


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


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
        self.K, self.M = K, M
        self.encoder = nn.Sequential(
            Down(in_channels=1, out_channels=32),
            DoubleConv(in_channels=32, out_channels=32),
            Down(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=1),
            )
        self.fc1 = nn.Linear(25*25, 2*self.dz)
        self.decoder = nn.Sequential(
            DoubleConv(in_channels=self.dz+2, out_channels=32),
            DoubleConv(in_channels=32, out_channels=32),
            DoubleConv(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=1),
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
            LinearBlock(self.dz*self.K, 64),
            nn.Linear(64, 1),
            )   
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, h_old, Rs_old, Rb_old):
        "get estimated sources: s"
        Rx = h_old@Rs_old.permute(1,2,0,3,4)@h_old.transpose(-1,-2).conj()+Rb_old # [N,F,I,M,M]
        W = Rs_old.permute(1,2,0,3,4)@h_old.transpose(-1,-2).conj()@Rx.inverse() # [N,F,I,J,M]
        s = (W.permute(2,0,1,3,4) @ x.permute(0,2,3,1)[...,None]).squeeze().abs()
        
        zk ,v_all, h_all = [], [], []
        for i in range(self.K):
            "Encoder"
            ss = self.encoder(s[:,None, :,:,i])
            batch_size = ss.shape[0]
            "Get latent variable"
            zz = self.fc1(ss.reshape(batch_size,-1))
            mu = zz[:,::2]
            logvar = zz[:,1::2]
            zk.append(self.reparameterize(mu, logvar))

            "Decoder1 get V"
            # View z as 4D tensor to be tiled across new N and F dimensions            
            zr = zk[i].view(zk[i].shape + (1, 1))  #Shape: IxDxNxF
            # Tile across to match image size
            zr = zr.expand(-1, -1, self.im_size, self.im_size)  #Shape: IxDx64x64
            # Expand grids to batches and concatenate on the channel dimension
            zbd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                        self.y_grid.expand(batch_size, -1, -1, -1), zr), dim=1) # Shape: Ix(dz*K+2)xNxF
            v = self.decoder(zbd).exp()
            v_all.append(threshold(v, ceiling=1e4)) # 1e-3 to 1e4
            "Decoder2 get H"
            ang = self.fc_h(zk[i])
            h_all.append((ang*torch.pi*1j*torch.arange(self.M, device=ang.device)).exp())
        "Decoder3 get sig_b"
        sig_b = self.fc_b(torch.cat(zk, dim=1)).exp()

        vhat = torch.stack(v_all, 4).squeeze() # shape:[I, N, F, K], float32
        Hhat = torch.stack(h_all, 2) # shape:[I, M, K], cfloat
        Rb = sig_b[:,:,None]**2 * torch.ones(batch_size, \
            self.M, device=sig_b.device).diag_embed() # shape:[I, M, M], float32

        return vhat.diag_embed(), Hhat, Rb, mu, logvar


class NN1(nn.Module):
    """This is spatial broadcast decoder (SBD) version, similar to Neri's
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M=3, K=3, im_size=100):
        super().__init__()

        # Estimate V
        self.dz = 32
        self.K, self.M = K, M
        self.encoder = nn.Sequential(
            Down(in_channels=M*2, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            Down(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=K),
            )
        self.fc1 = nn.Linear(25*25*K, 2*self.dz*K)
        self.decoder = nn.Sequential(
            DoubleConv(in_channels=self.dz+2, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            DoubleConv(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=1),
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
            LinearBlock(self.dz*self.K, 64),
            nn.Linear(64, 1),
            )   
    
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

        "Decoders"
        rz = z.reshape(batch_size, self.K, self.dz)
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
        sig_b = self.fc_b(z).exp()

        vhat = torch.stack(v_all, 4).squeeze().to(torch.cfloat) # shape:[I, N, F, K]
        Hhat = torch.stack(h_all, 2) # shape:[I, M, K], cfloat
        Rb = sig_b[:,:,None]**2 * torch.ones(batch_size, \
            self.M, device=sig_b.device).diag_embed().to(torch.cfloat) # shape:[I, M, M]

        return vhat.diag_embed(), Hhat, Rb, mu, logvar


class NN2(nn.Module):
    """This is input is x and H_init
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M=3, K=3, im_size=100):
        super().__init__()

        # Estimate V
        self.dz = 32
        self.K, self.M = K, M
        self.encoder = nn.Sequential(
            Down(in_channels=1, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            Down(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=1),
            )
        self.fc1 = nn.Linear(25*25, 2*self.dz)
        self.decoder = nn.Sequential(
            DoubleConv(in_channels=self.dz+2, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            DoubleConv(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=4),
            OutConv(in_channels=4, out_channels=1),
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
            LinearBlock(self.dz*self.K, 64),
            nn.Linear(64, 1),
            )   
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, h_int):
        batch_size = x.shape[0]
        z_all, v_all, h_all = [], [], [] 
        for i in range(self.K):
            "Encoder"
            inp = h_int[:,i:i+1].t().conj()@x.permute(0,2,3,1).unsqueeze(-1)
            inp = inp.squeeze().abs()
            xx = self.encoder(inp[:,None,:,:])
            "Get latent variable"
            zz = self.fc1(xx.reshape(batch_size,-1))
            mu = zz[:,::2]
            logvar = zz[:,1::2]
            z = self.reparameterize(mu, logvar)
            z_all.append(z)
            
            "Decoder1 get V"
            # View z as 4D tensor to be tiled across new N and F dimensions            
            zr = z.view((batch_size, self.dz)+ (1, 1))  #Shape: IxDxNxF
            # Tile across to match image size
            zr = zr.expand(-1, -1, self.im_size, self.im_size)  #Shape: IxDx64x64
            # Expand grids to batches and concatenate on the channel dimension
            zbd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                        self.y_grid.expand(batch_size, -1, -1, -1), zr), dim=1) # Shape: Ix(dz*K+2)xNxF
            v = self.decoder(zbd).exp()
            v_all.append(threshold(v, floor=1e-6, ceiling=1e3)) # 1e-6 to 1e3
            "Decoder2 get H"
            ang = self.fc_h(z)
            h_all.append((ang*torch.pi*1j*torch.arange(self.M, device=ang.device)).exp())
        "Decoder3 get sig_b"
        sig_b = self.fc_b(torch.cat(z_all, dim=-1)).exp()

        vhat = torch.stack(v_all, 4).squeeze().to(torch.cfloat) # shape:[I, N, F, K]
        Hhat = torch.stack(h_all, 2) # shape:[I, M, K], cfloat
        Rb = threshold(sig_b[:,:,None]**2, 1e-6, 1e3)*torch.ones(batch_size, \
            self.M, device=sig_b.device).diag_embed().to(torch.cfloat) # shape:[I, M, M]

        return vhat.diag_embed(), Hhat, Rb, mu, logvar


class NN3(nn.Module):
    """This is a matched filter version
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M=3, K=3, im_size=100):
        super().__init__()

        # Estimate V
        self.dz = 32
        self.K, self.M = K, M
        self.encoder = nn.Sequential(
            Down(in_channels=1, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            Down(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=1),
            )
        self.fc1 = nn.Linear(25*25, 2*self.dz)
        self.decoder = nn.Sequential(
            DoubleConv(in_channels=self.dz+2, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            DoubleConv(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=4),
            OutConv(in_channels=4, out_channels=1),
            ) 

        self.im_size = im_size
        x = torch.linspace(-1, 1, im_size)
        y = torch.linspace(-1, 1, im_size)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

        # Estimate H
        self.h_net = nn.Sequential(
            Down(in_channels=M*2, out_channels=32),
            Down(in_channels=32, out_channels=16),
            Down(in_channels=16, out_channels=8),
            Reshape(-1, 8*12*12),
            LinearBlock(8*12*12, 64),
            nn.Linear(64, K),
            nn.Tanh()
            )   
        
        # Estimate Rb
        self.fc_b = nn.Sequential(
            LinearBlock(self.dz*self.K, 64),
            nn.Linear(64, 1),
            )   
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        batch_size = x.shape[0]
        z_all, v_all  = [], []

        "Neural nets for H"
        ang = self.h_net(torch.cat((x.real, x.imag), dim=1))
        ch = torch.pi*torch.arange(self.M, device=ang.device)
        Hhat = ((ch[:,None] @ ang[:,None])*1j).exp()
        for i in range(self.K):
            "Encoder"
            inp = Hhat[...,i:i+1].transpose(-1,-2).conj()@x.permute(2,3,0,1).unsqueeze(-1)
            inp = inp.squeeze().abs().permute(2,0,1)
            xx = self.encoder(inp[:,None,:,:])
            "Get latent variable"
            zz = self.fc1(xx.reshape(batch_size,-1))
            mu = zz[:,::2]
            logvar = zz[:,1::2]
            z = self.reparameterize(mu, logvar)
            z_all.append(z)
            
            "Decoder to get V"
            # View z as 4D tensor to be tiled across new N and F dimensions            
            zr = z.view((batch_size, self.dz)+ (1, 1))  #Shape: IxDxNxF
            # Tile across to match image size
            zr = zr.expand(-1, -1, self.im_size, self.im_size)  #Shape: IxDx64x64
            # Expand grids to batches and concatenate on the channel dimension
            zbd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                        self.y_grid.expand(batch_size, -1, -1, -1), zr), dim=1) # Shape: Ix(dz*K+2)xNxF
            v = self.decoder(zbd).exp()
            v_all.append(threshold(v, floor=1e-4, ceiling=1e3)) # 1e-6 to 1e3

        "Decoder3 get sig_b"
        sig_b = self.fc_b(torch.cat(z_all, dim=-1)).exp()

        vhat = torch.stack(v_all, 4).squeeze().to(torch.cfloat) # shape:[I, N, F, K]
        Rb = threshold(sig_b[:,:,None]**2, 1e-4, 1e3)*torch.ones(batch_size, \
            self.M, device=sig_b.device).diag_embed().to(torch.cfloat) # shape:[I, M, M]

        return vhat.diag_embed(), Hhat, Rb, mu, logvar


class NN4(nn.Module):
    """This is Wiener filter version
    Input shape [I,M,N,F], e.g.[32,3,100,100]
    J <=K
    """
    def __init__(self, M=3, K=3, im_size=100):
        super().__init__()
        self.dz = 32
        self.K, self.M = K, M

        # Estimate H and coarse V
        self.v_net = nn.Sequential(
            DoubleConv(in_channels=M*2, out_channels=32),
            DoubleConv(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=4),
            ) 
        self.v_out = OutConv(in_channels=4, out_channels=K)
        self.hb_net = nn.Sequential(
            Down(in_channels=4, out_channels=32),
            Down(in_channels=32, out_channels=16),
            Down(in_channels=16, out_channels=8),
            Reshape(-1, 8*12*12),
            )
        # Estimate H
        self.h_net = nn.Sequential(
            LinearBlock(8*12*12, 64),
            nn.Linear(64, K),
            nn.Tanh()
            )   
        # Estimate Rb
        self.b_net = nn.Sequential(
            LinearBlock(8*12*12, 64),
            nn.Linear(64, 1),
            )   
        # Estimate V using auto encoder
        self.encoder = nn.Sequential(
            Down(in_channels=1, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            Down(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=1),
            )
        self.fc1 = nn.Linear(25*25, 2*self.dz)
        self.decoder = nn.Sequential(
            DoubleConv(in_channels=self.dz+2, out_channels=64),
            DoubleConv(in_channels=64, out_channels=32),
            DoubleConv(in_channels=32, out_channels=16),
            DoubleConv(in_channels=16, out_channels=4),
            OutConv(in_channels=4, out_channels=1),
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
        batch_size, _, N, F = x.shape
        z_all, v_all  = [], []

        "Neural nets for H,V,b"
        temp = self.v_net(torch.cat((x.real, x.imag), dim=1))
        V_coarse = threshold(self.v_out(temp), 1e-4, 1e3).to(torch.cfloat)
        hb = self.hb_net(temp)
        ang = self.h_net(hb)
        sig_b = self.b_net(hb).exp()
        "Get H"
        ch = torch.pi*torch.arange(self.M, device=ang.device)
        Hhat = ((ch[:,None] @ ang[:,None])*1j).exp()
        "Get Rb"
        Rb = threshold(sig_b[:,:,None]**2, 1e-4, 1e2)*torch.ones(batch_size, \
            self.M, device=sig_b.device).diag_embed().to(torch.cfloat) # shape:[I, M, M]
        "Wienter filter to get coarse shat"
        Rs = V_coarse.reshape(batch_size,self.K,N,F).permute(0,2,3,1).diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        W = Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() @ Rx.inverse()  # shape of [N, F, I, J, M]
        shat = (W.permute(2,0,1,3,4) @ x.permute(0,2,3,1)[...,None]).squeeze()
        
        for i in range(self.K):
            "Encoder"
            xx = self.encoder(shat[...,i].abs()[:,None])
            "Get latent variable"
            zz = self.fc1(xx.reshape(batch_size,-1))
            mu = zz[:,::2]
            logvar = zz[:,1::2]
            z = self.reparameterize(mu, logvar)
            z_all.append(z)
            
            "Decoder to get V"
            # View z as 4D tensor to be tiled across new N and F dimensions            
            zr = z.view((batch_size, self.dz)+ (1, 1))  #Shape: IxDxNxF
            # Tile across to match image size
            zr = zr.expand(-1, -1, self.im_size, self.im_size)  #Shape: IxDx64x64
            # Expand grids to batches and concatenate on the channel dimension
            zbd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                        self.y_grid.expand(batch_size, -1, -1, -1), zr), dim=1) # Shape: Ix(dz*K+2)xNxF
            v = self.decoder(zbd).exp()
            v_all.append(threshold(v, floor=1e-4, ceiling=1e3)) # 1e-4 to 1e3
        vhat = torch.stack(v_all, 4).squeeze().to(torch.cfloat) # shape:[I, N, F, K]

        return vhat.diag_embed(), Hhat, Rb, mu, logvar
