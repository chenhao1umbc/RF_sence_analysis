import torch
import torch.nn as nn
torch.pi = torch.acos(torch.zeros(1)).item()*2

class FC_layer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()

        self.layer = nn.Sequential(
        nn.Linear(in_d,out_d),
        nn.BatchNorm1d(out_d),
        nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.layer(x)
        return out


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#%% without batch norm
class FC_layer_(nn.Module):
    "without batchnorm"
    def __init__(self, in_d, out_d):
        super().__init__()

        self.layer = nn.Sequential(
        nn.Linear(in_d,out_d),
        nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.layer(x)
        return out


class DoubleConv_(nn.Module):
    "Without batchnorm"

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_(nn.Module):
    "without batchnorm"

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, larger=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if larger:
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv_(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_(in_channels//2, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


#%% change batch norm to group norm
class FC_layer_g(nn.Module):
    "without batchnorm"
    def __init__(self, in_d, out_d):
        super().__init__()

        self.layer = nn.Sequential(
        nn.Linear(in_d,out_d)
        )
        self.layer2 = nn.Sequential(
        nn.GroupNorm(1,1),
        nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        temp = self.layer(x)
        out = self.layer2(temp[:,None]).squeeze() #[I, n_feat]
        return out


class DoubleConv_g(nn.Module):
    """with group normalization"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(out_channels//4,1), num_channels=mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(out_channels//4,1), num_channels=out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_g(nn.Module):
    "without batchnorm"

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_g(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_g(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, larger=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if larger:
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv_g(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_g(in_channels//2, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

