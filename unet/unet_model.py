""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch.nn as nn
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64 * factor, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


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


class MyUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels//2, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class UNetHalf4to150(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet

        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf4to150, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.up5 = Up_(self.n_ch//16, self.n_ch//32, bilinear)
        self.up6 = Up_(self.n_ch//32, self.n_ch//32, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//32, self.n_ch//16, kernel_size=3, padding=1, stride=2),
            nn.ConvTranspose2d(self.n_ch//16, 16, kernel_size=5, dilation=3, output_padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, dilation=3, output_padding=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)  # output has W=256, H=256, for gamma = 16
        x = self.up5(x) # input has W=32, H=32, for gamma = 2
        x = self.up6(x)
        x = self.reshape(x) # input 256 output 150
        out = self.outc(x)
        return out


class UNetHalf4to50(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet

        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf4to50, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=3, padding=1, stride=2),
            nn.ConvTranspose2d(self.n_ch//16, 16, kernel_size=5, dilation=3, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, dilation=3, output_padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)  # output has W=256, H=256, for gamma = 16
        x = self.reshape(x) # input 256 output 150
        out = self.outc(x)
        return out


class UNetHalf8to50(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to50, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//8, self.n_ch//8, kernel_size=3, padding=1, stride=2),
            nn.ConvTranspose2d(self.n_ch//8, 16, kernel_size=5, dilation=3, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, dilation=3, output_padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.reshape(x) # input 256 output 150
        out = self.outc(x)
        return out


class UNetHalf8to128(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to128, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100_morelayers(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_morelayers, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100_16_FC_128(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_16_FC_128, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128
        
        self.fc = nn.Linear(9, 8)
        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.fc(x)
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100_16_FC(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_16_FC, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256
        
        self.fc = nn.Linear(9, 8)
        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.fc(x)
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100_stack(nn.Module):
    "16 layers here, input is not added noise, just stack noise and resized mixture"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_stack, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = F.max_pool2d(x, (2,1))
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100_stack_256(nn.Module):
    "16 layers here, input is not added noise, just stack noise and resized mixture"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_stack_256, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = F.max_pool2d(x, (2,1))
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100_stack2(nn.Module):
    "16 layers here, input is not added noise, just stack noise and resized mixture"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_stack2, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.MaxPool2d((2,1)),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100_stack2_256(nn.Module):
    "16 layers here, input is not added noise, just stack noise and resized mixture"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_stack2_256, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.MaxPool2d((2,1)),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100_256(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_256, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100_256_sig(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_256_sig, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        out = self.sig(x)
        return out


class UNetHalf8to100_256_bnsig(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_256_bnsig, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.bn = nn.BatchNorm2d(n_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        x = self.bn(x)
        out = self.sig(x)
        return out


class UNetHalf8to100_256_bnsig2(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_256_bnsig2, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, n_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_classes),
            nn.LeakyReLU(inplace=True))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.sig(x)
        return out


class UNetHalf8to100_256_bnsig3(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_256_bnsig3, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_classes),
            nn.LeakyReLU(inplace=True) )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.sig(x)
        return out


class UNetHalf8to100_256_bnsig4(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_256_bnsig4, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=3, padding=1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.sig(x)
        return out


class UNetHalf8to100_256_bnsig5(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_256_bnsig5, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=3, padding=1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.sig(x)
        out = x/x.detach().amax(keepdim=True, dim=(-1,-2))
        return out


class UNetHalf8to100_relu(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_relu, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        out = self.relu(x)
        return out


class UNetHalf8to100_vjto1(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_vjto1, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        x = self.sig(x) # output shape of [nsamples,1, 100, 100]
        out = x/x.amax(keepdim=True, dim=(-1,-2))
        return out


class UNetHalf8to100_vjto1_2(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_vjto1_2, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        x = self.relu(x) # output shape of [nsamples,1, 100, 100]
        out = x/x.amax(keepdim=True, dim=(-1,-2))
        return out


class UNetHalf8to100_vjto1_3(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_vjto1_3, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=3, padding=1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.relu(x) # output shape of [nsamples,1, 100, 100]
        out = x/x.amax(keepdim=True, dim=(-1,-2))
        return out


class UNetHalf8to100_vjto1_4(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_vjto1_4, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=3, padding=1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.sig(x) # output shape of [nsamples,1, 100, 100]
        out = x/x.amax(keepdim=True, dim=(-1,-2))
        return out


class UNetHalf8to100_vjto1_5(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_vjto1_5, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        x = self.sig(x)
        out = x/x.detach().amax(keepdim=True, dim=(-1,-2))
        return out


class UNetHalf8to100_vjto1_6(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_vjto1_6, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256+128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        x = self.sig(x)
        out = x/x.detach().amax(keepdim=True, dim=(-1,-2))
        return out


class UNetHalf8to100_vjto1_7(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_vjto1_7, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 320

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        x = self.sig(x)
        out = x/x.detach().amax(keepdim=True, dim=(-1,-2))
        return out


class UNetHalf8to100_vjto1_lsbn(nn.Module):
    "less batch norm"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_vjto1_lsbn, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        x = self.sig(x) # output shape of [nsamples,1, 100, 100]
        out = x/x.amax(keepdim=True, dim=(-1,-2))
        return out


class UNetHalf8to100_lsbn(nn.Module):
    "less batch norm withouth vj to 1"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_lsbn, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        x = self.sig(x) # output shape of [nsamples,1, 100, 100]
        return x


class UNetHalf8to100_lsbn2(nn.Module):
    "less batch norm withouth vj to 1"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_lsbn2, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.bn = nn.BatchNorm2d(n_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        x = self.bn(x)
        x = self.sig(x) # output shape of [nsamples,1, 100, 100]
        return x


class UNetHalf8to100_256_stack1(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_256_stack1, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=(1,0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True) )
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.conv(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100_stack3(nn.Module):
    "16 layers here, input is not added noise, just stack noise and resized mixture"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_stack3, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3, stride=(1,2)),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=(3,4)),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=(1,0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100_stack3_256(nn.Module):
    "16 layers here, input is not added noise, just stack noise and resized mixture"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_stack3_256, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3, stride=(1,2)),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=(3,4)),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=(1,0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf8to100_19(nn.Module):
    "16 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf8to100_19, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, self.n_ch//8, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, self.n_ch//8, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, self.n_ch//8, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class UNetHalf16to100(nn.Module):
    "14 layers here"
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf16to100, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//8, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out


class Model171000(nn.Module):
    "UNetHalf, size 8to100, 256 inner channel, 16 layers, relu "
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(Model171000, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True) 
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        x = self.outc(x)
        out = self.relu(x)
        return out


class Model171010(nn.Module):
    "UNetHalf, size 8to100, 256 inner channel, 16 layers, e^x "
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet
        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(Model171010, self).__init__()
        self.n_ch = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 256

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, True) 
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x) 
        out = self.outc(x)
        return out.exp()


class SBD1(nn.Module):
    """This is spatial broadcast decoder (SBD) version
    Input shape [I, d_gamma], e.g.[32,64]"""
    def __init__(self, dz=32, im_size=100, max_ch=128):
        super(SBD1, self).__init__()
        self.decoder = nn.Sequential(
            DoubleConv(in_channels=dz+2, out_channels=max_ch),
            DoubleConv(in_channels=max_ch, out_channels=max_ch//2),
            OutConv(max_ch//2, 1),
            nn.Sigmoid()
            ) 

        self.im_size = im_size
        x = torch.linspace(-1, 1, im_size)
        y = torch.linspace(-1, 1, im_size)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

    def forward(self, gamma):
        batch_size = gamma.size(0)
        # View z as 4D tensor to be tiled across new H and W dimensions
        # Shape: NxDx1x1
        x = gamma.view(gamma.shape + (1, 1))
        x = x.expand(-1, -1, self.im_size, self.im_size)
        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)xim_sizexim_size
        zbd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                       self.y_grid.expand(batch_size, -1, -1, -1), x), dim=1)
        x_hat = self.decoder(zbd)
        out = x_hat/x_hat.detach().amax(keepdim=True, dim=(-1,-2))

        return out


# Full UNet shape structure
class UNet8to100(nn.Module):
    "Too large to train"
    def __init__(self, n_channels, n_classes):
        super(UNet8to100, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_ch = 128

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.down1 = Down(self.n_ch, self.n_ch//2)
        self.down2 = Down(self.n_ch//2, self.n_ch//4)
        self.down3 = Down(self.n_ch//4, self.n_ch//8)
        self.trim = nn.Sequential(
            nn.Conv2d(self.n_ch//8, self.n_ch//16, kernel_size=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, 1, kernel_size=3),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True))            

        self.up1 = MyUp(self.n_ch, self.n_ch//2)
        self.up2 = MyUp(self.n_ch//2, self.n_ch//4)
        self.up3 = MyUp(self.n_ch//4, self.n_ch//8)
        self.up4 = MyUp(self.n_ch//8, self.n_ch//8)
        self.reshape = nn.Sequential(
            nn.MaxPool2d((2,1)),
            nn.Conv2d(self.n_ch//8, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//16, kernel_size=5, dilation=3),
            nn.BatchNorm2d(self.n_ch//16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//16, self.n_ch//8, kernel_size=3, dilation=2),
            nn.BatchNorm2d(self.n_ch//8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_ch//8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.outc = OutConv(32, n_classes)

    def forward(self, x, gamma):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5  = self.trim(x4)
        # stacking x and gamma to have regular larger input size
        x = torch.cat((x5, gamma), dim=-2)
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.reshape(x)
        out = self.outc(x)
        return out
