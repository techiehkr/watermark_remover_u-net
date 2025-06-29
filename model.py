import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, up_in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch=out_ch + skip_ch, out_ch=out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad if needed to match spatial size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=3):
        super().__init__()
        self.inc = ConvBlock(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, 512))

        # Corrected channel sizes for skip connections:
        self.up1 = Up(up_in_ch=512, skip_ch=512, out_ch=256)  
        self.up2 = Up(up_in_ch=256, skip_ch=256, out_ch=128)  
        self.up3 = Up(up_in_ch=128, skip_ch=128, out_ch=64)   
        self.up4 = Up(up_in_ch=64, skip_ch=64, out_ch=64)     

        self.outc = nn.Conv2d(64, out_ch, 1)

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

        return torch.sigmoid(self.outc(x))
