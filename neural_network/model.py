import torch
import torch.nn as nn
import torch.nn.functional as F

class ReFlowNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):  # <-- atualizado para 3 canais
        super(ReFlowNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)

        # Output
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # [B, 32, 100, 100]
        e2 = self.enc2(self.pool(e1))  # [B, 64, 50, 50]
        e3 = self.enc3(self.pool(e2))  # [B, 128, 25, 25]

        # Decoder
        d2 = self.up2(e3)        # [B, 64, 50, 50]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)        # [B, 32, 100, 100]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)  # [B, 3, 100, 100]
        return out.permute(0, 2, 3, 1)  # [B, 100, 100, 3]
