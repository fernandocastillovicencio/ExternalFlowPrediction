import torch
import torch.nn as nn
import torch.nn.functional as F

class ReFlowNet(nn.Module):
    def __init__(self, in_channels=5):  # <-- saída fixa: 3 canais totais (Ux, Uy, p)
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

        # Cabeça de saída para Ux e Uy (2 canais)
        self.out_vel = nn.Conv2d(32, 2, kernel_size=1)

        # Cabeça de saída para p (1 canal)
        self.out_p = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Decoder
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Saídas especializadas
        out_vel = self.out_vel(d1)      # [B, 2, H, W]
        out_p   = self.out_p(d1)        # [B, 1, H, W]

        # Concatena: [B, 3, H, W] → [B, H, W, 3]
        out = torch.cat([out_vel, out_p], dim=1)
        return out.permute(0, 2, 3, 1)
