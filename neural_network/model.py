import torch
import torch.nn as nn
import torch.nn.functional as F


class ReFlowNet(nn.Module):
    def __init__(self, in_channels=2):

        super(ReFlowNet, self).__init__()

        # Encoder compartilhado
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.dropout = nn.Dropout2d(p=0.1)

        # Decoder para velocidades (Ux, Uy)
        self.up3_vel = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_vel = self.conv_block(256, 128)

        self.up2_vel = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_vel = self.conv_block(128, 64)

        self.up1_vel = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1_vel = self.conv_block(64, 32)

        self.out_vel = nn.Conv2d(32, 2, kernel_size=1)

        # Decoder para pressão (p)
        self.up3_p = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_p = self.conv_block(256, 128)

        self.up2_p = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_p = self.conv_block(128, 64)

        self.up1_p = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1_p = self.conv_block(64, 32)

        self.out_p = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def crop_or_resize(self, upsampled, skip):
        """Garante que as dimensões espaciais coincidam antes de concatenar."""
        _, _, h, w = upsampled.shape
        if skip.shape[2:] != (h, w):
            skip = F.interpolate(skip, size=(h, w), mode="nearest")
        return torch.cat([upsampled, skip], dim=1)

    def forward(self, x):
        # Define dinamicamente altura e largura do input
        height, width = x.shape[2:]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # ---------------- Decoder Velocidades ----------------
        d3_vel = self.up3_vel(e4)
        d3_vel = self.dropout(self.dec3_vel(self.crop_or_resize(d3_vel, e3)))

        d2_vel = self.up2_vel(d3_vel)
        d2_vel = self.dropout(self.dec2_vel(self.crop_or_resize(d2_vel, e2)))

        d1_vel = self.up1_vel(d2_vel)
        d1_vel = self.dec1_vel(self.crop_or_resize(d1_vel, e1))

        out_vel = self.out_vel(d1_vel)  # [B, 2, H, W]

        # ---------------- Decoder Pressão ----------------
        d3_p = self.up3_p(e4)
        d3_p = self.dropout(self.dec3_p(self.crop_or_resize(d3_p, e3)))

        d2_p = self.up2_p(d3_p)
        d2_p = self.dropout(self.dec2_p(self.crop_or_resize(d2_p, e2)))

        d1_p = self.up1_p(d2_p)
        d1_p = self.dec1_p(self.crop_or_resize(d1_p, e1))

        out_p = self.out_p(d1_p)  # [B, 1, H, W]

        # Concatena as saídas e reorganiza para [B, H, W, 3]
        out = torch.cat([out_vel, out_p], dim=1)

        # Redimensiona dinamicamente para o shape original do input
        out = F.interpolate(
            out, size=(height, width), mode="bilinear", align_corners=False
        )
        return out.permute(0, 2, 3, 1)
