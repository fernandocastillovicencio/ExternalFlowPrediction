import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding se necessário para ajustar tamanhos
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ReFlowUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 4096),
            nn.ReLU()
        )

        # Reshape para imagem (16x16x16)
        self.init_conv = nn.Conv2d(16, 64, kernel_size=3, padding=1)

        # U-Net Encoder
        self.enc1 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)

        # Decoder
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)                      # (B, 4096)
        x = x.view(-1, 16, 16, 16)               # (B, C=16, H=16, W=16)
        x = x.permute(0, 1, 3, 2)                # (B, C=16, H=16, W=16)
        x = self.init_conv(x)                    # (B, 64, 16, 16)

        # Encoder
        x1 = self.enc1(x)                        # (B, 64, 16, 16)
        x2 = self.enc2(self.pool1(x1))           # (B, 128, 8, 8)
        x3 = self.enc3(self.pool2(x2))           # (B, 256, 4, 4)

        # Decoder
        x = self.up1(x3, x2)                     # (B, 128, 8, 8)
        x = self.up2(x, x1)                      # (B, 64, 16, 16)
        x = F.interpolate(x, size=(100, 100), mode='bilinear', align_corners=False)
        x = self.final_conv(x)                   # (B, 3, 100, 100)
        return x.permute(0, 2, 3, 1)             # (B, 100, 100, 3)




# -------------------------------------------------------- #
#                           TESTE                          #
# -------------------------------------------------------- #
if __name__ == '__main__':
    model = ReFlowUNet()
    x = torch.rand((2, 1))
    y = model(x)
    print(f'Saída: {y.shape}')  # Esperado: (2, 100, 100, 3)
