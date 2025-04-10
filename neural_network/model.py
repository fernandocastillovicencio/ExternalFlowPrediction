import torch
import torch.nn as nn

class ReFlowNet(nn.Module):
    def __init__(self):
        super().__init__()

        # MLP Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 4096),
            nn.ReLU()
        )

        # Decoder CNN
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)  # Última camada → 3 canais
        )

    def forward(self, x):
        x = self.encoder(x)                     # (B, 1) → (B, 4096)
        x = x.view(-1, 16, 16, 16)              # (B, 16, 16, 16)
        x = x.permute(0, 1, 3, 2)               # Corrige ordem: (B, 16, 16, 16)
        x = self.decoder(x)                     # (B, 3, H, W)
        x = x[:, :, :100, :100]                 # Crop se ultrapassar 100x100
        x = x.permute(0, 2, 3, 1)               # (B, 100, 100, 3)
        return x
if __name__ == '__main__':
    model = ReFlowNet()
    x = torch.rand((2, 1))
    y = model(x)
    print(f'Saída: {y.shape}')  # Esperado: (2, 100, 100, 3)
