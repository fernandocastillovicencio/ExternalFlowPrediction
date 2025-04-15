import torch
import torch.nn as nn
import torch.nn.functional as F

class ReFlowNet(nn.Module):
    def __init__(self):
        super(ReFlowNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 100 * 100 * 16),
            nn.ReLU()
        )

        self.decoder_velocity = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1)  # Ux, Uy
        )

        self.decoder_pressure = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  # p
        )

    def forward(self, x):
        x = self.fc(x)  # [B, 1] -> [B, 100*100*16]
        x = x.view(-1, 16, 100, 100)  # reshape para imagem

        vel_out = self.decoder_velocity(x)  # [B, 2, 100, 100]
        p_out   = self.decoder_pressure(x)  # [B, 1, 100, 100]

        out = torch.cat([vel_out, p_out], dim=1)  # [B, 3, 100, 100]
        return out.permute(0, 2, 3, 1)  # [B, 100, 100, 3]
