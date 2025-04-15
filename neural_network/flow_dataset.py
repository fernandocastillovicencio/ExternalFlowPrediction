import torch
from torch.utils.data import Dataset
import numpy as np
import os

# Índices conforme a tabela do usuário
TRAIN_IDX = [1, 2, 3, 5, 6, 8, 9, 10]
VAL_IDX   = [4, 7]
TEST_IDX  = [0, 11]

class FlowDataset(Dataset):
    def __init__(self, subset='all',
                 path_x='data/dataX.npy',
                 path_y='data/dataY-normalized.npy',
                 path_mask='data/mask.npy'):

        x = np.load(path_x)  # shape [N, 5, H, W]
        y = np.load(path_y)  # shape [N, H, W, 3]

        if subset == 'train':
            idx = TRAIN_IDX
        elif subset == 'val':
            idx = VAL_IDX
        elif subset == 'test':
            idx = TEST_IDX
        else:
            idx = list(range(len(x)))

        if np.isnan(y).any():
            print("⚠️ y contém NaNs (esperado nas regiões internas do obstáculo).")

        self.x = torch.tensor(x[idx], dtype=torch.float32)  # entrada [C_in, H, W]
        self.y_raw = torch.tensor(y[idx], dtype=torch.float32)  # usada para visualização
        self.y = torch.tensor(np.nan_to_num(y[idx], nan=0.0), dtype=torch.float32)  # usada no treino

        if os.path.exists(path_mask):
            mask = np.load(path_mask)

            if mask.ndim == 2:
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # [H, W, 3]
                mask = np.repeat(mask[np.newaxis, ...], y.shape[0], axis=0)  # [N, H, W, 3]

            if mask.shape != y.shape:
                raise ValueError(f"Máscara com shape incompatível: {mask.shape} != {y.shape}")

            self.mask = torch.tensor(mask[idx], dtype=torch.bool)
        else:
            self.mask = torch.ones_like(self.y, dtype=torch.bool)

        self.height = self.y.shape[1]
        self.width  = self.y.shape[2]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx], self.y_raw[idx]

    def get_domain_mask(self):
        return self.mask
