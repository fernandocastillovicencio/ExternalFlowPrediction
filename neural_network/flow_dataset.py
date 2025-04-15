import torch
from torch.utils.data import Dataset
import numpy as np
import os

# Índices conforme a tabela do usuário
TRAIN_IDX = [1, 2, 3, 5, 6, 8, 9, 10]
VAL_IDX   = [4, 7]
TEST_IDX  = [0, 11]

class FlowDataset(Dataset):
    def get_domain_mask(self):
        return self.mask


    def __init__(self, subset='all',
                 path_x='data/dataX.npy',
                 path_y='data/dataY-normalized.npy',
                 path_mask='data/mask.npy'):

        x = np.load(path_x)
        y = np.load(path_y)

        if subset == 'train':
            idx = TRAIN_IDX
        elif subset == 'val':
            idx = VAL_IDX
        elif subset == 'test':
            idx = TEST_IDX
        else:
            idx = list(range(len(x)))

        # Verifica se há NaNs
        if np.isnan(y).any():
            print("⚠️ y contém NaNs (esperado nas regiões internas do obstáculo).")

        # Converte para tensor
        self.x = torch.tensor(x[idx], dtype=torch.float32)
        self.y = torch.tensor(np.nan_to_num(y[idx], nan=0.0), dtype=torch.float32)  # evita passar NaN ao modelo

        # Máscara para a perda
        if os.path.exists(path_mask):
            mask2d = np.load(path_mask)  # (100, 100)
            assert mask2d.shape == y.shape[1:3], f"Máscara 2D com shape incompatível: {mask2d.shape} != {y.shape[1:3]}"

            # Expande para shape (100, 100, 3)
            mask3d = np.repeat(mask2d[:, :, np.newaxis], 3, axis=2)

            # Replica para todas as amostras
            mask_full = np.repeat(mask3d[np.newaxis, :, :, :], y.shape[0], axis=0)  # (N, 100, 100, 3)

            self.mask = torch.tensor(mask_full[idx], dtype=torch.bool)

        else:
            self.mask = torch.ones_like(self.y, dtype=torch.bool)




    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]




if __name__ == '__main__':
    dataset = FlowDataset()
    print(f'Tamanho do dataset: {len(dataset)} amostras')
    x0, y0, m0 = dataset[0]
    print(f'\nPrimeira entrada (Re): {x0}')
    print(f'Saída correspondente - shape: {y0.shape}')
    print(f'Máscara correspondente: {m0.shape}')