import torch
from torch.utils.data import Dataset
import numpy as np
import os

# Índices conforme a tabela do usuário
TRAIN_IDX = [1, 2, 3, 5, 6, 8, 9, 10]
VAL_IDX   = [4, 7]
TEST_IDX  = [0, 11]

class FlowDataset(Dataset):
    def __init__(self, subset='all', path_x='data/dataX.npy', path_y='data/dataY-normalized.npy'):
        if not os.path.exists(path_x):
            raise FileNotFoundError(f"Arquivo não encontrado: {path_x}")
        if not os.path.exists(path_y):
            raise FileNotFoundError(f"Arquivo não encontrado: {path_y}")

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

        self.x = torch.tensor(x[idx], dtype=torch.float32)
        self.y = torch.tensor(y[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



if __name__ == '__main__':
    dataset = FlowDataset()
    print(f'Tamanho do dataset: {len(dataset)} amostras')
    x0, y0 = dataset[0]
    print(f'\nPrimeira entrada (Re): {x0}')
    print(f'Saída correspondente - shape: {y0.shape}')
    print(f'Variáveis mín e máx: {y0.min():.4f} → {y0.max():.4f}')
