import torch
from torch.utils.data import Dataset
import numpy as np
import os

# Índices conforme a tabela do usuário
TRAIN_IDX = [1, 2, 3, 5, 6, 8, 9, 10]
VAL_IDX = [4, 7]
TEST_IDX = [0, 11]


class FlowDataset(Dataset):
    def __init__(
        self,
        subset="all",
        path_x="data/dataX-log-normalized.npy",
        path_y="data/dataY-normalized.npy",
        path_mask="data/mask.npy",
    ):

        x_raw = np.load(path_x)  # shape [N, 1]

        # Expande o escalar Re para um canal 2D [H, W]
        y = np.load(path_y)  # shape [N, H, W, 3]
        H, W = y.shape[1:3]
        re_maps = np.repeat(x_raw[:, :, np.newaxis, np.newaxis], H, axis=2)
        re_maps = np.repeat(re_maps, W, axis=3)  # [N, 1, H, W]

        # Aqui você pode adicionar outros canais auxiliares se desejar
        # Carrega flow_mask (assume arquivo: data/flow_mask.npy)
        flow_mask = np.load("data/flow_mask.npy")  # [N, H, W]
        flow_mask = flow_mask[:, np.newaxis, :, :]  # [N, 1, H, W]

        if subset == "train":
            idx = TRAIN_IDX
        elif subset == "val":
            idx = VAL_IDX
        elif subset == "test":
            idx = TEST_IDX
        else:
            idx = list(range(len(x)))

        # Carrega a máscara de regiões categóricas (1 = livre, 2 = frontal, 3 = esteira)
        region_path = "data/region_mask_tensor.npy"
        if os.path.exists(region_path):
            region = np.load(region_path)  # shape [N, H, W]
            self.region = torch.tensor(region[idx], dtype=torch.long)  # [N, H, W]
        else:
            raise FileNotFoundError("Arquivo region_mask_tensor.npy não encontrado.")

        # Concatena Re_map e flow_mask → [N, 2, H, W]
        x = np.concatenate([re_maps, flow_mask], axis=1)

        if np.isnan(y).any():
            print("⚠️ y contém NaNs (esperado nas regiões internas do obstáculo).")

        self.x = torch.tensor(x[idx], dtype=torch.float32)  # entrada [C_in, H, W]
        self.y_raw = torch.tensor(
            y[idx], dtype=torch.float32
        )  # usada para visualização
        self.y = torch.tensor(
            np.nan_to_num(y[idx], nan=0.0), dtype=torch.float32
        )  # usada no treino

        if os.path.exists(path_mask):
            mask = np.load(path_mask)

            if mask.ndim == 2:
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # [H, W, 3]
                mask = np.repeat(
                    mask[np.newaxis, ...], y.shape[0], axis=0
                )  # [N, H, W, 3]

            if mask.shape != y.shape:
                raise ValueError(
                    f"Máscara com shape incompatível: {mask.shape} != {y.shape}"
                )

            self.mask = torch.tensor(mask[idx], dtype=torch.bool)
        else:
            self.mask = torch.ones_like(self.y, dtype=torch.bool)

        self.height = self.y.shape[1]
        self.width = self.y.shape[2]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self.mask[idx],
            self.y_raw[idx],
            self.region[idx],
        )

    def get_domain_mask(self):
        return self.mask
