import torch
from torch.utils.data import DataLoader
from flow_dataset import FlowDataset

# 1. Carrega os conjuntos conforme os índices fixos
train_set = FlowDataset(subset="train")
val_set = FlowDataset(subset="val")
test_set = FlowDataset(subset="test")  # opcional para inferência

# 2. Cria DataLoaders
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
val_loader = DataLoader(val_set, batch_size=2, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# Teste rápido
if __name__ == "__main__":
    for xb, yb, mb, _, _ in train_loader:
        print(f"[Treino] Entrada: {xb.shape} | Saída: {yb.shape} | Máscara: {mb.shape}")
        break

    for xb, yb, mb, _, _ in val_loader:
        print(
            f"[Validação] Entrada: {xb.shape} | Saída: {yb.shape} | Máscara: {mb.shape}"
        )
        break

    for xb, yb, mb, _, _ in test_loader:
        print(
            f"[Inferência] Entrada: {xb.shape} | Saída: {yb.shape} | Máscara: {mb.shape}"
        )
        break
