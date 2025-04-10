import torch
from torch.utils.data import DataLoader
from flow_dataset import FlowDataset

# 1. Carrega os conjuntos conforme os índices fixos
train_set = FlowDataset(subset='train')
val_set   = FlowDataset(subset='val')
test_set  = FlowDataset(subset='test')  # opcional para inferência

# 2. Cria DataLoaders
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=2, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=1, shuffle=False)

# Teste rápido
if __name__ == '__main__':
    for xb, yb in train_loader:
        print(f'[Treino] Batch entrada: {xb.shape} | Batch saída: {yb.shape}')
        break
    for xb, yb in val_loader:
        print(f'[Validação] Batch entrada: {xb.shape} | Batch saída: {yb.shape}')
        break
    for xb, yb in test_loader:
        print(f'[Inferência] Entrada: {xb} | shape saída: {yb.shape}')
        break


# Teste rápido
if __name__ == '__main__':
    for xb, yb in train_loader:
        print(f'Batch entrada: {xb.shape} | Batch saída: {yb.shape}')
        break