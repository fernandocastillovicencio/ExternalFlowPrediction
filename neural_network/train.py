import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib import cm
import matplotlib as mpl
from flow_dataset import FlowDataset
from model import ReFlowNet
from loss_plot import plot_loss_curve
from visualize_fields import plot_prediction_fields
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Configuração global para visualização
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap'] = 'jet'

# Ajuste de diretório raiz do projeto
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Parâmetros
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS = 200
LR = 1e-3
MODEL_PATH = "neural_network/best_model.pt"

# Pesos para cada variável
W_UX = 1.0
W_UY = 1.0
W_P = 3.0

# Datasets e Loaders
train_set = FlowDataset(subset='train')
val_set = FlowDataset(subset='val')

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Modelo
model = ReFlowNet(in_channels=3, out_channels=3).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Treinamento
train_losses = []
val_losses = []
best_val_loss = float("inf")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for xb, yb, mb,_ in train_loader:
        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        if mb.sum() == 0:
            continue
        loss = (
            W_UX * criterion(preds[..., 0][mb[..., 0]], yb[..., 0][mb[..., 0]]) +
            W_UY * criterion(preds[..., 1][mb[..., 1]], yb[..., 1][mb[..., 1]]) +
            W_P  * criterion(preds[..., 2][mb[..., 2]], yb[..., 2][mb[..., 2]])
        )
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb, mb, _ in val_loader:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            preds = model(xb)
            if mb.sum() == 0:
                continue
            loss = (
                W_UX * criterion(preds[..., 0][mb[..., 0]], yb[..., 0][mb[..., 0]]) +
                W_UY * criterion(preds[..., 1][mb[..., 1]], yb[..., 1][mb[..., 1]]) +
                W_P  * criterion(preds[..., 2][mb[..., 2]], yb[..., 2][mb[..., 2]])
            )
            val_loss += loss.item() * xb.size(0)

    val_loss /= len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✔️ Novo melhor modelo salvo: {MODEL_PATH}")

print("\n🌝 Treinamento finalizado.")

# Plota a curva de perda
plot_loss_curve(train_losses, val_losses)

# Visualização dos campos com a máscara original do domínio
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
domain_mask = val_set.get_domain_mask()
plot_prediction_fields(model, val_loader, DEVICE, domain_mask)