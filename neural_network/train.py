import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ReFlowNet
from flow_dataset import FlowDataset

# Configurações
BATCH_SIZE = 2
EPOCHS = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dados com divisão fixa (T/V/I)
train_set = FlowDataset(subset='train')
val_set   = FlowDataset(subset='val')

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Modelo e otimizador
model = ReFlowNet().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Loop de treino
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validação
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Época {epoch+1:03d} | Treino: {train_loss:.6f} | Validação: {val_loss:.6f}")

    # Salvar melhor modelo
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'neural_network/best_model.pt')
        print("📦 Modelo salvo!")


# Após o treino: visualização final da melhor predição na validação
import matplotlib.pyplot as plt
import os

# Garante que o melhor modelo está carregado
model.load_state_dict(torch.load('neural_network/best_model.pt', map_location=DEVICE))
model.eval()

# Seleciona primeiro exemplo da validação
sample_Re, sample_true = val_set[0]
sample_Re = sample_Re.unsqueeze(0).to(DEVICE)
sample_true = sample_true.numpy()

with torch.no_grad():
    sample_pred = model(sample_Re).squeeze(0).cpu().numpy()

var_names = ['Ux', 'Uy', 'p']
os.makedirs("validacao_final", exist_ok=True)

for j in range(3):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(sample_true[:, :, j], cmap='jet')
    axs[0].set_title(f'{var_names[j]} real (validação)')
    axs[1].imshow(sample_pred[:, :, j], cmap='jet')
    axs[1].set_title(f'{var_names[j]} predito (validação)')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'validacao_final/{var_names[j]}_final.png')
    plt.close()
