import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ReFlowNet
from flow_dataset import FlowDataset
from loss_plot import plot_loss_curve
from visualize_fields import plot_prediction_fields

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparâmetros
BATCH_SIZE = 2
EPOCHS = 200
LR = 1e-3
LAMBDA_NS = 0.5  # Peso da penalização física
MODEL_PATH = "neural_network/best_model.pt"

# Dados
train_set = FlowDataset(subset='train')
val_set = FlowDataset(subset='val')
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

model = ReFlowNet(in_channels=5).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses = [], []
best_val_loss = float("inf")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def compute_NS_loss(preds, coords, Re):
    """
    Calcula o resíduo das equações de Navier-Stokes 2D adimensionais.
    """
    preds = preds.permute(0, 3, 1, 2).clone().detach().requires_grad_(True)

    u = preds[:, 0:1, :, :]
    v = preds[:, 1:2, :, :]
    p = preds[:, 2:3, :, :]

    x = coords[:, 0:1, :, :].detach()
    y = coords[:, 1:2, :, :].detach()

    grads = lambda f, wrt: torch.autograd.grad(f, wrt, grad_outputs=torch.ones_like(f),
                                               create_graph=True, retain_graph=True)[0]

    # Primeiras derivadas
    du_dx = grads(u, x)
    du_dy = grads(u, y)
    dv_dx = grads(v, x)
    dv_dy = grads(v, y)
    dp_dx = grads(p, x)
    dp_dy = grads(p, y)

    # Segundas derivadas (viscosidade)
    d2u_dx2 = grads(du_dx, x)
    d2u_dy2 = grads(du_dy, y)
    d2v_dx2 = grads(dv_dx, x)
    d2v_dy2 = grads(dv_dy, y)

    # Resíduos
    continuity = du_dx + dv_dy
    momentum_x = u * du_dx + v * du_dy + dp_dx - (1/Re) * (d2u_dx2 + d2u_dy2)
    momentum_y = u * dv_dx + v * dv_dy + dp_dy - (1/Re) * (d2v_dx2 + d2v_dy2)

    loss_c = torch.mean(continuity ** 2)
    loss_x = torch.mean(momentum_x ** 2)
    loss_y = torch.mean(momentum_y ** 2)

    return loss_c + loss_x + loss_y

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for xb, yb, mb, coords in train_loader:
        xb, yb, mb, coords = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE), coords.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)

        # Loss de dados
        loss_data = (
            criterion(preds[..., 0][mb[..., 0]], yb[..., 0][mb[..., 0]]) +
            criterion(preds[..., 1][mb[..., 1]], yb[..., 1][mb[..., 1]]) +
            criterion(preds[..., 2][mb[..., 2]], yb[..., 2][mb[..., 2]])
        )

        # Re recuperado da entrada (canal 0)
        Re_val = xb[:, 0, 0, 0].mean()
        loss_phys = compute_NS_loss(preds, coords, Re_val)

        # Loss total
        loss = loss_data + LAMBDA_NS * loss_phys
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    # Validação padrão
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb, mb, _ in val_loader:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            preds = model(xb)
            val_loss += (
                criterion(preds[..., 0][mb[..., 0]], yb[..., 0][mb[..., 0]]) +
                criterion(preds[..., 1][mb[..., 1]], yb[..., 1][mb[..., 1]]) +
                criterion(preds[..., 2][mb[..., 2]], yb[..., 2][mb[..., 2]])
            ).item() * xb.size(0)

    val_loss /= len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✔️ Novo melhor modelo salvo: {MODEL_PATH}")

# Pós-treino
plot_loss_curve(train_losses, val_losses)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
domain_mask = val_set.get_domain_mask()
plot_prediction_fields(model, val_loader, DEVICE, domain_mask)
