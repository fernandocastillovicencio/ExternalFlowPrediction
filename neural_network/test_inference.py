import torch
import matplotlib.pyplot as plt
import os
from model import ReFlowNet
from flow_dataset import FlowDataset
import numpy as np

# Cria diretórios de saída, se ainda não existem
os.makedirs("comparacoes_com_erro", exist_ok=True)
os.makedirs("inferencias", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega o modelo treinado
model = ReFlowNet().to(DEVICE)
model.load_state_dict(torch.load("neural_network/best_model.pt", map_location=DEVICE))
model.eval()

# Dados de inferência (usando o conjunto de teste)
test_dataset = FlowDataset(subset='test')
var_names = ['Ux', 'Uy', 'p']

for i in range(len(test_dataset)):
    # Agora o dataset retorna (Re, true_field, mask)
    Re, true_field, mask = test_dataset[i]
    Re_val = Re.item()  # Valor real de Reynolds da amostra
    Re_input = Re.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_field = model(Re_input).squeeze(0).cpu().numpy()

    # Converter para numpy
    true_field = true_field.numpy()
    mask = mask.numpy()
    # Onde a máscara for False, preserva NaN
    true_field[~mask] = np.nan

    # Campo de erro: calculado somente nos pontos válidos
    error_field = np.full_like(pred_field, np.nan)
    error_field[mask] = pred_field[mask] - true_field[mask]

    # Para cada variável, imprimir estatísticas e salvar gráficos
    for j, var in enumerate(var_names):
        print(f'\n📌 Re = {Re_val:.2f} | Variável: {var}')
        print(f'  ▸ Real     → min: {np.nanmin(true_field[:, :, j]):.4f}, max: {np.nanmax(true_field[:, :, j]):.4f}')
        print(f'  ▸ Predito  → min: {np.nanmin(pred_field[:, :, j]):.4f}, max: {np.nanmax(pred_field[:, :, j]):.4f}')
        erro = error_field[:, :, j]
        print(f'  ▸ Erro     → min: {np.nanmin(erro):.4f}, max: {np.nanmax(erro):.4f}, máx(abs): {np.nanmax(np.abs(erro)):.4f}')

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
        # Gráfico 1: Campo real
        vmin = np.nanmin(true_field[:, :, j])
        vmax = np.nanmax(true_field[:, :, j])
        im0 = axs[0].imshow(true_field[:, :, j], cmap='jet', vmin=vmin, vmax=vmax)
        axs[0].set_title(f'{var} - Real | Re = {Re_val:.2f}')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
        # Gráfico 2: Campo predito
        im1 = axs[1].imshow(pred_field[:, :, j], cmap='jet', vmin=vmin, vmax=vmax)
        axs[1].set_title(f'{var} - Predito')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
        # Gráfico 3: Erro com máscara
        lim = np.nanmax(np.abs(error_field[:, :, j]))
        im2 = axs[2].imshow(error_field[:, :, j], cmap='jet', vmin=-lim, vmax=lim)
        axs[2].set_title(f'{var} - Erro')
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    
        for ax in axs:
            ax.axis('off')
    
        plt.tight_layout()
        plt.savefig(f"inferencias/{var}_Re_{i:02d}_{Re_val:.2f}.png")
        plt.close()
