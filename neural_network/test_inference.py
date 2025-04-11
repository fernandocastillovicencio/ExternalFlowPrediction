import torch
import matplotlib.pyplot as plt
import os
from model import ReFlowUNet
from flow_dataset import FlowDataset
import numpy as np

# Garante que o diretório de saída exista
os.makedirs("comparacoes_com_erro", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega o modelo
model = ReFlowUNet().to(DEVICE)
model.load_state_dict(torch.load("neural_network/best_model.pt", map_location=DEVICE))
model.eval()

# Dados de inferência (fora da faixa de treino)
test_dataset = FlowDataset(subset='test')
var_names = ['Ux', 'Uy', 'p']

for i in range(len(test_dataset)):
    Re, true_field = test_dataset[i]
    Re_val = Re.item()
    Re_input = Re.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_field = model(Re_input).squeeze(0).cpu().numpy()

    true_field = true_field.numpy()
    error_field = np.abs(true_field - pred_field)

    for j in range(3):
        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        
        # Campo real
        im0 = axs[0].imshow(true_field[:, :, j], cmap='jet')
        axs[0].set_title(f'{var_names[j]} real | Re = {Re_val:.2f}')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        # Campo predito
        im1 = axs[1].imshow(pred_field[:, :, j], cmap='jet')
        axs[1].set_title(f'{var_names[j]} predito')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # Erro absoluto
        im2 = axs[2].imshow(error_field[:, :, j], cmap='jet')
        axs[2].set_title(f'Erro absoluto')
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'comparacoes_com_erro/{var_names[j]}_Re_{Re_val:.2f}.png')
        plt.close()
