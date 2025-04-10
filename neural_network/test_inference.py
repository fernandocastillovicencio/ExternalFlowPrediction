import torch
import matplotlib.pyplot as plt
import os
from model import ReFlowNet
from flow_dataset import FlowDataset
import numpy as np

# Garante que o diretório de saída exista
os.makedirs("comparacoes_com_erro", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega o modelo
model = ReFlowNet().to(DEVICE)
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

    # Um gráfico 3-em-1 para cada variável
    for j in range(3):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        axs[0].imshow(true_field[:, :, j], cmap='jet')
        axs[0].set_title(f'{var_names[j]} real | Re = {Re_val:.2f}')
        axs[1].imshow(pred_field[:, :, j], cmap='jet')
        axs[1].set_title(f'{var_names[j]} predito')
        axs[2].imshow(error_field[:, :, j], cmap='jet')
        axs[2].set_title(f'Erro absoluto')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'comparacoes_com_erro/{var_names[j]}_Re_{Re_val:.2f}.png')
        plt.close()
