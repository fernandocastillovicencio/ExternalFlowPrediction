import torch
import os
import numpy as np
from model import ReFlowNet
from flow_dataset import FlowDataset
from inference_plot import plot_inference_result

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("neural_network/fig/inferences", exist_ok=True)

# Carrega o modelo treinado
model = ReFlowNet(in_channels=2).to(DEVICE)
model.load_state_dict(torch.load("neural_network/best_model.pt", map_location=DEVICE))
model.eval()

# Dados de inferência
test_dataset = FlowDataset(subset="test")
var_names = ["Ux", "Uy", "p"]

# Carrega a flow_mask completa (1 = fluido, 0 = obstáculo)
flow_mask_all = np.load("data/flow_mask.npy")  # shape: [N, H, W]

for i in range(len(test_dataset)):
    x_input, y_true, _, _, _ = test_dataset[i]
    x_input = x_input.unsqueeze(0).to(DEVICE)  # [1, 2, H, W]
    Re_val = x_input[0, 0, 0, 0].item()  # canal 0 = Re_map

    with torch.no_grad():
        pred_field = model(x_input).squeeze(0).cpu().numpy()  # [H, W, 3]

    # Recupera a flow_mask do caso i
    flow_mask = flow_mask_all[i]  # shape: [H, W], valores 1 (fluido) e 0 (obstáculo)

    # Passa a flow_mask como máscara para visualização
    plot_inference_result(
        y_true.numpy(),
        pred_field,
        flow_mask,
        var_names,
        Re_val,
        i,
        "neural_network/fig/inferences",
    )
