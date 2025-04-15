import torch
import os
from model import ReFlowNet
from flow_dataset import FlowDataset
from inference_plot import plot_inference_result

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("neural_network/fig/inferences", exist_ok=True)

# Carrega o modelo treinado (nova arquitetura com 2 cabeças)
model = ReFlowNet(in_channels=5).to(DEVICE)
model.load_state_dict(torch.load("neural_network/best_model.pt", map_location=DEVICE))
model.eval()

# Dados de inferência
test_dataset = FlowDataset(subset='test')
var_names = ['Ux', 'Uy', 'p']

for i in range(len(test_dataset)):
    x_input, y_true, mask, _ = test_dataset[i]           # x_input: [5, H, W]
    x_input = x_input.unsqueeze(0).to(DEVICE)            # [1, 5, H, W]
    Re_val = x_input[0, 0, 0, 0].item()                   # canal 0 = Re_map

    with torch.no_grad():
        pred_field = model(x_input).squeeze(0).cpu().numpy()  # [H, W, 3]

    plot_inference_result(y_true.numpy(), pred_field, mask.numpy(), var_names, Re_val, i, "neural_network/fig/inferences")
