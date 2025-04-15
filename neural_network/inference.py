import torch
import os
from model import ReFlowNet
from flow_dataset import FlowDataset
from inference_plot import plot_inference_result

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("neural_network/fig/inferences", exist_ok=True)

# Carrega o modelo treinado
model = ReFlowNet().to(DEVICE)
model.load_state_dict(torch.load("neural_network/best_model.pt", map_location=DEVICE))
model.eval()

# Dados de inferência
test_dataset = FlowDataset(subset='test')
var_names = ['Ux', 'Uy', 'p']

for i in range(len(test_dataset)):
    Re, true_field, mask = test_dataset[i]
    Re_input = Re.unsqueeze(0).to(DEVICE)
    Re_val = Re.item()

    with torch.no_grad():
        pred_field = model(Re_input).squeeze(0).cpu().numpy()

    plot_inference_result(true_field.numpy(), pred_field, mask.numpy(), var_names, Re_val, i, "neural_network/fig/inferences")