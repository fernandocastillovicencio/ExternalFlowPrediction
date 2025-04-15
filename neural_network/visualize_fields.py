import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib import cm
import matplotlib as mpl

# Configuração global para visualização
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap'] = 'jet'

def plot_prediction_fields(model, val_loader, device, domain_mask, var_names=['Ux', 'Uy', 'p'], save_dir="neural_network/fig/validation"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    xb, yb, _, yb_raw= next(iter(val_loader))
    xb = xb.to(device)

    with torch.no_grad():
        pred = model(xb)[0].cpu().numpy()

    true = yb_raw[0].numpy()
    mask = domain_mask[0].numpy().astype(bool)  # Shape: (100, 100, 3)

    error = np.abs(pred - true)

    # Aplicar NaN nas regiões internas do obstáculo (False na máscara)
    true[~mask] = np.nan
    pred[~mask] = np.nan
    error[~mask] = np.nan

    for j in range(3):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        # Colormap com NaNs em branco
        cmap = cm.get_cmap('jet').copy()
        cmap.set_bad(color='white')

        # Real
        im0 = axs[0].imshow(true[:, :, j], cmap=cmap)
        axs[0].set_title(f"{var_names[j]} Real")
        plt.colorbar(im0, ax=axs[0])

        # Predito
        im1 = axs[1].imshow(pred[:, :, j], cmap=cmap)
        axs[1].set_title(f"{var_names[j]} Predito")
        plt.colorbar(im1, ax=axs[1])

        # Erro
        im2 = axs[2].imshow(error[:, :, j], cmap=cmap)
        axs[2].set_title("Erro Absoluto")
        plt.colorbar(im2, ax=axs[2])

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/fig_{var_names[j]}.png")
        plt.close()