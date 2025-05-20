import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib import cm
import matplotlib as mpl

# Configuração global para visualização
mpl.rcParams["image.interpolation"] = "none"
mpl.rcParams["image.cmap"] = "jet"


def plot_prediction_fields(
    model,
    val_loader,
    device,
    domain_mask,
    var_names=["Ux", "Uy", "p"],
    save_dir="neural_network/fig/validation",
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    xb, yb, _, yb_raw, _ = next(iter(val_loader))

    xb = xb.to(device)

    with torch.no_grad():
        pred = model(xb)[0].detach().cpu().numpy().copy()

    true = yb_raw[0].numpy().copy()
    flow_mask = np.load("data/flow_mask.npy")[0]  # shape: [H, W]
    mask2d = flow_mask.astype(bool)

    # Garante que pred seja uma cópia independente
    pred = model(xb)[0].detach().cpu().numpy().copy()

    error = np.abs(pred - true)

    for j in range(3):
        true_plot = true[:, :, j].copy()
        pred_plot = pred[:, :, j].copy()
        error_plot = error[:, :, j].copy()

        true_plot[~mask2d] = np.nan
        pred_plot = np.where(mask2d, pred_plot, np.nan)

        error_plot[~mask2d] = np.nan

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        cmap = cm.get_cmap("jet").copy()
        cmap.set_bad(color="white")

        im0 = axs[0].imshow(true_plot, cmap=cmap)
        axs[0].set_title(f"{var_names[j]} Real")
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(pred_plot, cmap=cmap)
        axs[1].set_title(f"{var_names[j]} Predito")
        plt.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(error_plot, cmap=cmap)
        axs[2].set_title("Erro Absoluto")
        plt.colorbar(im2, ax=axs[2])

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/fig_{var_names[j]}.png")
        plt.close()
