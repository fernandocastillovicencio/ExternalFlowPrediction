import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_inference_result(
    true_field, pred_field, mask, var_names, Re_val, idx, save_dir
):
    # Converte máscara para 2D, se vier como [H, W, 3]
    if mask.ndim == 3:
        mask2d = np.all(mask, axis=-1)
    else:
        mask2d = mask

    cmap = cm.get_cmap("jet").copy()
    cmap.set_bad(color="white")

    for j, var in enumerate(var_names):
        # Cópias seguras dos campos por canal
        true_j = true_field[:, :, j].copy()
        pred_j = pred_field[:, :, j].copy()

        # Aplica NaN nos campos real e predito
        true_j = np.where(mask2d, true_j, np.nan)
        pred_j = np.where(mask2d, pred_j, np.nan)

        # Calcula erro canal a canal com NaN no obstáculo
        error_j = pred_j - true_j
        error_j = np.where(mask2d, error_j, np.nan)

        # Escalas
        vmin = np.nanmin(true_j)
        vmax = np.nanmax(true_j)
        lim = np.nanmax(np.abs(error_j))

        # Plotagem
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axs[0].imshow(true_j, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[0].set_title(f"{var} - Real | Re = {Re_val:.2f}")
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        im1 = axs[1].imshow(pred_j, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].set_title(f"{var} - Predito")
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        im2 = axs[2].imshow(error_j, cmap=cmap, vmin=-lim, vmax=lim)
        axs[2].set_title(f"{var} - Erro")
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{var}_Re_{idx:02d}_{Re_val:.2f}.png")
        plt.close()
