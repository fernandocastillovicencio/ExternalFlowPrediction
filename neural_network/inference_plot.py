import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def plot_inference_result(true_field, pred_field, mask, var_names, Re_val, idx, save_dir):
    true_field = np.where(mask, true_field, np.nan)
    pred_field = np.where(mask, pred_field, np.nan)
    error_field = np.full_like(true_field, np.nan)
    error_field[mask] = pred_field[mask] - true_field[mask]

    cmap = cm.get_cmap('jet').copy()
    cmap.set_bad(color='white')

    for j, var in enumerate(var_names):
        erro = error_field[:, :, j]
        vmin = np.nanmin(true_field[:, :, j])
        vmax = np.nanmax(true_field[:, :, j])
        lim = np.nanmax(np.abs(erro))

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axs[0].imshow(true_field[:, :, j], cmap=cmap, vmin=vmin, vmax=vmax)
        axs[0].set_title(f'{var} - Real | Re = {Re_val:.2f}')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        im1 = axs[1].imshow(pred_field[:, :, j], cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].set_title(f'{var} - Predito')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        im2 = axs[2].imshow(erro, cmap=cmap, vmin=-lim, vmax=lim)
        axs[2].set_title(f'{var} - Erro')
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{var}_Re_{idx:02d}_{Re_val:.2f}.png")
        plt.close()