import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker

# Garantir saída visual
os.makedirs("../data/verificacao_normalizacao", exist_ok=True)

# Carregar os CSVs
raw_df  = pd.read_csv("../data/dataY-raw.csv", na_values=[""])
dim_df  = pd.read_csv("../data/dataY-dimensionless.csv", na_values=[""])
norm_df = pd.read_csv("../data/dataY-normalized.csv", na_values=[""])


import glob

# Descobrir os valores de Re a partir dos diretórios
pattern = "../cases/Re_*/postProcessing/cloud/1/ref_point_p_U.xy"
paths = sorted(glob.glob(pattern))

# Extração dos Reynolds reais
reynolds_numbers = []
for path in paths:
    folder = path.split("/postProcessing")[0]
    re_str = folder.split("_")[-1]
    Re = float(re_str)
    reynolds_numbers.append(Re)


# Cada campo tem 100x100 pontos × 12 amostras → 120000 linhas por CSV
n_cases = 12
grid_size = 100

# Nomes das variáveis e colunas
cols = ['Ux', 'Uy', 'p']

# Função para aplicar colormap com melhor contraste local
def plot_with_colorbar(ax, data, title):
    mean = np.nanmean(data)
    std = np.nanstd(data)
    vmin = mean - 3 * std
    vmax = mean + 3 * std
    
    im = ax.imshow(data, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, format='%.2g')

# Geração dos gráficos para cada uma das 12 simulações
for sample_idx in range(n_cases):
    offset = sample_idx * grid_size * grid_size
    end = offset + grid_size * grid_size

    raw_data = raw_df.iloc[offset:end].values.reshape((grid_size, grid_size, 3))
    dim_data = dim_df.iloc[offset:end].values.reshape((grid_size, grid_size, 3))
    norm_data = norm_df.iloc[offset:end].values.reshape((grid_size, grid_size, 3))

    for i, var in enumerate(cols):
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        plot_with_colorbar(axs[0], raw_data[:, :, i], f'{var} - Original | Re = {reynolds_numbers[sample_idx]:.2f}')
        plot_with_colorbar(axs[1], dim_data[:, :, i], f'{var} - Adimensional | Re = {reynolds_numbers[sample_idx]:.2f}')
        plot_with_colorbar(axs[2], norm_data[:, :, i], f'{var} - Normalizado | Re = {reynolds_numbers[sample_idx]:.2f}')

        plt.tight_layout()
        plt.savefig(f'../data/verificacao_normalizacao/{var}_sample_{sample_idx+1}.png')
        plt.close()

        # Exibir os limites de normalização e adimensionalização para todos os casos, ignorando NaNs
        raw_min = np.nanmin(raw_data[:, :, i])  # Ignora NaN
        raw_max = np.nanmax(raw_data[:, :, i])  # Ignora NaN
        dim_min = np.nanmin(dim_data[:, :, i])  # Ignora NaN
        dim_max = np.nanmax(dim_data[:, :, i])  # Ignora NaN
        norm_min = np.nanmin(norm_data[:, :, i])  # Ignora NaN
        norm_max = np.nanmax(norm_data[:, :, i])  # Ignora NaN

        # Exibir os limites no console em formato .3e
        print(f"Limites para {var} (Re = {reynolds_numbers[sample_idx]:.2f}):")
        print(f"  Original:   min = {raw_min:.3e}, max = {raw_max:.3e}")
        print(f"  Adimensional: min = {dim_min:.3f}, max = {dim_max:.3f}")
        print(f"  Normalizado:  min = {norm_min:.3f}, max = {norm_max:.3f}")
        print("-" * 50)

print(f"✅ Gráficos e limites salvos em '../data/verificacao_normalizacao/' para todas as amostras.")
