import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Garantir saída visual
os.makedirs("../data/verificacao_normalizacao", exist_ok=True)

# Carregar os CSVs
raw_df = pd.read_csv("../data/dataY-raw.csv")
dim_df = pd.read_csv("../data/dataY-dimensionless.csv")
norm_df = pd.read_csv("../data/dataY-normalized.csv")

# Cada campo tem 100x100 pontos × 12 amostras → 120000 linhas por CSV
n_cases = 12
grid_size = 100

# Nomes das variáveis e colunas
cols = ['Ux', 'Uy', 'p']
norm_cols = ['Ux_n', 'Uy_n', 'p_n']

# Geração dos gráficos para cada uma das 12 simulações
for sample_idx in range(n_cases):
    offset = sample_idx * grid_size * grid_size
    end = offset + grid_size * grid_size

    # Recorta os dados da amostra
    raw_data = raw_df.iloc[offset:end].values.reshape((grid_size, grid_size, 3))
    dim_data = dim_df.iloc[offset:end].values.reshape((grid_size, grid_size, 3))
    norm_data = norm_df.iloc[offset:end].values.reshape((grid_size, grid_size, 3))

    # Geração dos gráficos para cada variável
    for i, var in enumerate(cols):
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axs[0].imshow(raw_data[:, :, i], cmap='jet')
        axs[0].set_title(f'{var} - Original | Re = {sample_idx+1}')
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(dim_data[:, :, i], cmap='jet')
        axs[1].set_title(f'{var} - Adimensional | Re = {sample_idx+1}')
        plt.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(norm_data[:, :, i], cmap='jet')
        axs[2].set_title(f'{var} - Normalizado | Re = {sample_idx+1}')
        plt.colorbar(im2, ax=axs[2])

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'../data/verificacao_normalizacao/{var}_sample_{sample_idx+1}.png')
        plt.close()

        # Exibir os limites de normalização e adimensionalização para todos os casos
        raw_min = raw_data[:, :, i].min()
        raw_max = raw_data[:, :, i].max()
        dim_min = dim_data[:, :, i].min()
        dim_max = dim_data[:, :, i].max()
        norm_min = norm_data[:, :, i].min()
        norm_max = norm_data[:, :, i].max()

        # Exibir os limites no console em formato .3e
        print(f"Limites para {var} (Re = {sample_idx+1}):")
        print(f"  Original:   min = {raw_min:.3e}, max = {raw_max:.3e}")
        print(f"  Adimensional: min = {dim_min:.3e}, max = {dim_max:.3e}")
        print(f"  Normalizado:  min = {norm_min:.3e}, max = {norm_max:.3e}")
        print("-" * 50)

print(f"✅ Gráficos e limites salvos em '../data/verificacao_normalizacao/' para todas as amostras.")
