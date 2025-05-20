import pathlib
import glob
import numpy as np
import json
from utils_normalization import load_domain_and_velocities

# -------------------------------------------------------- #
# STEP 1: Extrair configuração do domínio e Reynolds
# -------------------------------------------------------- #

print("\n🔵 ETAPA 1 — Lendo configuração da janela espacial...\n")

config_path = pathlib.Path("simulations/scripts/cloud_config.json")
velocities_path = pathlib.Path("simulations/scripts/velocities.json")

cfg = load_domain_and_velocities(config_path, velocities_path)

NX, NY = cfg["NX"], cfg["NY"]
XMIN, XMAX = cfg["XMIN"], cfg["XMAX"]
YMIN, YMAX = cfg["YMIN"], cfg["YMAX"]
velocities = cfg["velocities"]
sorted_reynolds = cfg["sorted_reynolds"]

# -------------------------------------------------------- #
# ETAPA 2: FLOW MASK                                        #
# -------------------------------------------------------- #
print("\n🔵 ETAPA 2 — Lendo arquivos *.xy e gerando flow_mask...\n")

pattern = "simulations/cases/Re_*/postProcessing/cloud/1/ref_point_p_U.xy"

xy_paths = sorted(glob.glob(pattern))
flow_masks = {}

for path in xy_paths:
    try:
        parts = pathlib.Path(path).parts
        re_folder = [p for p in parts if p.startswith("Re_")][0]
        Re = float(re_folder.replace("Re_", "").replace("_", "."))

        Uref = velocities[Re]
        data = np.genfromtxt(path)

        p = data[:, 3]
        Ux = data[:, 4]
        Uy = data[:, 5]

        # Define máscara: True onde é fluido (não obstáculo)
        flow_mask_flat = ~((p == 0.0) & (Ux == 0.0) & (Uy == 0.0))
        flow_mask_flat = flow_mask_flat.astype(np.uint8)

        assert (
            flow_mask_flat.shape[0] == NX * NY
        ), f"Shape inconsistente: {flow_mask_flat.shape[0]} != {NX*NY}"

        # Redimensiona para o grid 2D
        flow_mask = flow_mask_flat.reshape(NY, NX)
        flow_masks[Re] = flow_mask

        # ✅ Sucesso com contagem de pontos válidos e shape
        print(
            f"  ✅ Re {Re:.2f} → flow_mask gerada com {np.sum(flow_mask)} pontos válidos, shape = {flow_mask.shape}"
        )

    except Exception as e:
        print(f"  ⚠️  Falha ao processar {path} → {e}")
# -------------------------------------------------------- #
print("\n📦 Empilhando máscaras em tensor 3D [N, NY, NX]...\n")

flow_mask_tensor = np.stack([flow_masks[Re] for Re in sorted_reynolds], axis=0)

print(f"✅ flow_mask_tensor criado com shape: {flow_mask_tensor.shape}")

# -------------------------------------------------------- #
# ETAPA 3: REYNOLDS MASK                                   #
# -------------------------------------------------------- #
print("\n🔵 ETAPA 3 — Gerando reynolds_mask_raw e reynolds_mask_norm (em memória)...\n")

H, W = NY, NX
N = len(sorted_reynolds)

reynolds_mask_raw = np.zeros((N, H, W), dtype=np.float32)
reynolds_mask_norm = np.zeros((N, H, W), dtype=np.float32)

Re_min = min(velocities.keys())
Re_max = max(velocities.keys())

for i, Re in enumerate(sorted_reynolds):
    # Mapa bruto: todos os pontos com o valor de Re correspondente
    Re_raw_map = np.full((H, W), Re, dtype=np.float32)

    # Mapa normalizado: mesmo valor em todos os pontos, mas Re norm.
    Re_norm_value = (Re - Re_min) / (Re_max - Re_min)
    Re_norm_map = np.full((H, W), Re_norm_value, dtype=np.float32)

    reynolds_mask_raw[i] = Re_raw_map
    reynolds_mask_norm[i] = Re_norm_map

    print(f"  • Re = {Re:.2f} → raw = {Re:.2f} | norm = {Re_norm_value:.4f}")
# -------------------------------------------------------- #
print(f"📐 reynolds_mask_raw.shape  = {reynolds_mask_raw.shape}")
print(f"\n📐 reynolds_mask_norm.shape = {reynolds_mask_norm.shape}")
print("\n✅ Máscaras de Reynolds geradas e armazenadas em memória.")

# -------------------------------------------------------- #
# ETAPA 4: REGION MASK                                     #
# -------------------------------------------------------- #
print("\n🔵 ETAPA 4 — Gerando region_mask...\n")

region_masks = {}

for path in xy_paths:
    try:
        parts = pathlib.Path(path).parts
        re_folder = [p for p in parts if p.startswith("Re_")][0]
        Re = float(re_folder.replace("Re_", "").replace("_", "."))

        Uref = velocities[Re]
        data = np.genfromtxt(path)
        p = data[:, 3].astype(float)
        Ux = data[:, 4].astype(float)
        Uy = data[:, 5].astype(float)

        # Recupera a máscara de fluxo binária (1 para fluido, 0 para obstáculo)
        flow_mask = flow_masks[Re].reshape(-1)

        # Inicializa region_mask com 1s (fluido) e nan nos obstáculos
        region_mask = np.ones_like(flow_mask, dtype=np.float32)
        region_mask[flow_mask == 0] = np.nan

        # Região de recirculação: Ux/Uref < 0.4
        U_adim = Ux / Uref
        region_mask[(U_adim < 0.4) & (flow_mask == 1)] = 2

        # Região de baixa/alta pressão (z-score da pressão)
        p_valid = p[flow_mask == 1]
        if len(p_valid) > 1:
            mu_p, sigma_p = np.mean(p_valid), np.std(p_valid)
            if sigma_p > 0:
                p_z = (p - mu_p) / sigma_p
                region_mask[(p_z < -2) & (flow_mask == 1)] = 3
                region_mask[(p_z > 2) & (flow_mask == 1)] = 4

        # Verifica e salva
        assert region_mask.shape[0] == NX * NY
        region_masks[Re] = region_mask.reshape(NY, NX)

        print(
            f"  ✅ Re {Re:.2f} → region_mask criada com shape {region_masks[Re].shape}"
        )

    except Exception as e:
        print(f"  ⚠️  Falha ao gerar region_mask para {path} → {e}")
# -------------------------------------------------------- #
print("\n📦 Empilhando region_masks em tensor 3D [N, NY, NX]...\n")

region_mask_tensor = np.stack([region_masks[Re] for Re in sorted_reynolds], axis=0)

print(f"✅ region_mask_tensor criado com shape: {region_mask_tensor.shape}")

# -------------------------------------------------------- #
# ETAPA 5: GERAR dataX_tensor [3, 12, 100, 100]            #
# -------------------------------------------------------- #
print("\n🔵 ETAPA 5 — Gerando dataX_tensor [3, 12, 100, 100]...\n")

# Converte NaNs da region_mask para 0.0 se necessário
region_mask_tensor_clean = np.nan_to_num(region_mask_tensor, nan=0.0)

# Empilha os três canais como [C, N, H, W]
dataX_tensor = np.stack(
    [
        flow_mask_tensor.astype(np.float32),  # Canal 0
        reynolds_mask_norm.astype(np.float32),  # Canal 1
        region_mask_tensor_clean.astype(np.float32),  # Canal 2
    ],
    axis=0,
)

print(f"✅ dataX_tensor gerado com shape: {dataX_tensor.shape}")
# Esperado: (3, 12, 100, 100)
output_path = pathlib.Path("simulations/data/dataX.npy")
np.save(output_path, dataX_tensor)

print(f"💾 dataX_tensor salvo em: {output_path}")


# -------------------------------------------------------- #
# ETAPA 6: SALVAR 12 FIGURAS COM AS 3 MÁSCARAS             #
# -------------------------------------------------------- #
print("\n🔵 ETAPA 6 — Salvando figuras das máscaras por caso...\n")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

output_dir = "simulations/data_analysis"
os.makedirs(output_dir, exist_ok=True)

region_cmap = ListedColormap(["white", "lightblue", "orange", "purple", "red"])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm = BoundaryNorm(bounds, region_cmap.N, clip=False)

for i, Re in enumerate(sorted_reynolds):
    flow = flow_mask_tensor[i]
    reyn = reynolds_mask_norm[i]
    region = region_mask_tensor[i]  # mantém NaNs para visualização
    region_plot = np.nan_to_num(region, nan=0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(flow, cmap="Greys", interpolation="none")
    axes[0].set_title("flow_mask", fontsize=12)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(reyn, cmap="viridis", interpolation="none")
    axes[1].set_title("reynolds_mask_norm", fontsize=12)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(region_plot, cmap=region_cmap, norm=norm, interpolation="none")
    axes[2].set_title("region_mask", fontsize=12)
    cbar = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=[1, 2, 3, 4])
    cbar.ax.set_yticklabels(["fluido", "recirc.", "baixa p", "alta p"])

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    filename = f"{output_dir}/dataX_masks_case{i:02d}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"  ✅ Figura salva: {filename}")

print("\n✅ Todas as figuras das máscaras foram geradas e salvas.")
