import pathlib
import glob
import numpy as np
import json
from utils_normalization import load_domain_and_velocities
from matplotlib.colors import ListedColormap, BoundaryNorm
from functions import (
    detectar_frontal_e_esteira,
    calculo_da_inflexao,
    detectar_x_inflexao,
    aplicar_frontal_por_inflexao,
)

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

import matplotlib.pyplot as plt
import os

pattern = "simulations/cases/Re_*/postProcessing/cloud/1/ref_point_p_U.xy"
xy_paths = sorted(glob.glob(pattern))
flow_masks = {}

# Diretório para salvar as figuras
plot_dir = "simulations/data_analysis"
os.makedirs(plot_dir, exist_ok=True)

for idx, path in enumerate(xy_paths):
    try:
        parts = pathlib.Path(path).parts
        re_folder = [p for p in parts if p.startswith("Re_")][0]
        Re = float(re_folder.replace("Re_", "").replace("_", "."))

        Uref = velocities[Re]
        data = np.genfromtxt(path)

        p = data[:, 3]
        Ux = data[:, 4]
        Uy = data[:, 5]

        zero_mask = (p == 0.0) & (Ux == 0.0) & (Uy == 0.0)
        print(f"  → Nº de pontos com p = Ux = Uy = 0: {np.sum(zero_mask)}")

        # Define máscara: True onde é fluido (não obstáculo)
        invalid_mask = (p == 0.0) & (Ux == 0.0) & (Uy == 0.0) | np.isnan(p) & np.isnan(
            Ux
        ) & np.isnan(Uy)
        flow_mask_flat = (~invalid_mask).astype(np.uint8)

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

        print(f"Valores únicos da flow_mask para Re={Re:.2f}: {np.unique(flow_mask)}")

        # 🔵 Plot e salvamento
        fig, ax = plt.subplots(figsize=(5, 5))

        # Cores: [obstáculo=branco, fluido=azul claro]
        flow_cmap = ListedColormap(["white", "lightblue"])

        im = ax.imshow(flow_mask, cmap=flow_cmap, interpolation="none")

        ax.set_title(f"flow_mask – Re {Re:.2f}")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(f"{plot_dir}/flow_mask_case{idx:02d}.png", dpi=150)
        plt.close()

    except Exception as e:
        print(f"  ⚠️  Falha ao processar {path} → {e}")
# -------------------------------------------------------- #
print("\n📦 Empilhando máscaras em tensor 3D [N, NY, NX]...\n")

flow_mask_tensor = np.stack([flow_masks[Re] for Re in sorted_reynolds], axis=0)

print(f"✅ flow_mask_tensor criado com shape: {flow_mask_tensor.shape}")


# -------------------------------------------------------- #
# ETAPA 3: REYNOLDS MASK com flow_mask + PLOT CORRIGIDO    #
# -------------------------------------------------------- #
print(
    "\n🔵 ETAPA 3 — Gerando reynolds_mask_raw, reynolds_mask_norm e salvando figuras...\n"
)

import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

H, W = NY, NX
N = len(sorted_reynolds)

reynolds_mask_raw = np.zeros((N, H, W), dtype=np.float32)
reynolds_mask_norm = np.zeros((N, H, W), dtype=np.float32)

Re_min = min(velocities.keys())
Re_max = max(velocities.keys())

output_dir = "simulations/data_analysis"
os.makedirs(output_dir, exist_ok=True)


for i, Re in enumerate(sorted_reynolds):
    flow_mask = flow_mask_tensor[i]

    Re_norm_value = (Re - Re_min) / (Re_max - Re_min)

    raw_mask = np.zeros((H, W), dtype=np.float32)
    norm_mask = np.zeros((H, W), dtype=np.float32)

    raw_mask[flow_mask == 1] = Re
    norm_mask[flow_mask == 1] = Re_norm_value

    reynolds_mask_raw[i] = raw_mask
    reynolds_mask_norm[i] = norm_mask

    print(f"  • Re = {Re:.2f} → raw = {Re:.2f} | norm = {Re_norm_value:.4f}")

    # Plot da máscara norm
    fig, ax = plt.subplots(figsize=(5, 5))
    # Define máscara temporária com NaN no obstáculo para plot
    masked_plot = norm_mask.copy()
    masked_plot[flow_mask == 0] = np.nan

    im = ax.imshow(masked_plot, cmap="turbo", vmin=0.0, vmax=1.0)

    ax.set_title(f"Re_norm – Re {Re:.2f}", fontsize=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Re_norm", fontsize=10)
    ax.axis("off")

    filename = f"{output_dir}/reynolds_mask_case{i:02d}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"    ✅ Figura salva: {filename}")

# -------------------------------------------------------- #
print(f"\n📐 reynolds_mask_raw.shape  = {reynolds_mask_raw.shape}")
print(f"📐 reynolds_mask_norm.shape = {reynolds_mask_norm.shape}")
print("\n✅ Máscaras e figuras da ETAPA 3 geradas com sucesso.")

# -------------------------------------------------------- #
#                          ETAPA 4                         #
# -------------------------------------------------------- #

region_masks = {}

for i, Re in enumerate(sorted_reynolds):
    path = f"simulations/cases/Re_{Re:.2f}/postProcessing/cloud/1/ref_point_p_U.xy"
    try:
        parts = pathlib.Path(path).parts
        re_folder = [p for p in parts if p.startswith("Re_")][0]
        Re = float(re_folder.replace("Re_", "").replace("_", "."))

        Uref = velocities[Re]
        data = np.genfromtxt(path)
        p = data[:, 3].astype(float)
        Ux = data[:, 4].astype(float)
        Uy = data[:, 5].astype(float)
        x = data[:, 0].astype(float)
        y = data[:, 1].astype(float)

        flow_mask = flow_masks[Re].reshape(-1)

        # Inicializa region_mask
        region_mask = np.ones_like(flow_mask, dtype=np.float32)
        region_mask[flow_mask == 0] = np.nan

        # Classificação base por velocidade adimensional
        U_adim = Ux / Uref
        region_mask[(U_adim <= 0.2) & (flow_mask == 1)] = 2
        region_mask[(U_adim > 0.2) & (flow_mask == 1)] = 1

        # ➕ Identificação da borda e ponto de inflexão
        borda_sup, borda_inf = calculo_da_inflexao(region_mask, x, y, flow_mask)
        x_inflexao_sup = detectar_x_inflexao(borda_sup, label="superior")
        x_inflexao_inf = detectar_x_inflexao(borda_inf, label="inferior")

        # ⚠️ Só aplica se ambos os pontos forem válidos
        if x_inflexao_sup is not None and x_inflexao_inf is not None:
            print(
                f"📍 x_inflexao_sup = {x_inflexao_sup:.4f}, x_inflexao_inf = {x_inflexao_inf:.4f}"
            )
            region_mask = aplicar_frontal_por_inflexao(
                region_mask, x, y, flow_mask, x_inflexao_inf, x_inflexao_sup
            )
        else:
            print(
                f"⚠️  Inflection points não definidos para Re {Re:.2f}, mantendo classificação anterior."
            )

        # Reshape para 2D e salvar
        region_2d = region_mask.reshape(NY, NX)
        region_masks[Re] = region_2d

        print(f"  ✅ Re {Re:.2f} → region_mask criada com shape {region_2d.shape}")

        # 🔵 PLOT
        region_plot = region_2d.copy()
        region_plot[np.isnan(region_plot)] = np.nan

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(region_plot, cmap="turbo", vmin=1, vmax=3, interpolation="none")
        ax.set_title(f"region_mask – Re {Re:.2f}", fontsize=12)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(
            "região (1 = fluido, 2 = frontal , 3 = esteira)",
            fontsize=9,
        )

        ax.axis("off")
        plt.tight_layout()

        filename = f"{output_dir}/region_mask_case{i:02d}_turbo.png"
        plt.savefig(filename, dpi=150)
        plt.close()

        print(f"     🎨 Figura salva: {filename}")

    except Exception as e:
        print(f"  ⚠️  Falha ao gerar region_mask para {path} → {e}")


# -------------------------------------------------------- #
# ETAPA 5: Salvamento final dos dados para treinamento     #
# -------------------------------------------------------- #
import os

print("\n💾 ETAPA 5 — Salvando arquivos finais para treino da rede...\n")

# 🔧 Cria pasta se necessário
output_dir_final = "data"
os.makedirs(output_dir_final, exist_ok=True)

# ✅ 1. Reynolds log-normalizado
reynolds_raw = np.array(sorted_reynolds, dtype=np.float32).reshape(-1, 1)
log_Re = np.log10(reynolds_raw)
log_min = log_Re.min()
log_max = log_Re.max()
reynolds_log_norm = ((log_Re - log_min) / (log_max - log_min)).astype(np.float32)

np.save(f"{output_dir_final}/dataX-log-normalized.npy", reynolds_log_norm)

# ✅ 2. flow_mask_tensor (já empilhado)
np.save(f"{output_dir_final}/flow_mask.npy", flow_mask_tensor)

# ✅ 3. region_mask_tensor (empilhar os dicionários)
region_mask_tensor = np.stack(
    [region_masks[Re] for Re in sorted_reynolds], axis=0
).astype(np.float32)
np.save(f"{output_dir_final}/region_mask_tensor.npy", region_mask_tensor)

# ✅ 4. parâmetros de normalização do Reynolds
norm_params = {
    "Re_raw_min": float(reynolds_raw.min()),
    "Re_raw_max": float(reynolds_raw.max()),
    "log10_min": float(log_min),
    "log10_max": float(log_max),
    "Re_sorted": [float(r) for r in sorted_reynolds],
}
with open(f"{output_dir_final}/dataX-log-norm-params.json", "w") as f:
    json.dump(norm_params, f, indent=4)

print("✅ ETAPA 5 finalizada. Arquivos salvos em 'data/'\n")
