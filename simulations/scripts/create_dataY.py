# -------------------------------------------------------- #
# create_dataY.py – Gera o tensor de saída [Ux, Uy, p]    #
# -------------------------------------------------------- #

import json
import pathlib
import glob
import numpy as np
from utils_normalization import load_domain_and_velocities

# 🔹 Caminhos dos arquivos de configuração
config_path = pathlib.Path("simulations/scripts/cloud_config.json")
velocities_path = pathlib.Path("simulations/scripts/velocities.json")

# -------------------------------------------------------- #
#                ETAPA 1: LER CONFIGURAÇÕES                #
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
#      ETAPA 2: LER ARQUIVOS *.xy E MONTAR DATASET        #
# -------------------------------------------------------- #

print("\n🔵 ETAPA 2 — Lendo arquivos *.xy, aplicando máscara e empilhando dados...\n")

pattern = "simulations/cases/Re_*/postProcessing/cloud/1/ref_point_p_U.xy"
xy_paths = sorted(glob.glob(pattern))
raw_data = []

for path in xy_paths:
    try:
        # Determinar Reynolds da pasta
        parts = pathlib.Path(path).parts
        re_folder = [p for p in parts if p.startswith("Re_")][0]
        Re = float(re_folder.replace("Re_", "").replace("_", "."))

        # Ler dados
        data = np.genfromtxt(path)
        p = data[:, 3]
        Ux = data[:, 4]
        Uy = data[:, 5]

        # Máscara do obstáculo (p = Ux = Uy = 0)
        mask_obstacle = (p == 0.0) & (Ux == 0.0) & (Uy == 0.0)
        p[mask_obstacle] = np.nan
        Ux[mask_obstacle] = np.nan
        Uy[mask_obstacle] = np.nan

        # Reshape para [NY, NX]
        p = p.reshape((NY, NX))
        Ux = Ux.reshape((NY, NX))
        Uy = Uy.reshape((NY, NX))

        # Stack local → [NY, NX, 3]
        tensor = np.stack([Ux, Uy, p], axis=-1)
        raw_data.append(tensor)

        print(f"  ✅ Re = {Re:.2f} → campos extraídos e reorganizados.")

    except Exception as e:
        print(f"  ❌ Erro ao processar {path}: {e}")

# Stack final: [N, NY, NX, 3]
raw_dataset = np.stack(raw_data, axis=0)

# Salvar
np.save("simulations/data/dataY-raw.npy", raw_dataset)
print("\n✅ Arquivo salvo: simulations/data/dataY-raw.npy")
print(f"📐 Shape final: {raw_dataset.shape}  (N, NY, NX, 3)")

# -------------------------------------------------------- #
#     ETAPA 3: ADIMENSIONALIZAÇÃO E SALVAMENTO .NPY       #
# -------------------------------------------------------- #

print("\n🔵 ETAPA 3 — Adimensionalizando os campos com base em Uref e ρ = 1.225...\n")

rho = 1.225
dimless_data = []

for i, Re in enumerate(sorted_reynolds):
    Uref = velocities[Re]
    tensor = raw_data[i]  # shape: [NY, NX, 3]

    Ux = tensor[:, :, 0] / Uref
    Uy = tensor[:, :, 1] / Uref
    p = tensor[:, :, 2] / (0.5 * rho * Uref**2)

    dimless_tensor = np.stack([Ux, Uy, p], axis=-1)
    dimless_data.append(dimless_tensor)

    print(f"  ✅ Re = {Re:.2f} → adimensionalização aplicada.")

# Stack final: [N, NY, NX, 3]
dimless_dataset = np.stack(dimless_data, axis=0)

# Salvar
np.save("simulations/data/dataY-dimensionless.npy", dimless_dataset)
print("\n✅ Arquivo salvo: simulations/data/dataY-dimensionless.npy")
print(f"📐 Shape final: {dimless_dataset.shape}  (N, NY, NX, 3)")

# -------------------------------------------------------- #
#     ETAPA 4: NORMALIZAÇÃO (Z-score) E SALVAMENTO .NPY   #
# -------------------------------------------------------- #

print("\n🔵 ETAPA 4 — Normalizando os campos com Z-score (por canal)...\n")

# Separar os canais do tensor adimensionalizado
Ux_all = dimless_dataset[:, :, :, 0]
Uy_all = dimless_dataset[:, :, :, 1]
p_all = dimless_dataset[:, :, :, 2]

# Calcular médias e desvios (ignorando NaNs)
mu_Ux = np.nanmean(Ux_all)
mu_Uy = np.nanmean(Uy_all)
mu_p = np.nanmean(p_all)

std_Ux = np.nanstd(Ux_all)
std_Uy = np.nanstd(Uy_all)
std_p = np.nanstd(p_all)

print(f"  • Ux: μ = {mu_Ux:.4f}, σ = {std_Ux:.4f}")
print(f"  • Uy: μ = {mu_Uy:.4f}, σ = {std_Uy:.4f}")
print(f"  •  p: μ = {mu_p:.4f}, σ = {std_p:.4f}")

# Normalizar canal por canal (preservando NaNs)
Ux_norm = (Ux_all - mu_Ux) / std_Ux
Uy_norm = (Uy_all - mu_Uy) / std_Uy
p_norm = (p_all - mu_p) / std_p

# Reempilhar
normalized_dataset = np.stack([Ux_norm, Uy_norm, p_norm], axis=-1)

# Salvar
np.save("simulations/data/dataY-normalized.npy", normalized_dataset)
print("\n✅ Arquivo salvo: simulations/data/dataY-normalized.npy")
print(f"📐 Shape final: {normalized_dataset.shape}  (N, NY, NX, 3)")


# -------------------------------------------------------- #
#  ETAPA 5 — SALVAR TUDO EM UM ÚNICO JSON PARA REVERTER   #
# -------------------------------------------------------- #

print("\n🔵 ETAPA 5 — Salvando parâmetros de normalização e dimensionalização...\n")

# Construir dicionário único
stats_all = {
    "rho": 1.225,
    "Ux": {"mean": float(mu_Ux), "std": float(std_Ux)},
    "Uy": {"mean": float(mu_Uy), "std": float(std_Uy)},
    "p": {"mean": float(mu_p), "std": float(std_p)},
    "Uref": {f"{Re:.2f}": float(velocities[Re]) for Re in sorted_reynolds},
}

# Salvar em simulations/data/dataY-norm-params.json
stats_path = pathlib.Path("simulations/data/dataY-norm-params.json")
with stats_path.open("w") as f:
    json.dump(stats_all, f, indent=4)

print(f"✅ Arquivo salvo: {stats_path}")


# -------------------------------------------------------- #
#           ETAPA 6 — GERAÇÃO DE FLOW_MASK (opcional)     #
# -------------------------------------------------------- #

print("\n🔵 ETAPA 6 — Gerando flow_mask.npy com base em p...\n")

# A máscara é baseada na presença de NaN no campo de pressão adimensionalizado
flow_mask = ~np.isnan(dimless_dataset[:, :, :, 2])
np.save("simulations/data/flow_mask.npy", flow_mask.astype(np.uint8))

print(f"✅ flow_mask.npy gerado com shape: {flow_mask.shape}")
