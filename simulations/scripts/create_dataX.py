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

pattern = "../simulations/cases/Re_*/postProcessing/cloud/1/ref_point_p_U.xy"
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

        flow_mask_flat = ~((p == 0.0) & (Ux == 0.0) & (Uy == 0.0))
        flow_mask_flat = flow_mask_flat.astype(np.uint8)

        assert flow_mask_flat.shape[0] == NX * NY, f"Shape inconsistente: {flow_mask_flat.shape[0]} != {NX*NY}"
        flow_mask = flow_mask_flat.reshape(NY, NX)

        flow_masks[Re] = flow_mask
        print(f"  ✅ Re {Re:.2f} → flow_mask gerada com {np.sum(flow_mask)} pontos válidos")

    except Exception as e:
        print(f"  ⚠️  Falha ao processar {path} → {e}")

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
    Re_raw_map = np.full((H, W), Re, dtype=np.float32)
    Re_norm_map = (Re - Re_min) / (Re_max - Re_min)

    reynolds_mask_raw[i] = Re_raw_map
    reynolds_mask_norm[i] = Re_norm_map

    print(f"  • Re = {Re:.2f} → raw = {Re:.2f} | norm = {Re_norm_map:.4f}")

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

        flow_mask = flow_masks[Re].reshape(-1)
        region_mask = np.ones_like(flow_mask, dtype=np.float32)
        region_mask[~flow_mask.astype(bool)] = np.nan

        U_adim = Ux / Uref
        region_mask[(U_adim < 0.4) & (flow_mask == 1)] = 2

        p_valid = p[flow_mask == 1]
        if len(p_valid) > 1:
            mu_p, sigma_p = np.mean(p_valid), np.std(p_valid)
            if sigma_p > 0:
                p_z = (p - mu_p) / sigma_p
                region_mask[(p_z < -2) & (flow_mask == 1)] = 3
                region_mask[(p_z > 2) & (flow_mask == 1)] = 4

        assert region_mask.shape[0] == NX * NY
        region_masks[Re] = region_mask.reshape(NY, NX)

        print(f"  ✅ Re {Re:.2f} → region_mask criada com shape {region_masks[Re].shape}")

    except Exception as e:
        print(f"  ⚠️  Falha ao gerar region_mask para {path} → {e}")

# ETAPA 5: SALVAR dataX.npy
print("\n💾 Empilhando e salvando dataX.npy...")

# Apenas Reynolds com máscaras válidas
valid_reynolds = [Re for Re in sorted_reynolds if Re in flow_masks]
N = len(valid_reynolds)
dataX = np.zeros((N, NY, NX, 3), dtype=np.float32)

for i, Re in enumerate(valid_reynolds):
    print(f"  🔹 Inserindo Re = {Re:.2f}")
    dataX[i, :, :, 0] = flow_masks[Re]
    dataX[i, :, :, 1] = reynolds_mask_norm[sorted_reynolds.index(Re)]
    dataX[i, :, :, 2] = region_masks[Re]

np.save("simulations/data/dataX.npy", dataX)
print("✅ Arquivo salvo em: simulations/data/dataX.npy")
