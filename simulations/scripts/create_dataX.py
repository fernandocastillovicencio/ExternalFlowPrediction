# -------------------------------------------------------- #
# STEP 1: Extrair configuração do domínio de cloud_config.json
#         e os Reynolds disponíveis de velocities.json
# -------------------------------------------------------- #

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
#                    ETAPA 2: FLOW MASK                    #
# -------------------------------------------------------- #
print("\n🔵 ETAPA 2 — Lendo arquivos *.xy e gerando flow_mask...\n")

pattern = "../simulations/cases/Re_*/postProcessing/cloud/1/ref_point_p_U.xy"
xy_paths = sorted(glob.glob(pattern))
flow_masks = {}

for path in xy_paths:
    try:
        # Extrair Re a partir do nome da pasta
        parts = pathlib.Path(path).parts
        re_folder = [p for p in parts if p.startswith("Re_")][0]
        Re = float(re_folder.replace("Re_", "").replace("_", "."))

        # Carregar Uref
        Uref = velocities[Re]

        # Ler dados
        data = np.genfromtxt(path)
        p = data[:, 3]
        Ux = data[:, 4]
        Uy = data[:, 5]

        # Criar máscara: 0 onde todas as variáveis são zero
        flow_mask = ~((p == 0.0) & (Ux == 0.0) & (Uy == 0.0))
        flow_mask = flow_mask.astype(np.uint8)  # 1 = fluido, 0 = obstáculo

        flow_masks[Re] = flow_mask

        print(f"  ✅ Re {Re:.2f} → flow_mask gerada ({np.sum(flow_mask)} pontos válidos)")

    except Exception as e:
        print(f"  ⚠️  Falha ao processar {path} → {e}")

# -------------------------------------------------------- #
#                  ETAPA 3: REYNOLDS MASK                  #
# -------------------------------------------------------- #
print("\n🔵 ETAPA 3 — Gerando reynolds_mask_raw e reynolds_mask_norm (em memória)...\n")

# Parâmetros da malha (do cloud_config)
H, W = NY, NX
N = len(sorted_reynolds)

# Inicializar máscaras
reynolds_mask_raw = np.zeros((N, H, W), dtype=np.float32)
reynolds_mask_norm = np.zeros((N, H, W), dtype=np.float32)

Re_min = min(velocities.keys())
Re_max = max(velocities.keys())

for i, Re in enumerate(sorted_reynolds):
    Re_raw_map = np.ones((H, W), dtype=np.float32) * Re
    Re_norm_map = (Re - Re_min) / (Re_max - Re_min)

    reynolds_mask_raw[i] = Re_raw_map
    reynolds_mask_norm[i] = Re_norm_map

    print(f"  • Re = {Re:.2f} → raw = {Re:.2f} | norm = {Re_norm_map:.4f}")

print("\n✅ Máscaras de Reynolds geradas e armazenadas em memória.")

# -------------------------------------------------------- #
#                   ETAPA 4: REGION MASK                   #
# -------------------------------------------------------- #