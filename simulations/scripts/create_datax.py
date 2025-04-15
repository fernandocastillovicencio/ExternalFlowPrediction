# -------------------------------------------------------- #
#                CREATE DATAX FOR FIXED CIRCLE             #
#     (Entrada: Reynolds + posição espacial normalizada)   #
# -------------------------------------------------------- #

import numpy as np
import json
import pathlib
import os
import pandas as pd

# Caminhos dos arquivos
velocities_path = pathlib.Path("../data/velocities.json")
output_npy = pathlib.Path("../data/dataX.npy")
output_csv = pathlib.Path("../data/dataX.csv")

# Parâmetros do domínio
H, W = 100, 100  # resolução da malha (y, x)

# -------------------------------------------------------- #
#                  1. CARREGAR VELOCITIES                  #
# -------------------------------------------------------- #
with velocities_path.open("r") as f:
    velocities = json.load(f, object_pairs_hook=lambda x: {float(k): v for k, v in x})

sorted_reynolds = sorted(velocities.keys())

# -------------------------------------------------------- #
#                 2. CONSTRUIR OS MAPAS                    #
# -------------------------------------------------------- #
N = len(sorted_reynolds)
dataX = np.zeros((N, 3, H, W), dtype=np.float32)

# Mapa de coordenadas normalizadas
x_coords = np.linspace(0, 1, W)[np.newaxis, :].repeat(H, axis=0)  # (H, W)
y_coords = np.linspace(0, 1, H)[:, np.newaxis].repeat(W, axis=1)  # (H, W)

# Preencher canais
for i, Re in enumerate(sorted_reynolds):
    dataX[i, 0] = Re  # Re_map
    dataX[i, 1] = x_coords
    dataX[i, 2] = y_coords

# -------------------------------------------------------- #
#                  3. SALVAR OS DADOS                      #
# -------------------------------------------------------- #
os.makedirs(output_npy.parent, exist_ok=True)
np.save(output_npy, dataX)

# CSV para visualização básica
df = pd.DataFrame({"Re": sorted_reynolds})
df.to_csv(output_csv, index=False)

# -------------------------------------------------------- #
#                    4. VERIFICAÇÃO                        #
# -------------------------------------------------------- #
print(f"✅ dataX salvo em: {output_npy} — shape: {dataX.shape}")
print(f"✅ CSV salvo em: {output_csv}")
print("\n🔍 Visualização dos Reynolds:")
print(df.head())
