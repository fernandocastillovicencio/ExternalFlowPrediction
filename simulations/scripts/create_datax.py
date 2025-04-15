# -------------------------------------------------------- #
#           CREATE DATAX COM COORDENADAS X, Y             #
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
H, W = 100, 100  # resolução da malha

# 1. CARREGAR VALORES DE VELOCIDADE
with velocities_path.open("r") as f:
    velocities = json.load(f, object_pairs_hook=lambda x: {float(k): v for k, v in x})

sorted_reynolds = sorted(velocities.keys())
N = len(sorted_reynolds)

# 2. CONSTRUIR OS MAPAS
dataX = np.zeros((N, 5, H, W), dtype=np.float32)

# Coordenadas normalizadas
x_coords = np.linspace(0, 1, W)[np.newaxis, :].repeat(H, axis=0)  # (H, W)
y_coords = np.linspace(0, 1, H)[:, np.newaxis].repeat(W, axis=1)  # (H, W)

for i, Re in enumerate(sorted_reynolds):
    dataX[i, 0] = Re  # Re_map
    dataX[i, 1] = x_coords
    dataX[i, 2] = y_coords
    dataX[i, 3] = x_coords  # repetição redundante será útil se aplicar outros mapas
    dataX[i, 4] = y_coords

# 3. SALVAR OS DADOS
os.makedirs(output_npy.parent, exist_ok=True)
np.save(output_npy, dataX)

# CSV com valores de Re
df = pd.DataFrame({"Re": sorted_reynolds})
df.to_csv(output_csv, index=False)

# 4. VERIFICAÇÃO
print(f"✅ dataX salvo em: {output_npy} — shape: {dataX.shape}")
print(f"✅ CSV salvo em: {output_csv}")
print("\n🔍 Visualização dos Reynolds:")
print(df.head())
