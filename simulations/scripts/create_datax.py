# -------------------------------------------------------- #
#                CREATE DATAX FOR FIXED CIRCLE             #
#          (Entrada apenas: Número de Reynolds)            #
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

# -------------------------------------------------------- #
#                  1. CARREGAR VELOCITIES                  #
# -------------------------------------------------------- #
with velocities_path.open("r") as f:
    velocities = json.load(f, object_pairs_hook=lambda x: {float(k): v for k, v in x})

# Ordenar os Reynolds (garante ordem crescente)
sorted_reynolds = sorted(velocities.keys())

# -------------------------------------------------------- #
#                 2. CONSTRUIR O DATASET                   #
# -------------------------------------------------------- #
# Criar dataX: vetor com apenas o Reynolds
dataX = np.array([[Re] for Re in sorted_reynolds])  # shape (12, 1)

# Criar pasta de saída se necessário
os.makedirs(output_npy.parent, exist_ok=True)

# -------------------------------------------------------- #
#                 3. SALVAR OS ARQUIVOS                    #
# -------------------------------------------------------- #
# Salvar .npy
np.save(output_npy, dataX)

# Salvar .csv para visualização
df = pd.DataFrame(dataX, columns=["Re"])
df.to_csv(output_csv, index=False)

# -------------------------------------------------------- #
#                   4. VERIFICAÇÃO                         #
# -------------------------------------------------------- #
print(f"✅ dataX salvo em: {output_npy} — shape: {dataX.shape}")
print(f"✅ CSV salvo em: {output_csv}")
print("\n🔍 Visualização:")
print(df.head())
