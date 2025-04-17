# -------------------------------------------------------- #
# denormalize_dataY.py – Reverte normalização e unidades
# -------------------------------------------------------- #

import numpy as np
import json
import pathlib

# Caminhos
dataY_path = pathlib.Path("simulations/data/dataY-normalized.npy")
stats_path = pathlib.Path("simulations/data/dataY_stats.json")
output_path = pathlib.Path("simulations/data/dataY-recovered.npy")

# Carregar dados e parâmetros
data = np.load(dataY_path)
with stats_path.open("r") as f:
    stats = json.load(f)

rho = stats["rho"]
Uref_dict = {float(k): v for k, v in stats["Uref"].items()}

mu_Ux, std_Ux = stats["Ux"]["mean"], stats["Ux"]["std"]
mu_Uy, std_Uy = stats["Uy"]["mean"], stats["Uy"]["std"]
mu_p,  std_p  = stats["p"]["mean"],  stats["p"]["std"]

N = data.shape[0]
reynolds = sorted(Uref_dict.keys())[:N]  # garantir ordem compatível

print("🔄 Revertendo normalização e adimensionalização...\n")

recovered = []
for i in range(N):
    Uref = Uref_dict[reynolds[i]]
    sample = data[i]  # shape: [NY, NX, 3]

    Ux = sample[:, :, 0] * std_Ux + mu_Ux
    Uy = sample[:, :, 1] * std_Uy + mu_Uy
    p  = sample[:, :, 2] * std_p  + mu_p

    # Redimensionalização
    Ux *= Uref
    Uy *= Uref
    p  *= rho * Uref**2

    recovered.append(np.stack([Ux, Uy, p], axis=-1))
    print(f"  ✅ Re = {reynolds[i]:.2f} recuperado.")

# Stack final
recovered_array = np.stack(recovered, axis=0)
np.save(output_path, recovered_array)

print(f"\n✅ Arquivo salvo: {output_path}")
print(f"📐 Shape: {recovered_array.shape}")
