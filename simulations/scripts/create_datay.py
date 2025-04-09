import os
import json

# Caminho base das simulações
base_path = "simulations/cases"

# Caminho do JSON de velocidades
velocities_path = "../data/velocities.json"

# Carregar o JSON
with open(velocities_path, "r") as f:
    velocities_raw = json.load(f)

# Converter as chaves para float e ordenar
velocities = {float(k): v for k, v in velocities_raw.items()}
sorted_reynolds = sorted(velocities.keys())

# Lista final: (pasta, Re, Uref, caminho_arquivo)
cases_info = []

for Re in sorted_reynolds:
    Uref = velocities[Re]
    folder = f"Re_{Re:.2f}"
    filepath = os.path.join(
        base_path, folder, "postProcessing/cloud/1/ref_point_p_U.xy"
    )
    cases_info.append((folder, Re, Uref, filepath))

# ✅ Verificação: imprimir a lista
print("Casos encontrados:")
for folder, Re, Uref, path in cases_info:
    print(f"Re = {Re:.2f}, Uref = {Uref:.2e}, Caminho = {path}")

# -------------------------------------------------------- #
