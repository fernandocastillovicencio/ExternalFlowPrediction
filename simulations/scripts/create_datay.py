# -------------------------------------------------------- #
#                   STEP 1 - READ FOLDERS                  #
# -------------------------------------------------------- #
import os
import json
import glob

# Caminho base das simulações
base_path = "../cases"

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

    # Procurar o arquivo .xy em qualquer subpasta de tempo dentro de postProcessing/cloud/
    search_pattern = os.path.join(
        base_path, folder, "postProcessing", "cloud", "*", "ref_point_p_U.xy"
    )
    matched_files = glob.glob(search_pattern)

    if not matched_files:
        print(f"[ERRO] Arquivo não encontrado para Re = {Re:.2f}")
        continue

    filepath = matched_files[0]  # Usa o primeiro encontrado
    cases_info.append((folder, Re, Uref, filepath))

# ✅ Verificação: imprimir a lista
print("Casos encontrados:")
for folder, Re, Uref, path in cases_info:
    print(f"Re = {Re:.2f}, Uref = {Uref:.2e}, Caminho = {path}")


# -------------------------------------------------------- #
#              STEP 2: CREATE COORDINATES FILE             #
# -------------------------------------------------------- #
import numpy as np

# Usar o primeiro caso para extrair as coordenadas (malha é fixa)
_, _, _, first_file_path = cases_info[0]

# Carregar o arquivo: espera-se colunas [x, y, z, p, Ux, Uy, Uz]
data = np.loadtxt(first_file_path)

# Extrair apenas colunas x (0) e y (1)
x = data[:, 0]
y = data[:, 1]

# Reformatar para (100, 100)
x_2d = x.reshape((100, 100))
y_2d = y.reshape((100, 100))

# Empilhar em um único array com shape (100, 100, 2)
xy_coords = np.stack([x_2d, y_2d], axis=-1)

# Salvar (caminho relativo ao script)
os.makedirs("../data", exist_ok=True)
np.save("../data/xy_coords.npy", xy_coords)

# Verificação
print(f"\n✅ Coordenadas salvas em '../data/xy_coords.npy'. Shape: {xy_coords.shape}")
print("🔍 Exemplo de ponto (0,0):", xy_coords[0, 0])
