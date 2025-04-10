# -------------------------------------------------------- #
#                   STEP 1 - READ FOLDERS                  #
# -------------------------------------------------------- #
import json
import glob
import pathlib

# Caminho base das simulações
base_path = pathlib.Path("../cases")

# Caminho do JSON de velocidades
velocities_path = pathlib.Path("../data/velocities.json")

# Carregar o JSON
with velocities_path.open("r") as f:
    velocities = json.load(f, object_pairs_hook=lambda x: {float(k): v for k, v in x})

# Lista final: (pasta, Re, Uref, caminho_arquivo)
cases_info = []

for Re, Uref in velocities.items():
    folder = f"Re_{Re:.2f}"

    # Procurar o arquivo .xy em qualquer subpasta de tempo dentro de postProcessing/cloud/
    search_pattern = (
        base_path / folder / "postProcessing" / "cloud" / "*" / "ref_point_p_U.xy"
    )
    matched_files = glob.glob(str(search_pattern))

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
import os
import pandas as pd

# Usar o primeiro caso para extrair as coordenadas (malha é fixa)
_, _, _, first_file_path = cases_info[0]

# Carregar o arquivo: espera-se colunas [x, y, z, p, Ux, Uy, Uz]
data = np.genfromtxt(first_file_path)

# Extrair apenas colunas x (0) e y (1)
x = data[:, 0]
y = data[:, 1]

# Reformatar para (100, 100)
x_2d = np.reshape(x, (100, 100))
y_2d = np.reshape(y, (100, 100))

# Empilhar em um único array com shape (100, 100, 2)
xy_coords = np.stack([x_2d, y_2d], axis=-1)

# Salvar (caminho relativo ao script)
os.makedirs("../data", exist_ok=True)
np.save("../data/xy_coords.npy", xy_coords)

# Criar um arquivo CSV para verificar o conteúdo do arquivo NPY
df = pd.DataFrame(xy_coords.reshape(-1, 2), columns=["x", "y"])
df.to_csv("../data/xy_coords.csv", index=False)

# Verificação
print(f"\n✅ Coordenadas salvas em '../data/xy_coords.npy'. Shape: {xy_coords.shape}")
print("✅ Arquivo CSV criado em '../data/xy_coords.csv'")
print("🔍 Exemplo de ponto (0,0):", xy_coords[0, 0])

# -------------------------------------------------------- #
#                STEP 3: CREATE RAW DATA FILE              #
# -------------------------------------------------------- #

raw_data = []

for folder, Re, Uref, path in cases_info:
    try:
        data = np.genfromtxt(path)

        # Extrair colunas: p (3), Ux (4), Uy (5)
        p = data[:, 3].reshape((100, 100))
        Ux = data[:, 4].reshape((100, 100))
        Uy = data[:, 5].reshape((100, 100))

        # Stack como canais: (100, 100, 3)
        tensor = np.stack([Ux, Uy, p], axis=-1)
        raw_data.append(tensor)

        print(f"✅ Dados carregados para Re = {Re:.2f}")

    except Exception as e:
        print(f"[ERRO] Falha ao processar Re = {Re:.2f}: {e}")

# Empilhar todos os casos → shape final: (12, 100, 100, 3)
raw_dataset = np.stack(raw_data, axis=0)

# Salvar
np.save("../data/dataY-raw.npy", raw_dataset)
print(f"\n✅ Dataset bruto salvo em '../data/dataY-raw.npy' — shape: {raw_dataset.shape}")

# Criar um arquivo CSV para verificar o conteúdo do arquivo NPY
import pandas as pd
df = pd.DataFrame(raw_dataset.reshape(-1, 3), columns=["Ux", "Uy", "p"])
df.to_csv("../data/dataY-raw.csv", index=False)
print("✅ Arquivo CSV criado em '../data/dataY-raw.csv'")

# -------------------------------------------------------- #
#              STEP 4: CREATE DIMENSIONLESS FILE           #
# -------------------------------------------------------- #

rho = 1.225
dimensionless_data = []

for i, (folder, Re, Uref, _) in enumerate(cases_info):
    # Extrair os canais do raw_dataset
    Ux = raw_dataset[i, :, :, 0]
    Uy = raw_dataset[i, :, :, 1]
    p = raw_dataset[i, :, :, 2]

    # Adimensionalização
    Ux_dimless = Ux / Uref
    Uy_dimless = Uy / Uref
    p_dimless = p / (0.5 * rho * Uref**2)

    # Empilhar novamente
    tensor = np.stack([Ux_dimless, Uy_dimless, p_dimless], axis=-1)
    dimensionless_data.append(tensor)

    print(f"✅ Dados adimensionais gerados para Re = {Re:.2f}")

# Empilhar todos → shape: (12, 100, 100, 3)
dimless_dataset = np.stack(dimensionless_data, axis=0)

# Salvar
np.save("../data/dataY-dimensionless.npy", dimless_dataset)
print(f"\n✅ Dataset adimensional salvo em '../data/dataY-dimensionless.npy' — shape: {dimless_dataset.shape}")

# Criar CSV para verificação
df = pd.DataFrame(dimless_dataset.reshape(-1, 3), columns=["Ux*", "Uy*", "p*"])
df.to_csv("../data/dataY-dimensionless.csv", index=False)
print("✅ Arquivo CSV criado em '../data/dataY-dimensionless.csv'")

# -------------------------------------------------------- #
#              STEP 5: CREATE NORMALIZED FILE              #
# -------------------------------------------------------- #
import json

# Calcular min e max globais por canal
Ux_min = np.min(dimless_dataset[:, :, :, 0])
Ux_max = np.max(dimless_dataset[:, :, :, 0])
Uy_min = np.min(dimless_dataset[:, :, :, 1])
Uy_max = np.max(dimless_dataset[:, :, :, 1])
p_min = np.min(dimless_dataset[:, :, :, 2])
p_max = np.max(dimless_dataset[:, :, :, 2])

print("\n🔍 Parâmetros da normalização (min, max):")
print(f"Ux*: ({Ux_min:.4f}, {Ux_max:.4f})")
print(f"Uy*: ({Uy_min:.4f}, {Uy_max:.4f})")
print(f"p* : ({p_min:.4f}, {p_max:.4f})")

# Aplicar normalização Min-Max
normalized_dataset = np.empty_like(dimless_dataset)

normalized_dataset[:, :, :, 0] = (dimless_dataset[:, :, :, 0] - Ux_min) / (Ux_max - Ux_min)
normalized_dataset[:, :, :, 1] = (dimless_dataset[:, :, :, 1] - Uy_min) / (Uy_max - Uy_min)
normalized_dataset[:, :, :, 2] = (dimless_dataset[:, :, :, 2] - p_min) / (p_max - p_min)

# Salvar arquivo normalizado
np.save("../data/dataY-normalized.npy", normalized_dataset)
print("✅ Dataset normalizado salvo em '../data/dataY-normalized.npy'")

# Salvar parâmetros de normalização para de-normalização futura
norm_params = {
    "Ux": {"min": float(Ux_min), "max": float(Ux_max)},
    "Uy": {"min": float(Uy_min), "max": float(Uy_max)},
    "p":  {"min": float(p_min), "max": float(p_max)}
}

with open("../data/dataY-norm-params.json", "w") as f:
    json.dump(norm_params, f, indent=4)

print("✅ Parâmetros de normalização salvos em '../data/dataY-norm-params.json'")

# CSV opcional para checagem
df = pd.DataFrame(normalized_dataset.reshape(-1, 3), columns=["Ux_n", "Uy_n", "p_n"])
df.to_csv("../data/dataY-normalized.csv", index=False)
print("✅ Arquivo CSV criado em '../data/dataY-normalized.csv'")
