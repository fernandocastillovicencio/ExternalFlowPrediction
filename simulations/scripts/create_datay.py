import numpy as np
import os
import pandas as pd
import json

# -------------------------------------------------------- #
#            STEP 1: DETECT AVAILABLE SIMULATIONS          #
# -------------------------------------------------------- #
import glob

pattern = "../cases/Re_*/postProcessing/cloud/1/ref_point_p_U.xy"
paths = sorted(glob.glob(pattern))

cases_info = []

print("Casos encontrados:")
for path in paths:
    folder = path.split("/postProcessing")[0]
    re_str = folder.split("_")[-1]
    Re = float(re_str)

    # A referência de velocidade pode ser arbitrária (usada na adimensionalização)
    Uref = 1.5e-7 * (Re / 0.01)  # base de Re_0.01
    cases_info.append((folder, Re, Uref, path))

    print(f"Re = {Re:.2f}, Uref = {Uref:.2e}, Caminho = {path}")

# -------------------------------------------------------- #
#            STEP 2: EXTRACT COORDINATES (XY)              #
# -------------------------------------------------------- #

# Usar o primeiro caso para extrair as coordenadas (malha é fixa)
_, _, _, first_file_path = cases_info[0]

# Carregar o arquivo: espera-se colunas [x, y, z, p, Ux, Uy, Uz]
data = np.genfromtxt(first_file_path)

x = data[:, 0]
y = data[:, 1]

x_2d = np.reshape(x, (100, 100))
y_2d = np.reshape(y, (100, 100))

xy_coords = np.stack([x_2d, y_2d], axis=-1)

os.makedirs("../data", exist_ok=True)
np.save("../data/xy_coords.npy", xy_coords)

# CSV de coordenadas
df = pd.DataFrame(xy_coords.reshape(-1, 2), columns=["x", "y"])
df.to_csv("../data/xy_coords.csv", index=False)

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

        # Extrair colunas (sem adimensionalizar)
        p = data[:, 3]
        Ux = data[:, 4]
        Uy = data[:, 5]

        # Substituir valores do obstáculo por NaN
        obstacle_mask = (p == 0.0) & (Ux == 0.0) & (Uy == 0.0)
        p[obstacle_mask] = np.nan
        Ux[obstacle_mask] = np.nan
        Uy[obstacle_mask] = np.nan

        # Reformatar (mantendo valores originais)
        p = p.reshape((100, 100))
        Ux = Ux.reshape((100, 100))
        Uy = Uy.reshape((100, 100))

        tensor = np.stack([Ux, Uy, p], axis=-1)
        raw_data.append(tensor)

        print(f"✅ Dados carregados para Re = {Re:.2f}")

    except Exception as e:
        print(f"[ERRO] Falha ao processar Re = {Re:.2f}: {e}")

# Stack final
raw_dataset = np.stack(raw_data, axis=0)
np.save("../data/dataY-raw.npy", raw_dataset)

# CSV para visualização
df = pd.DataFrame(raw_dataset.reshape(-1, 3), columns=["Ux", "Uy", "p"])
df.to_csv("../data/dataY-raw.csv", index=False)

print("\n✅ Dataset bruto salvo em '../data/dataY-raw.npy' — shape:", raw_dataset.shape)
print("✅ Arquivo CSV criado em '../data/dataY-raw.csv'")

# -------------------------------------------------------- #
#              STEP 4: CREATE DIMENSIONLESS FILE           #
# -------------------------------------------------------- #

dimless_dataset = np.empty_like(raw_dataset)
rho = 1.225  # densidade constante

for i, (_, Re, Uref, _) in enumerate(cases_info):
    Ux = raw_dataset[i, :, :, 0]
    Uy = raw_dataset[i, :, :, 1]
    p  = raw_dataset[i, :, :, 2]

    # Adimensionalizar
    Ux_star = Ux / Uref
    Uy_star = Uy / Uref
    p_star  = p / (0.5 * rho * Uref ** 2)

    # Empilhar de volta
    dimless_dataset[i] = np.stack([Ux_star, Uy_star, p_star], axis=-1)

# Salvar NPY
np.save("../data/dataY-dimensionless.npy", dimless_dataset)

# Salvar CSV
df = pd.DataFrame(dimless_dataset.reshape(-1, 3), columns=["Ux", "Uy", "p"])
df.to_csv("../data/dataY-dimensionless.csv", index=False)

print("✅ Dados adimensionais salvos em '../data/dataY-dimensionless.npy'")
print("✅ CSV exportado para '../data/dataY-dimensionless.csv'")

# -------------------------------------------------------- #
#              STEP 5: CREATE NORMALIZED FILE              #
# -------------------------------------------------------- #

normalized_dataset = np.empty_like(dimless_dataset)
norm_stats = {}

# Normalização Z-score por canal
for i, var in enumerate(["Ux", "Uy", "p"]):
    channel = dimless_dataset[:, :, :, i]

    mean = np.nanmean(channel)
    std  = np.nanstd(channel)

    # Aplica Z-score e preserva NaNs
    normalized = (channel - mean) / std
    normalized_dataset[:, :, :, i] = normalized

    norm_stats[var] = {"mean": float(mean), "std": float(std)}
    print(f"✅ {var} normalizado: média = {mean:.3e}, desvio = {std:.3e}")

# Salva o dataset normalizado
np.save("../data/dataY-normalized.npy", normalized_dataset)

# CSV para visualização
df = pd.DataFrame(normalized_dataset.reshape(-1, 3), columns=["Ux_n", "Uy_n", "p_n"])
df.to_csv("../data/dataY-normalized.csv", index=False)

# Salva as estatísticas de normalização para uso posterior (ex: desnormalização)
with open("../data/dataY-norm-params.json", "w") as f:
    json.dump(norm_stats, f, indent=4)

print("✅ Dataset normalizado (Z-score) salvo em '../data/dataY-normalized.npy'")
print("✅ Estatísticas salvas em '../data/dataY-norm-params.json'")
