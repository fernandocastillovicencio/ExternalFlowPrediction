import os
import json
import numpy as np

# Carregar configurações do arquivo cloud_config.json
with open("cloud_config.json", "r") as f:
    config = json.load(f)

# Definir constantes
NX = config["NX"]
NY = config["NY"]
X_MIN = config["XMIN"]
X_MAX = config["XMAX"]
Y_MIN = config["YMIN"]
Y_MAX = config["YMAX"]

# Calcular DX e DY
DX = (X_MAX - X_MIN) / NX
DY = (Y_MAX - Y_MIN) / NY

# Criar o cabeçalho do arquivo
header = """/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM: Cloud Dict
\\*---------------------------------------------------------------------------*/"""

# Criar a lista de linhas do arquivo
lines = [
    "FoamFile",
    "{",
    "    version     2.0;",
    "    format      ascii;",
    "    class       dictionary;",
    "    object      cloud;",
    "}",
    "",
    "type                sets;",
    "libs                (sampling);",
    "interpolationScheme cell;",
    "setFormat           raw;",
    "",
    "writeControl        writeTime;",
    "",
    "fields",
    "(",
    "    U",
    "    p",
    ");",
    "",
    "sets",
    "{",
    "    ref_point",
    "    {",
    "        type    cloud;",
    "        axis    xyz;",
    "        points",
    "(",
]

# Adicionar as coordenadas à lista de linhas
for x in np.arange(NX):
    for y in np.arange(NY):
        x_value = X_MIN + DX / 2 + x * DX
        y_value = Y_MIN + DY / 2 + y * DY
        lines.append(f"            ({x_value:.6f} {y_value:.6f} 0)")

# Fechar a lista de linhas
lines.extend(
    [
        ");",
        "    }",
        "}",
    ]
)

# Criar o arquivo
with open(
    os.path.join("..", "template", "postprocessing", "system", "cloud"),
    "w",
) as f:
    f.write(header + "\n")
    f.write("\n".join(lines))

print("✅ Arquivo gerado com sucesso!")
print(f"📄 Número total de pontos gerados: {NX * NY}")
