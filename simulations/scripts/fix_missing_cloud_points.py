import os
import numpy as np
import json


def get_latest_time(case_dir):
    cloud_dir = os.path.join(case_dir, "postProcessing", "cloud")
    time_dirs = [d for d in os.listdir(cloud_dir) if d.isdigit()]
    if not time_dirs:
        print("ERRO: Nenhuma pasta de tempo encontrada em postProcessing/cloud/")
        return None
    latest_time = max(map(int, time_dirs))
    return latest_time


def fix_missing_cloud_points(case_dir):
    latest_time = get_latest_time(case_dir)
    if latest_time is None:
        return

    file_path = os.path.join(
        case_dir, "postProcessing", "cloud", str(latest_time), "ref_point_p_U.xy"
    )
    if not os.path.exists(file_path):
        print(f"ERRO: Arquivo {file_path} não encontrado!")
        return

    print(f"[DEBUG] Processando arquivo: {file_path}")

    # Carregar dados existentes
    data = np.loadtxt(file_path)

    if data.ndim == 1:
        data = data.reshape(1, -1)  # Garantir formato 2D

    # Extração das colunas x, y, z, p, Ux, Uy, Uz
    x_values = np.round(data[:, 0], 6)
    y_values = np.round(data[:, 1], 6)

    # Carregar configurações
    with open("cloud_config.json") as f:
        config = json.load(f)

    NX = config["NX"]
    NY = config["NY"]
    XMIN = config["XMIN"]
    XMAX = config["XMAX"]
    YMIN = config["YMIN"]
    YMAX = config["YMAX"]
    DX = (XMAX - XMIN) / NX
    DY = (YMAX - YMIN) / NY
    NPOINTS = NX * NY

    # Criar listas dos pontos esperados
    expected_x = np.round([XMIN + (2 * i + 1) / 2 * DX for i in range(NX)], 6)
    expected_y = np.round([YMIN + (2 * i + 1) / 2 * DY for i in range(NY)], 6)

    # Criar um conjunto de pares existentes (x, y)
    existing_points = set(zip(x_values, y_values))

    # Verificar quais pontos estão faltando
    missing_points = []
    for x in expected_x:
        for y in expected_y:
            if (x, y) not in existing_points:
                missing_points.append(
                    [x, y, 0.0, 0.0, 0.0, 0.0, 0.0]
                )  # Adicionar linha vazia

    if missing_points:
        print(f"[DEBUG] Adicionando {len(missing_points)} pontos ausentes...")

        # Concatenar os dados existentes com os pontos ausentes
        corrected_data = np.vstack((data, missing_points))
        corrected_data = corrected_data[
            np.lexsort((corrected_data[:, 1], corrected_data[:, 0]))
        ]  # Ordenação por x e y

        # Salvar o arquivo corrigido
        np.savetxt(file_path, corrected_data, fmt="%.6e", delimiter=" ")

        print(f"[DEBUG] Arquivo corrigido e salvo: {file_path}")

    else:
        print(
            "[DEBUG] Nenhum ponto ausente encontrado. Nenhuma modificação necessária."
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("ERRO: Número de argumentos inválido!")
        sys.exit(1)

    case_dir = sys.argv[1]
    fix_missing_cloud_points(case_dir)
