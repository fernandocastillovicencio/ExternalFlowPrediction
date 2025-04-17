 
# simulations/scripts/utils_normalization.py

import json
import pathlib

def load_domain_and_velocities(config_path, velocities_path):
    # Carregar parâmetros do domínio
    with config_path.open("r") as f:
        config = json.load(f)
    NX = config["NX"]
    NY = config["NY"]
    XMIN = config["XMIN"]
    XMAX = config["XMAX"]
    YMIN = config["YMIN"]
    YMAX = config["YMAX"]

    # Carregar velocidades
    with velocities_path.open("r") as f:
        velocities = json.load(f, object_pairs_hook=lambda x: {float(k): v for k, v in x})
    sorted_reynolds = sorted(velocities.keys())

    # Print resumido
    print("✅ Configuração do domínio carregada:")
    print(f"  ▸ NX × NY = {NX} × {NY}")
    print(f"  ▸ X range = [{XMIN}, {XMAX}]")
    print(f"  ▸ Y range = [{YMIN}, {YMAX}]")
    print(f"  ▸ Reynolds disponíveis: {sorted_reynolds}")

    return {
        "NX": NX,
        "NY": NY,
        "XMIN": XMIN,
        "XMAX": XMAX,
        "YMIN": YMIN,
        "YMAX": YMAX,
        "velocities": velocities,
        "sorted_reynolds": sorted_reynolds,
    }
