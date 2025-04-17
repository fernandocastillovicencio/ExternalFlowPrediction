import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import multiprocessing


# 🔧 CONFIGURAÇÃO: usar versão com log(p)?
usar_versao_log = True  # <<<<< Altere para False para usar versão normal


def _plot_single_case(caso, data, output_dir, variaveis, plot_data_func, tipo_nome):
    unidades_por_tipo = {
        "raw":         {0: "m/s", 1: "m/s", 2: "Pa"},
        "dimensionless": {0: "-",    1: "-",    2: "-"},
        "normalized":    {0: "-",    1: "-",    2: "-"}
    }

    num_variaveis = len(variaveis)
    fig = plt.figure(figsize=(5 * num_variaveis, 12))
    caso_atual = data[caso, :, :, :]

    for j in range(num_variaveis):
        data_to_plot = plot_data_func(caso_atual, j)
        nome_variavel_base = variaveis[j].split(" ")[0]  # "Ux", "Uy", "p"
        unidade = unidades_por_tipo[tipo_nome][j]
        label_com_unidade = f"{nome_variavel_base} ({unidade})" if unidade else nome_variavel_base

        # --- Linha 1: Mapa de calor ---
        plt.subplot(3, num_variaveis, j + 1)
        plt.imshow(data_to_plot.T, cmap="turbo", origin="lower")
        plt.title(f"{label_com_unidade} [{tipo_nome}]")
        plt.colorbar(label=unidade)


        # --- Linha 2: Histograma ---
        plt.subplot(3, num_variaveis, num_variaveis + j + 1)
        sns.histplot(data_to_plot.flatten(), bins=50, kde=True, stat="density")
        plt.title(f"Histograma de {label_com_unidade} [{tipo_nome}]")
        plt.xlabel(f"{label_com_unidade}")

        # --- Linha 3: Boxplot ---
        plt.subplot(3, num_variaveis, 2 * num_variaveis + j + 1)
        sns.boxplot(x=data_to_plot.flatten())
        plt.title(f"Boxplot de {label_com_unidade} [{tipo_nome}]")
        plt.xlabel(f"{label_com_unidade}")

    # 🔹 Título geral da figura (topo da imagem)
    nome_var_principal = variaveis[0].split(" ")[0]  # ex: Ux
    fig.suptitle(f"[{tipo_nome}] field and distribution", fontsize=16)

    # 🔹 Ajustar layout reservando espaço para o título
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # deixa espaço no topo (4% da altura)

    # 🔹 Salvar imagem com o título visível
    case_prefix = f"case{str(caso + 1).zfill(2)}"
    plt.savefig(os.path.join(output_dir, f"{tipo_nome}_{case_prefix}.png"), bbox_inches="tight")
    plt.close()



def _get_raw_data(caso_atual, var_index):
    return caso_atual[:, :, var_index]

def plot_parallel(base_dir, variaveis, plot_data_func, num_processes=None):
    output_dir = "simulations/data_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Arquivos a processar
    datasets = {
        "raw": np.load(f"{base_dir}/dataY-raw.npy"),
        "dimensionless": np.load(f"{base_dir}/dataY-dimensionless.npy"),
        "normalized": np.load(f"{base_dir}/dataY-normalized.npy")
    }

    num_casos = datasets["raw"].shape[0]
    assert all(ds.shape[0] == num_casos for ds in datasets.values()), "⚠️ Número de casos inconsistente entre os arquivos."

    # Geração paralela por tipo de dado
    args = []
    for tipo, data in datasets.items():
        for caso in range(num_casos):
            args.append((caso, data, output_dir, variaveis, plot_data_func, tipo))

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(_plot_single_case, args)

def plot_global_boxplots(base_dir, output_dir="simulations/data_analysis"):
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "raw": np.load(f"{base_dir}/dataY-raw.npy"),
        "dimensionless": np.load(f"{base_dir}/dataY-dimensionless.npy"),
        "normalized": np.load(f"{base_dir}/dataY-normalized.npy")
    }

    variaveis = ["Ux", "Uy", "p"]
    tipos = ["raw", "dimensionless", "normalized"]
    unidades = {
        "raw": ["m/s", "m/s", "Pa"],
        "dimensionless": ["-", "-", "-"],
        "normalized": ["-", "-", "-"]
    }

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

    for i, tipo in enumerate(tipos):
        data = datasets[tipo]
        for j in range(3):
            ax = axes[i, j]
            var_data = data[:, :, :, j].flatten()
            var_data = var_data[~np.isnan(var_data)]  # ignorar NaNs
            sns.boxplot(x=var_data, ax=ax)

            label = f"{variaveis[j]} ({unidades[tipo][j]})"
            ax.set_title(f"Boxplot de {label} [{tipo}]")
            ax.set_xlabel(label)

            # Ajustar limites apenas para a última fileira (normalized)
            # if tipo == "normalized" and j == 2:
            #     ax.set_xlim(-1, 1)

    fig.suptitle("Distribuição Global das Variáveis", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "global_boxplots.png"), bbox_inches="tight")
    plt.close()
    print("✅ Figura global salva em: global_boxplots.png")


def plot_global_histograms(base_dir, output_dir="simulations/data_analysis"):
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "raw": np.load(f"{base_dir}/dataY-raw.npy"),
        "dimensionless": np.load(f"{base_dir}/dataY-dimensionless.npy"),
        "normalized": np.load(f"{base_dir}/dataY-normalized.npy")
    }

    variaveis = ["Ux", "Uy", "p"]
    tipos = ["raw", "dimensionless", "normalized"]
    unidades = {
        "raw": ["m/s", "m/s", "Pa"],
        "dimensionless": ["-", "-", "-"],
        "normalized": ["-", "-", "-"]
    }

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

    for i, tipo in enumerate(tipos):
        data = datasets[tipo]
        for j in range(3):
            ax = axes[i, j]
            var_data = data[:, :, :, j].flatten()
            var_data = var_data[~np.isnan(var_data)]

            sns.histplot(var_data, bins=100, kde=True, stat="density", ax=ax)

            label = f"{variaveis[j]} ({unidades[tipo][j]})"
            ax.set_title(f"Histograma de {label} [{tipo}]")
            ax.set_xlabel(label)
            ax.set_ylabel("Densidade")

    fig.suptitle("Distribuição Global das Variáveis – Histogramas", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "global_histograms.png"), bbox_inches="tight")
    plt.close()
    print("✅ Figura global salva: global_histograms.png")


def plot_analysis_parallel(file_path, num_processes=None):
    variaveis = {0: "Ux (m/s)", 1: "Uy (m/s)", 2: "p (Pa)"}
    plot_parallel(file_path, variaveis, _get_raw_data, "Field", "field", num_processes)


if __name__ == "__main__":
    print("🚀 Executando análise paralela para [raw, dimensionless, normalized]...")
    variaveis = {0: "Ux (m/s)", 1: "Uy (m/s)", 2: "p (Pa)"}
    plot_parallel("simulations/data", variaveis, _get_raw_data, num_processes=None)

    print("📊 Gerando gráfico global de distribuição (boxplot)...")
    plot_global_boxplots("simulations/data")

    print("📊 Gerando histograma global...")
    plot_global_histograms("simulations/data")

    print("✅ Todas as figuras foram geradas.")