import pandas as pd
import matplotlib.pyplot as plt
import os

# Função para carregar os dados de cada arquivo CSV em ../data
def carregar_dados_por_Re(diretorio_base):
    """
    Carrega os dados dos arquivos CSV na pasta de dados unificados.
    """
    dados_por_Re = {}
    
    # Verificar se o diretório base existe
    if not os.path.exists(diretorio_base):
        raise FileNotFoundError(f"O diretório {diretorio_base} não foi encontrado.")
    
    # Iterar pelos arquivos na pasta de dados
    for arquivo in os.listdir(diretorio_base):
        caminho_arquivo = os.path.join(diretorio_base, arquivo)
        
        # Verificar se é um arquivo CSV que contém o valor de Reynolds
        if os.path.isfile(caminho_arquivo) and "dataY" in arquivo:
            try:
                # Carregar os dados de cada arquivo CSV
                if "dataY-raw" in arquivo:
                    df_raw = pd.read_csv(caminho_arquivo, na_values=[""])
                    dados_por_Re['raw'] = df_raw
                elif "dataY-dimensionless" in arquivo:
                    df_dim = pd.read_csv(caminho_arquivo, na_values=[""])
                    dados_por_Re['dim'] = df_dim
                elif "dataY-normalized" in arquivo:
                    df_norm = pd.read_csv(caminho_arquivo, na_values=[""])
                    dados_por_Re['norm'] = df_norm
            except Exception as e:
                print(f"Erro ao carregar dados de {arquivo}: {e}")
                
    return dados_por_Re

# Função para plotar boxplots para cada valor de Reynolds
def plotar_boxplots_por_Re(dados_por_Re, output_dir):
    """
    Gera os boxplots para as variáveis Ux, Uy e p para os dados unificados.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # Variáveis de interesse
    variables = ['Ux', 'Uy', 'p']
    
    # Criar boxplots para Ux, Uy e p
    for i, var in enumerate(variables):
        # Verificar se o var está presente em cada tipo de dados (raw, dim, norm)
        if var == 'Ux':
            raw_var = 'Ux'
            dim_var = 'Ux'
            norm_var = 'Ux_n'
        elif var == 'Uy':
            raw_var = 'Uy'
            dim_var = 'Uy'
            norm_var = 'Uy_n'
        elif var == 'p':
            raw_var = 'p'
            dim_var = 'p'
            norm_var = 'p_n'
        
        # Plotar boxplot para cada tipo de dados
        axes[i].boxplot([dados_por_Re['raw'][raw_var].dropna(), dados_por_Re['dim'][dim_var].dropna(), dados_por_Re['norm'][norm_var].dropna()],
                        labels=["Raw", "Dim", "Norm"], vert=False)
        axes[i].set_title(f"Boxplot {var}")
        axes[i].set_xlabel("Value")

    # Ajuste do layout
    plt.tight_layout()

    # Salvar o gráfico
    fig.savefig(os.path.join(output_dir, "boxplot_Reunificados.png"))
    plt.close(fig)

# Corrigir caminho do diretório base para a pasta 'data'
diretorio_base = "../data"  # Agora procuramos os dados na pasta ../data
output_dir = "../data/data_analysis"
os.makedirs(output_dir, exist_ok=True)

# Carregar os dados
dados_por_Re = carregar_dados_por_Re(diretorio_base)

# Gerar e salvar os boxplots
plotar_boxplots_por_Re(dados_por_Re, output_dir)
