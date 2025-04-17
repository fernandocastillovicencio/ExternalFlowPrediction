import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Diretórios
data_dir = "simulations/data"
output_dir = "simulations/data_analysis"
os.makedirs(output_dir, exist_ok=True)

# Carregar dataset normalizado original
data_norm = np.load(os.path.join(data_dir, "dataY-normalized.npy"))

# Extrair canal da pressão (índice 2)
p_all = data_norm[:, :, :, 2]
p_flat = p_all.flatten()
p_flat = p_flat[~np.isnan(p_flat)]

# Parâmetros de clipping
pmin = np.percentile(p_flat, 0.5)
pmax = np.percentile(p_flat, 99.5)

print(f"🔍 Clipping pressure: [{pmin:.3f}, {pmax:.3f}] (percentis 0.5%–99.5%)")

# Aplicar clipping
p_clipped = np.clip(p_all, pmin, pmax)

# Substituir no tensor original e salvar
data_clipped = data_norm.copy()
data_clipped[:, :, :, 2] = p_clipped
np.save(os.path.join(data_dir, "dataY-normalized-clipped.npy"), data_clipped)
print("✅ Arquivo salvo: dataY-normalized-clipped.npy")

# 🔹 Comparação visual (boxplot + histograma)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# Boxplot
sns.boxplot(x=p_flat, ax=axes[0], color="skyblue")
sns.boxplot(x=p_clipped.flatten(), ax=axes[0], color="orange")
axes[0].set_title("Boxplot da pressão normalizada (original vs clipped)")
axes[0].set_xlabel("p (-)")

# Adiciona legenda manualmente
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='skyblue', label='Original'),
    Patch(facecolor='orange', label='Clipped')
]
axes[0].legend(handles=legend_elements)

axes[0].set_title("Boxplot da pressão normalizada (original vs clipped)")
axes[0].set_xlabel("p (-)")

# Histograma
sns.histplot(p_flat, bins=100, kde=True, stat="density", ax=axes[1], color="skyblue", label="Original")
sns.histplot(p_clipped.flatten(), bins=100, kde=True, stat="density", ax=axes[1], color="orange", label="Clipped")
axes[1].set_title("Histograma da pressão normalizada (original vs clipped)")
axes[1].set_xlabel("p (-)")
axes[1].legend()

# Layout e salvar
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pressure_clipping_comparison.png"), bbox_inches="tight")
plt.close()

print("📊 Figura gerada: pressure_clipping_comparison.png")
