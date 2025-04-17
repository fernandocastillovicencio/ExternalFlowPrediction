import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Parâmetros
apply_zscore = True
data_dir = "simulations/data"
output_dir = "simulations/data_analysis"
os.makedirs(output_dir, exist_ok=True)

# Carregar arquivo adimensionalizado
data_dimless = np.load(os.path.join(data_dir, "dataY-dimensionless.npy"))
p_orig = data_dimless[:, :, :, 2]
p_flat = p_orig.flatten()
p_flat = p_flat[~np.isnan(p_flat)]

# 🔹 Transformação logarítmica simétrica
def log_symmetric(x):
    return np.sign(x) * np.log1p(np.abs(x))

def log_symmetric_inverse(x):
    return np.sign(x) * (np.expm1(np.abs(x)))

# Aplicar no campo de pressão
p_log = log_symmetric(p_orig)

# Salvar versão com log apenas
data_log = data_dimless.copy()
data_log[:, :, :, 2] = p_log
np.save(os.path.join(data_dir, "dataY-logscaled.npy"), data_log)
print("✅ Arquivo salvo: dataY-logscaled.npy")

# 🔹 Opcional: aplicar Z-score após log
if apply_zscore:
    p_flat_log = p_log.flatten()
    p_flat_log = p_flat_log[~np.isnan(p_flat_log)]
    mu = np.mean(p_flat_log)
    sigma = np.std(p_flat_log)

    print(f"🔍 Z-score após log: μ = {mu:.4f}, σ = {sigma:.4f}")

    p_log_norm = (p_log - mu) / sigma
    data_log_norm = data_dimless.copy()
    data_log_norm[:, :, :, 2] = p_log_norm
    np.save(os.path.join(data_dir, "dataY-normalized-log.npy"), data_log_norm)
    print("✅ Arquivo salvo: dataY-normalized-log.npy")

# 🔹 Plot comparativo
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Boxplot
sns.boxplot(x=p_flat, ax=axes[0], color="skyblue")
sns.boxplot(x=log_symmetric(p_flat), ax=axes[0], color="orange")
if apply_zscore:
    sns.boxplot(x=((log_symmetric(p_flat) - mu) / sigma), ax=axes[0], color="green")
axes[0].set_title("Boxplot da pressão (original, log, log+zscore)")
axes[0].set_xlabel("p (-)")

# Legenda manual
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="skyblue", label="Original"),
    Patch(facecolor="orange", label="Log-scaled"),
    Patch(facecolor="green", label="Log + Z-score")
]
axes[0].legend(handles=legend_elements)

# Histograma
sns.histplot(p_flat, bins=100, kde=True, stat="density", ax=axes[1], color="skyblue")
sns.histplot(log_symmetric(p_flat), bins=100, kde=True, stat="density", ax=axes[1], color="orange")
if apply_zscore:
    sns.histplot(((log_symmetric(p_flat) - mu) / sigma), bins=100, kde=True, stat="density", ax=axes[1], color="green")

axes[1].set_title("Histograma da pressão (original, log, log+zscore)")
axes[1].set_xlabel("p (-)")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pressure_logscale_comparison.png"), bbox_inches="tight")
plt.close()

print("📊 Figura gerada: pressure_logscale_comparison.png")
