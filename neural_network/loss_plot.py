import matplotlib.pyplot as plt
import os

def plot_loss_curve(train_losses, val_losses, path="neural_network/fig/validation/loss_curve.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss (MSE)")
    plt.title("Curva de Perda")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
