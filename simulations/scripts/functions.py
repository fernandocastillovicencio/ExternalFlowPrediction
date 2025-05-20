from numpy import arctan2
import numpy as np


def detectar_frontal_e_esteira(region_mask, x, y, flow_mask):
    """
    Divide a camada limite (region_mask == 2) com base no valor de y no ponto mais à esquerda do obstáculo.
    y < y_E → região inferior (3), y >= y_E → região superior (4)
    """
    x_obs = x[flow_mask == 0]
    y_obs = y[flow_mask == 0]

    if len(x_obs) == 0:
        return region_mask

    x_E = np.min(x_obs)
    y_E = np.median(y_obs[x_obs == x_E])

    region_mask[(region_mask == 2) & (y < y_E)] = 3
    region_mask[(region_mask == 2) & (y >= y_E)] = 4

    return region_mask


# -------------------------------------------------------- #


def calculo_da_inflexao(region_mask, x, y, flow_mask):

    idx_obs = np.where(flow_mask == 0)[0]
    if len(idx_obs) == 0:
        print("Nenhum obstáculo encontrado.")
        return

    x_obs = x[idx_obs]
    y_obs = y[idx_obs]

    x_E = np.min(x_obs)
    y_E = np.median(y_obs[x_obs == x_E])

    idx_sup = idx_obs[y_obs >= y_E]
    idx_inf = idx_obs[y_obs < y_E]

    # Camada superior — para cada x, pega o ponto com maior y
    borda_sup = {}
    for i in idx_sup:
        xi, yi = x[i], y[i]
        if xi not in borda_sup:
            borda_sup[xi] = yi
        else:
            if yi > borda_sup[xi]:
                borda_sup[xi] = yi

    # Camada inferior — para cada x, pega o ponto com menor y
    borda_inf = {}
    for i in idx_inf:
        xi, yi = x[i], y[i]
        if xi not in borda_inf:
            borda_inf[xi] = yi
        else:
            if yi < borda_inf[xi]:
                borda_inf[xi] = yi

    print("\n🔵 Borda superior do obstáculo (x, y_max):")
    for xi in sorted(borda_sup.keys()):
        print(f"x={xi:.4f}, y={borda_sup[xi]:.4f}")

    print("\n🟠 Borda inferior do obstáculo (x, y_min):")
    for xi in sorted(borda_inf.keys()):
        print(f"x={xi:.4f}, y={borda_inf[xi]:.4f}")

    return borda_sup, borda_inf


# -------------------------------------------------------- #
def detectar_x_inflexao(borda, label):
    x_sorted = sorted(borda.keys())
    pontos = []
    for i in range(len(x_sorted) - 1):
        x0, y0 = x_sorted[i], borda[x_sorted[i]]
        x1, y1 = x_sorted[i + 1], borda[x_sorted[i + 1]]
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0:
            ang = 90.0 if dy > 0 else -90.0
        else:
            ang = np.degrees(np.arctan(dy / dx))

        # Ignora ângulos muito próximos de zero
        if np.isclose(ang, 0.0, atol=1e-6):
            continue

        pontos.append((x0, y0, ang))

    print(f"\n🔁 Pontos com mudança de sinal na borda {label}:")
    for i in range(1, len(pontos)):
        ang1 = pontos[i - 1][2]
        ang2 = pontos[i][2]
        if ang1 * ang2 < 0:  # mudança de sinal
            xA, yA = pontos[i - 1][0], pontos[i - 1][1]
            xB, yB = pontos[i][0], pontos[i][1]
            x_inflexao = (xA + xB) / 2.0
            print(f"↔ Mudança entre:")
            print(f"   x={xA:.4f}, y={yA:.4f}, ang={ang1:+.2f}°")
            print(f"   x={xB:.4f}, y={yB:.4f}, ang={ang2:+.2f}°")
            print(f"⭐ Ponto de inflexão estimado em x = {x_inflexao:.4f}")

    return x_inflexao


def aplicar_frontal_por_inflexao(
    region_mask, x, y, flow_mask, x_inflexao_inf, x_inflexao_sup
):
    """
    Define a parte frontal da camada limite com base no valor de x de inflexão.
    - region_mask == 3: camada limite inferior até inflexão
    - region_mask == 4: camada limite superior até inflexão
    """
    idx_obs = np.where(flow_mask == 0)[0]
    if len(idx_obs) == 0:
        print("Nenhum obstáculo encontrado.")
        return region_mask

    x_obs = x[idx_obs]
    y_obs = y[idx_obs]
    x_E = np.min(x_obs)
    y_E = np.median(y_obs[x_obs == x_E])

    # # Região inferior (abaixo do eixo de simetria y_E)
    region_mask[(region_mask == 2) & (y < y_E) & (x > x_inflexao_inf)] = 3

    # # Região superior (acima do eixo de simetria y_E)
    region_mask[(region_mask == 2) & (y >= y_E) & (x > x_inflexao_sup)] = 3

    return region_mask
