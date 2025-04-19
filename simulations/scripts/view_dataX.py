import pandas as pd

df = pd.read_csv("simulations/data/flow_mask_case00.csv", header=None)
print(df.shape)  # deve mostrar (100, 100)