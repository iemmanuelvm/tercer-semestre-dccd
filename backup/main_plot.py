import pandas as pd
import matplotlib.pyplot as plt

# Leer el CSV que generamos al entrenar
df = pd.read_csv("metrics_log.csv")

# ---------- Gráfica MSE ----------
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_mse"], label="Train MSE")
plt.plot(df["epoch"], df["val_mse"], label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE vs Epoch")
plt.legend()
plt.grid(True)
plt.show()

# ---------- Gráfica RMSE ----------
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_rmse"], label="Train RMSE")
plt.plot(df["epoch"], df["val_rmse"], label="Val RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("RMSE vs Epoch")
plt.legend()
plt.grid(True)
plt.show()

# ---------- Gráfica CC (correlación) ----------
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_cc"], label="Train CC")
plt.plot(df["epoch"], df["val_cc"], label="Val CC")
plt.xlabel("Epoch")
plt.ylabel("CC")
plt.title("Correlation Coefficient vs Epoch")
plt.legend()
plt.grid(True)
plt.show()
