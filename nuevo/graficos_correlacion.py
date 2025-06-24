import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap_correlacion(matriz, carpeta_salida, metodo="pearson", tema="bloomberg_dark"):
    plt.figure(figsize=(12, 10))
    cmap = "coolwarm" if tema == "bloomberg_dark" else "YlGnBu"
    sns.set(font_scale=1.2)
    sns.heatmap(matriz, annot=True, fmt=".2f", cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .7})
    plt.title(f"Matriz de Correlación ({metodo.capitalize()})", fontsize=20)
    plt.tight_layout()
    out_path = os.path.join(carpeta_salida, f"heatmap_correlacion_{metodo}_{tema}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"🖼️ Heatmap guardado en {out_path}")
