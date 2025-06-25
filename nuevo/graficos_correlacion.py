import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

def plot_clustermap_correlacion(
    matriz,
    carpeta_salida,
    metodo="pearson",
    tema="bloomberg_dark",
    mostrar_dendrograma=True
):
    plt.close("all")
    cmap = "coolwarm" if tema == "bloomberg_dark" else "YlGnBu"
    sns.set(font_scale=1.1)

    cluster = sns.clustermap(
        matriz,
        method="average",
        metric="euclidean",
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        figsize=(13, 11),
        row_cluster=mostrar_dendrograma,
        col_cluster=mostrar_dendrograma
    )

    cluster.fig.suptitle(
        f"Clustermap de Correlación ({metodo.capitalize()})",
        fontsize=19,
        y=1.03
    )

    # Si NO querés dendrogramas, ocultá las ramas y reubicá los ejes y el colorbar
    if not mostrar_dendrograma:
        cluster.ax_row_dendrogram.set_visible(False)
        cluster.ax_col_dendrogram.set_visible(False)
        cluster.ax_heatmap.set_position([0.13, 0.1, 0.65, 0.8])  # [izq, abajo, ancho, alto]
        cluster.cax.set_position([.82, .3, .03, .4])  # [izq, abajo, ancho, alto]

    plt.tight_layout()
    out_path = os.path.join(
        carpeta_salida,
        f"clustermap_correlacion_{metodo}_{tema}{'_sin_dendrograma' if not mostrar_dendrograma else ''}.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🖼️ Clustermap guardado en {out_path} (dendrogramas={'Sí' if mostrar_dendrograma else 'No'})")

def plot_clustered_heatmap_sin_dendrograma(
    matriz,
    carpeta_salida,
    metodo="pearson",
    tema="bloomberg_dark"
):
    # Calculá linkage (agrupamiento) para el orden óptimo de filas/columnas
    linkage = sch.linkage(matriz, method="average")
    dendro = sch.dendrogram(linkage, labels=matriz.index, no_plot=True)
    idx = dendro["leaves"]

    # Reordená la matriz según el clustering
    matriz_reorder = matriz.iloc[idx, :].iloc[:, idx]

    plt.figure(figsize=(12, 10))
    cmap = "coolwarm" if tema == "bloomberg_dark" else "YlGnBu"
    sns.set(font_scale=1.2)
    ax = sns.heatmap(
        matriz_reorder,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .7, "location": "right"}
    )
    plt.title(f"Matriz de Correlación Clusterizada ({metodo.capitalize()})", fontsize=20)
    plt.tight_layout()
    out_path = os.path.join(carpeta_salida, f"heatmap_clusterizado_{metodo}_{tema}_sin_dendrograma.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"🖼️ Heatmap clusterizado SIN dendrograma guardado en {out_path}")
