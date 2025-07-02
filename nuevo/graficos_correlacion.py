import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import numpy as np

def plot_clustermap_correlacion(
    matriz,
    carpeta_salida,
    metodo="pearson",
    tema="modern_light",
    mostrar_dendrograma=True
):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    from matplotlib.colors import to_rgb

    plt.close("all")
    # Paleta elegante: azulada, moderna
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    sns.set(font_scale=1.18, style="whitegrid")  # Más pro para print

    # --- Filtrar la matriz para asegurar que está limpia ---
    matriz = matriz.dropna(axis=0, how='all').dropna(axis=1, how='all')
    matriz = matriz.replace([np.inf, -np.inf], np.nan)
    matriz = matriz.dropna(axis=0, how='any').dropna(axis=1, how='any')

    if matriz.shape[0] > 1 and matriz.shape[1] > 1 and np.isfinite(matriz.values).all():
        # Crear el clustermap
        cluster = sns.clustermap(
            matriz,
            method="average",
            metric="euclidean",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 13, "weight": "bold"},
            cmap=cmap,
            center=0,
            linewidths=0.15,
            figsize=(13, 12),
            row_cluster=mostrar_dendrograma,
            col_cluster=mostrar_dendrograma,
            cbar_kws={
                "label": "Correlación",
                "shrink": 0.7,
                "aspect": 24,
                "pad": 0.03
            }
        )

        # Fondo blanco PRO (todo blanco: matriz, dendro y colorbar)
        cluster.ax_heatmap.set_facecolor("white")
        cluster.fig.patch.set_facecolor("white")
        if cluster.ax_row_dendrogram:
            cluster.ax_row_dendrogram.set_facecolor("white")
        if cluster.ax_col_dendrogram:
            cluster.ax_col_dendrogram.set_facecolor("white")
        cluster.cax.set_facecolor("white")

        # Dendrograma y etiquetas en gris elegante
        for ax in [cluster.ax_row_dendrogram, cluster.ax_col_dendrogram]:
            if ax is not None:
                for line in ax.lines:
                    line.set_color("#444")
                    line.set_linewidth(2)
                for spine in ax.spines.values():
                    spine.set_color("#444")
                ax.tick_params(color="#444", labelcolor="#444")
        # Etiquetas de los ticks en gris fuerte
        cluster.ax_heatmap.set_xticklabels(cluster.ax_heatmap.get_xticklabels(), color='#222', fontsize=13, weight="bold")
        cluster.ax_heatmap.set_yticklabels(cluster.ax_heatmap.get_yticklabels(), color='#222', fontsize=13, weight="bold")
        # Etiquetas de la colorbar también en gris
        cluster.cax.yaxis.label.set_color("#444")
        cluster.cax.tick_params(labelcolor="#444", color="#444")

        # Números en negro o blanco según fondo de celda (nunca se pierden)
        for text in cluster.ax_heatmap.texts:
            val = float(text.get_text())
            cell_color = cluster.ax_heatmap.collections[0].cmap(cluster.ax_heatmap.collections[0].norm(val))
            r, g, b = to_rgb(cell_color)
            luminancia = 0.2126*r + 0.7152*g + 0.0722*b
            text.set_color("black" if luminancia > 0.6 else "white")
            text.set_fontweight("bold")

        # Título y subtítulo elegantes
        cluster.fig.suptitle(
            f"Matriz de Correlación Clusterizada ({metodo.capitalize()})",
            fontsize=22, color="#212529", fontweight="bold", y=1.03
        )
        cluster.fig.text(0.5, 0.96, "Análisis financiero automatizado | leocaliva.com", color="#5a5a5a", fontsize=13, ha="center")

        # Colorbar a la derecha, slim
        cluster.cax.set_position([.91, .32, .02, .48])

        plt.subplots_adjust(top=0.92)
        plt.tight_layout()
        out_path = os.path.join(
            carpeta_salida,
            f"clustermap_correlacion_{metodo}_{tema}_PRO.png"
        )
        plt.savefig(out_path, dpi=320, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"🖼️ Clustermap PRO guardado en {out_path}")
    else:
        print("⚠️ No se puede graficar clustermap: la matriz tiene NaN, infinitos o menos de dos activos válidos.")

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

def plot_rolling_correlation_lines(
    df_rolling_correlations,
    carpeta_salida,
    metodo,
    ventana,
    tema,
    pares_a_graficar=None, # Nuevo parámetro para especificar qué pares graficar
    plot_idx=None # Nuevo parámetro para el índice del gráfico
):
    plt.close("all")
    # Mapear el tema de entrada a un estilo de seaborn válido
    if tema == "bloomberg_dark":
        sns.set_style("darkgrid") # Usar un estilo oscuro válido
    elif tema == "modern_light":
        sns.set_style("whitegrid") # Usar un estilo claro válido
    else:
        sns.set_style("dark") # Por defecto, usar \'dark\'
    
    # Si no se especifican pares, graficar todos
    if pares_a_graficar is None:
        df_plot = df_rolling_correlations
        title_suffix = ""
        file_suffix = ""
    else:
        df_plot = df_rolling_correlations[pares_a_graficar]
        if plot_idx is not None and isinstance(plot_idx, int):
            title_suffix = f" (Parte {plot_idx + 1})"
            file_suffix = f"_parte_{plot_idx + 1}"
        elif plot_idx is not None and isinstance(plot_idx, str):
            title_suffix = f" ({plot_idx})"
            file_suffix = f"_{plot_idx}"
        else:
            title_suffix = ""
            file_suffix = ""

    num_pairs = df_plot.shape[1]
    # Ajustar el tamaño de la figura dinámicamente
    fig_height = max(6, num_pairs * 0.8) # Mínimo 6, y crece con el número de pares
    plt.figure(figsize=(15, fig_height))

    for column in df_plot.columns:
        plt.plot(df_plot.index, df_plot[column], label=column)

    plt.title(f"Correlaciones Rolling ({metodo.capitalize()}) - Ventana {ventana} días{title_suffix}", fontsize=16)
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Correlación", fontsize=12)
    plt.legend(title="Pares de Activos", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    out_path = os.path.join(carpeta_salida, f"correlaciones_rolling_lineas_{metodo}_{ventana}d_{tema}{file_suffix}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"🖼️ Gráfico de líneas de correlaciones rolling guardado en {out_path}")




