import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import skew, kurtosis

def generar_grafico_retorno_acumulado(
    ticker, df, tema="normal", carpeta_salida="RetornoDiarioAcumulado",
    logaritmico=False, calcular_rolling=False, ventanas=None
):
    os.makedirs(carpeta_salida, exist_ok=True)
    if ventanas is None:
        ventanas = []

    # --------- Estilos modernos ---------
    if tema == "bloomberg_dark":
        plt.style.use("dark_background")
        sns.set_palette("viridis")
        line_color = "#2EE6FF"
        rolling_color = "#FFD700"
        grid_alpha = 0.13
        title_color = "white"
        legend_face = "#222"
    elif tema == "modern_light":
        plt.style.use("default")
        plt.rcParams["axes.facecolor"] = "white"
        sns.set_palette("Set2")
        line_color = "#2460A7"
        rolling_color = "#63A375"
        grid_alpha = 0.15
        title_color = "#212529"
        legend_face = "white"
    elif tema == "jupyter_quant":
        plt.style.use("default")
        plt.rcParams["axes.facecolor"] = "#f5f5f5"
        sns.set_palette("pastel")
        line_color = "#42b883"
        rolling_color = "#e76f51"
        grid_alpha = 0.10
        title_color = "#212529"
        legend_face = "white"
    elif tema == "nyu_quant":
        plt.style.use("default")
        plt.rcParams["axes.facecolor"] = "#1a1626"
        sns.set_palette("cool")
        line_color = "#907AD6"
        rolling_color = "#F7B32B"
        grid_alpha = 0.12
        title_color = "white"
        legend_face = "#222"
    elif tema == "dark":
        plt.style.use("dark_background")
        sns.set_palette("viridis")
        line_color = "#92B4F4"
        rolling_color = "#F6C90E"
        grid_alpha = 0.13
        title_color = "white"
        legend_face = "#222"
    elif tema == "vintage":
        plt.style.use("classic")
        sns.set_palette("deep")
        line_color = "#13315C"
        rolling_color = "#E9A178"
        grid_alpha = 0.13
        title_color = "#222"
        legend_face = "white"
    elif tema == "modern":
        plt.style.use("ggplot")
        sns.set_palette("coolwarm")
        line_color = "#355C7D"
        rolling_color = "#F67280"
        grid_alpha = 0.14
        title_color = "#222"
        legend_face = "white"
    else:  # "normal"
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("pastel")
        line_color = "#5a7bd7"
        rolling_color = "#fbc531"
        grid_alpha = 0.13
        title_color = "#111"
        legend_face = "white"

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Cumulative_Return"], label="Retorno Acumulado", color=line_color, linewidth=2, alpha=0.9)
    if calcular_rolling:
        for window in ventanas:
            col_name = f"Cumulative_{window}d"
            if col_name in df.columns:
                plt.plot(df.index, df[col_name], label=f"{window} ruedas", linestyle="--", color=rolling_color, alpha=0.85)

    titulo = f"Retorno Diario Acumulado Logarítmico para {ticker}" if logaritmico else f"Retorno Diario Acumulado para {ticker}"
    plt.title(titulo, fontsize=18, color=title_color, fontweight='bold')
    plt.xlabel("Fecha", fontsize=13)
    plt.ylabel("Retorno Acumulado", fontsize=13)
    plt.grid(True, alpha=grid_alpha)
    legend = plt.legend(facecolor=legend_face, edgecolor="#222")
    plt.tight_layout()

    output_path = os.path.join(carpeta_salida, f"{ticker}_retorno_acumulado_{'log' if logaritmico else 'simple'}_{tema}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Gráfico de retorno acumulado para {ticker} ({'logarítmico' if logaritmico else 'simple'}, tema: {tema}) guardado en {output_path}")


def generar_histograma_retorno(
    ticker, df, tema="normal", carpeta_salida="RetornoDiarioAcumulado", bins=60,
    referencias=None
):
    import matplotlib.ticker as mtick
    from scipy.stats import skew, kurtosis

    if referencias is None:
        referencias = {
            "media": True,
            "sigma": True,
            "mediana": True,
            "p1": True,
            "p10": True,
            "p25": True,
            "p75": True,
            "p90": True,
            "p99": True
        }

    os.makedirs(carpeta_salida, exist_ok=True)

    # --------- Estilos modernos ---------
    if tema == "bloomberg_dark":
        plt.style.use("dark_background")
        bar_color = "#2EE6FF"
        media_color = "#FF4DFE"
        sigma_color = "#FFD700"
        mediana_color = "#8AFF6C"
        grid_alpha = 0.17
        title_color = "white"
        legend_face = "#222"
    elif tema == "modern_light":
        plt.style.use("default")
        plt.rcParams["axes.facecolor"] = "white"
        bar_color = "#2460A7"
        media_color = "#FC7F03"
        sigma_color = "#63A375"
        mediana_color = "#C10E70"
        grid_alpha = 0.18
        title_color = "#333"
        legend_face = "white"
    elif tema == "jupyter_quant":
        plt.style.use("default")
        plt.rcParams["axes.facecolor"] = "#f5f5f5"
        bar_color = "#42b883"
        media_color = "#e76f51"
        sigma_color = "#4895ef"
        mediana_color = "#8338ec"
        grid_alpha = 0.14
        title_color = "#212529"
        legend_face = "white"
    elif tema == "nyu_quant":
        plt.style.use("default")
        plt.rcParams["axes.facecolor"] = "#1a1626"
        bar_color = "#907AD6"
        media_color = "#F7B32B"
        sigma_color = "#907AD6"
        mediana_color = "#8AFF6C"
        grid_alpha = 0.12
        title_color = "white"
        legend_face = "#222"
    elif tema == "dark":
        plt.style.use("dark_background")
        bar_color = "#92B4F4"
        media_color = "#FF4DFE"
        sigma_color = "#FFD700"
        mediana_color = "#8AFF6C"
        grid_alpha = 0.13
        title_color = "white"
        legend_face = "#222"
    elif tema == "vintage":
        plt.style.use("classic")
        bar_color = "#13315C"
        media_color = "#E9A178"
        sigma_color = "#F6C90E"
        mediana_color = "#F67280"
        grid_alpha = 0.13
        title_color = "#222"
        legend_face = "white"
    elif tema == "modern":
        plt.style.use("ggplot")
        bar_color = "#355C7D"
        media_color = "#F67280"
        sigma_color = "#FBC531"
        mediana_color = "#26A69A"
        grid_alpha = 0.14
        title_color = "#222"
        legend_face = "white"
    else:  # "normal"
        plt.style.use("seaborn-v0_8-darkgrid")
        bar_color = "#5a7bd7"
        media_color = "orange"
        sigma_color = "gold"
        mediana_color = "green"
        grid_alpha = 0.3
        title_color = "black"
        legend_face = "white"

    returns = df["Daily_Return"].dropna()
    mean = returns.mean()
    std = returns.std()
    sk = skew(returns)
    kurt = kurtosis(returns)
    percentiles = np.percentile(returns, [1, 10, 25, 50, 75, 90, 99])
    N = returns.count()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.hist(
        returns, bins=bins, color=bar_color, alpha=0.82, edgecolor="k", density=True, label="Retornos diarios simples"
    )

    # LÍNEAS y LEYENDA SEGÚN FLAGS y colores
    legend_labels = []
    legend_handles = []

    if referencias.get("media", True):
        ax.axvline(mean, color=media_color, linestyle="--", linewidth=2)
        legend_labels.append(f"Media: {mean:.4f}")
        legend_handles.append(plt.Line2D([0], [0], color=media_color, lw=2, ls="--"))
    if referencias.get("sigma", True):
        ax.axvline(mean + std, color=sigma_color, linestyle="--", linewidth=2)
        ax.axvline(mean - std, color=sigma_color, linestyle="--", linewidth=2)
        legend_labels.append(f"+1σ: {mean+std:.4f}")
        legend_labels.append(f"-1σ: {mean-std:.4f}")
        legend_handles.append(plt.Line2D([0], [0], color=sigma_color, lw=2, ls="--"))
        legend_handles.append(plt.Line2D([0], [0], color=sigma_color, lw=2, ls="--"))
    if referencias.get("mediana", True):
        ax.axvline(percentiles[3], color=mediana_color, linestyle="--", linewidth=2)
        legend_labels.append(f"Mediana: {percentiles[3]:.4f}")
        legend_handles.append(plt.Line2D([0], [0], color=mediana_color, lw=2, ls="--"))

    # Percentiles (usando mismos colores para mantener estilo limpio)
    p_dict = {
        "p1": (percentiles[0], "#FF006E"),
        "p10": (percentiles[1], "#FFBE0B"),
        "p25": (percentiles[2], "#8338ec"),
        "p75": (percentiles[4], "#3A86FF"),
        "p90": (percentiles[5], "#FF006E"),
        "p99": (percentiles[6], "#FF006E"),
    }
    for key, (val, col) in p_dict.items():
        if referencias.get(key, True):
            ax.axvline(val, color=col, linestyle="--", linewidth=2)
            legend_labels.append(f"{key}%: {val:.4f}")
            legend_handles.append(plt.Line2D([0], [0], color=col, lw=2, ls="--"))

    ax.set_title(f"Histograma de Retornos Diarios Simples para {ticker}", fontsize=17, color=title_color, fontweight='bold')
    ax.set_xlabel("Retorno Diario", fontsize=13)
    ax.set_ylabel("Densidad", fontsize=13)
    ax.grid(True, alpha=grid_alpha)

    if legend_labels:
        ax.legend(
            legend_handles, legend_labels,
            fontsize=11, loc="center left", bbox_to_anchor=(1.01, 0.5),
            borderaxespad=1, title="Referencias", facecolor=legend_face
        )

    # --- Estadísticos abajo como antes ---
    stats_line1 = (
        f"μ (media): {mean:.5f}    σ (desv. estándar): {std:.5f}    Skewness: {sk:.2f}    "
        f"Kurtosis: {kurt:.2f}    Mediana: {percentiles[3]:.5f}"
    )
    stats_line2 = (
        f"p1%: {percentiles[0]:.5f}    p10%: {percentiles[1]:.5f}    p25%: {percentiles[2]:.5f}    "
        f"p75%: {percentiles[4]:.5f}    p90%: {percentiles[5]:.5f}    p99%: {percentiles[6]:.5f}    N: {N}"
    )
    fig.subplots_adjust(bottom=0.22, right=0.78)
    fig.lines.extend([plt.Line2D([0.05, 0.95], [0.11, 0.11], color='black', linewidth=1, transform=fig.transFigure, figure=fig)])
    fig.text(0.5, 0.07, stats_line1, ha='center', va='bottom', fontsize=13, family='monospace', fontweight='bold')
    fig.text(0.5, 0.03, stats_line2, ha='center', va='bottom', fontsize=13, family='monospace')

    output_path = os.path.join(
        carpeta_salida, f"{ticker}_histograma_retorno_diario.png"
    )
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Histograma de retornos diarios simples para {ticker} guardado en {output_path}")

# --- NUEVO: Calcular y graficar Drawdown ---
def calcular_drawdown(df):
    df = df.copy()
    df["Acumulado"] = (1 + df["Daily_Return"]).cumprod()
    df["Max_Acumulado"] = df["Acumulado"].cummax()
    df["Drawdown"] = df["Acumulado"] / df["Max_Acumulado"] - 1
    max_drawdown = df["Drawdown"].min()
    fecha_max_drawdown = df["Drawdown"].idxmin()
    return df, max_drawdown, fecha_max_drawdown

def generar_grafico_drawdown(ticker, df, tema="normal", carpeta_salida="RetornoDiarioAcumulado"):
    import matplotlib.dates as mdates

    os.makedirs(carpeta_salida, exist_ok=True)

    # Estilo del gráfico
    if tema == "bloomberg_dark":
        plt.style.use("dark_background")
        color = "#FF4DFE"
        grid_alpha = 0.2
        title_color = "white"
        text_color = "white"
        arrow_color = "white"
    elif tema == "modern_light":
        plt.style.use("default")
        color = "#C10E70"
        grid_alpha = 0.15
        title_color = "#212529"
        text_color = "#212529"
        arrow_color = "#212529"
    else:
        plt.style.use("seaborn-v0_8-darkgrid")
        color = "red"
        grid_alpha = 0.13
        title_color = "black"
        text_color = "black"
        arrow_color = "black"

    plt.figure(figsize=(12, 5))
    ax = df["Drawdown"].plot(color=color)

    # Título y ejes
    plt.title(f"Drawdown - {ticker}", fontsize=16, color=title_color, fontweight="bold")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Fecha")
    plt.grid(True, alpha=grid_alpha)

    # Poner el mínimo drawdown como punto y flecha
    fecha_min = df["Drawdown"].idxmin()
    valor_min = df["Drawdown"].min()

    ax.scatter(fecha_min, valor_min, color=arrow_color, zorder=5)
    ax.annotate(
        f'{valor_min:.2%}\n{fecha_min.strftime("%Y-%m-%d")}',
        xy=(fecha_min, valor_min),
        xytext=(fecha_min, valor_min + 0.1),
        arrowprops=dict(facecolor=arrow_color, arrowstyle="->"),
        fontsize=11,
        color=text_color,
        ha="center"
    )

    output_path = os.path.join(carpeta_salida, f"{ticker}_drawdown_{tema}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"📉 Gráfico de Drawdown para {ticker} guardado en {output_path}")

def exportar_y_graficar_ratios(ratios, carpeta_salida="Ratios", top_n=10, tema="bloomberg_dark"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs(carpeta_salida, exist_ok=True)
    df_ratios = pd.DataFrame(ratios)
    df_ordenado = df_ratios.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)

    # 📁 Excel completo
    path_excel = os.path.join(carpeta_salida, "ratios_completos.xlsx")
    df_ordenado.to_excel(path_excel, index=False)

    # 📊 Imagen con top N
    df_top = df_ordenado.head(top_n).set_index("Ticker")
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(df_top) + 1))
    sns.heatmap(df_top, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True,
                linewidths=0.5, linecolor='gray', ax=ax)
    plt.title("Top Sharpe y Sortino Ratios", fontsize=14, weight="bold")
    plt.tight_layout()

    path_img = os.path.join(carpeta_salida, f"tabla_ratios_top{top_n}.png")
    plt.savefig(path_img, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Ratios guardados en Excel: {path_excel}")
    print(f"🖼️ Imagen guardada: {path_img}")

def generar_qq_plot(ticker, df, tema="normal", carpeta_salida="RetornoDiarioAcumulado"):
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    os.makedirs(carpeta_salida, exist_ok=True)

    if tema == "bloomberg_dark":
        plt.style.use("dark_background")
        title_color = "white"
    elif tema == "modern_light":
        plt.style.use("default")
        title_color = "#212529"
    else:
        plt.style.use("seaborn-v0_8-darkgrid")
        title_color = "black"

    plt.figure(figsize=(6, 6))
    stats.probplot(df["Daily_Return"].dropna(), dist="norm", plot=plt)
    plt.title(f'Q-Q Plot - {ticker}', fontsize=14, color=title_color, fontweight="bold")

    output_path = os.path.join(carpeta_salida, f"{ticker}_qq_plot_{tema}.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"📊 Q-Q Plot guardado para {ticker} en {output_path}")

def test_jarque_bera(df, ticker, carpeta_salida="RetornoDiarioAcumulado"):
    from scipy.stats import jarque_bera
    import os

    jb_stat, jb_p = jarque_bera(df["Daily_Return"].dropna())
    conclusion = (
        "✅ No se rechaza H₀: los retornos podrían ser normales"
        if jb_p > 0.05 else
        "⚠️ Se rechaza H₀: no hay normalidad"
    )

    # Guardar en txt
    os.makedirs(carpeta_salida, exist_ok=True)
    resumen = (
        f"Jarque-Bera test para {ticker}\n"
        f"Estadístico: {jb_stat:.4f}\n"
        f"p-valor: {jb_p:.6f}\n"
        f"Conclusión: {conclusion}\n"
    )
    with open(os.path.join(carpeta_salida, f"{ticker}_jarque_bera.txt"), "w", encoding="utf-8") as f:
        f.write(resumen)


    print(f"📋 Test Jarque-Bera ({ticker}): {conclusion}")
    return jb_stat, jb_p, conclusion

def generar_tabla_jarque_bera_imagen(resultados, carpeta_salida="RetornoDiarioAcumulado", tema="bloomberg_dark"):
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    df = pd.DataFrame(resultados)

    # 🔄 Reemplazar texto por símbolos Unicode compatibles
    df["Normalidad"] = df["Normalidad"].replace({
        "❌ No": "✖ No normal",
        "✅ Sí": "✔ Normal"
    })

    os.makedirs(carpeta_salida, exist_ok=True)

    # Estilo visual
    if tema == "bloomberg_dark":
        plt.style.use("dark_background")
        text_color = "white"
        header_color = "#333"
        cell_color = "#111"
    else:
        plt.style.use("default")
        text_color = "black"
        header_color = "#ddd"
        cell_color = "white"

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(df) + 1))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("gray")
        if i == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color=text_color)
        else:
            cell.set_facecolor(cell_color)
            cell.set_text_props(color=text_color)

    path_img = os.path.join(carpeta_salida, "tabla_jarque_bera.png")
    plt.tight_layout()
    plt.savefig(path_img, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"🧾 Imagen de tabla Jarque-Bera guardada en {path_img}")


def generar_grafico_autocorrelacion(ticker, df, carpeta_salida="RetornoDiarioAcumulado", lags=20, tema="normal"):
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    import os

    os.makedirs(carpeta_salida, exist_ok=True)

    # Estilo
    if tema == "bloomberg_dark":
        plt.style.use("dark_background")
        title_color = "white"
        tick_color = "white"
        grid_color = "#444"
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
        title_color = "black"
        tick_color = "black"
        grid_color = "#ccc"

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_acf(df["Daily_Return"].dropna(), lags=lags, ax=ax, color="skyblue")

    ax.set_title(f'Autocorrelación de retornos - {ticker}', fontsize=14, color=title_color, weight="bold")
    ax.set_xlabel("Retardo (días)", fontsize=12, color=title_color)
    ax.set_ylabel("Coef. de autocorrelación", fontsize=12, color=title_color)

    ax.tick_params(colors=tick_color)
    ax.grid(True, color=grid_color, linestyle="--", alpha=0.4)

    output_path = os.path.join(carpeta_salida, f"{ticker}_autocorrelacion.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"🔁 ACF plot guardado en {output_path}")
