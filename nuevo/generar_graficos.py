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
