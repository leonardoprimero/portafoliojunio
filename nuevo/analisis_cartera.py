import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patheffects as pe
from scipy.optimize import minimize

def markowitz_simulacion(
    tickers,
    carpeta_datos_limpios="DatosLimpios",
    n_iter=10000,
    carpeta_salida="Montecarlo",
    tema="bloomberg_dark"  # <- Elegís el tema
):
    os.makedirs(carpeta_salida, exist_ok=True)

    # ========== Temas profesionales ==========
    temas = {
        "bloomberg_dark": {
            "style": "dark_background",
            "colormap": "viridis",
            "frontera_color": "#FF4DFE",
            "activos_color": "tab:blue",
            "optimo_color": "#00E6FB",
            "watermark": {"color": "#F2F2F2", "alpha": 0.10, "fontsize": 62},  # gris muy claro
            "label": "leocaliva.com"
        },
        "modern_light": {
            "style": "default",
            "colormap": "cubehelix",   #  rainbow, plasma, viridis, cividis, magma, inferno, coolwarm, cubehelix, Spectral, twilight
            "frontera_color": "#2574A9",
            "activos_color": "#20639B",
            "optimo_color": "#00b8d9",
            "watermark": {"color": "#222", "alpha": 0.08, "fontsize": 48},
            "label": "leocaliva.com"
        },
        "nyu_quant": {
            "style": "default",
            "colormap": "cividis",
            "frontera_color": "#907AD6",
            "activos_color": "#F7B32B",
            "optimo_color": "#907AD6",
            "watermark": {"color": "#111", "alpha": 0.09, "fontsize": 54},
            "label": "leocaliva.com"
        },
        "classic_white": {
            "style": "seaborn-v0_8-whitegrid",
            "colormap": "coolwarm",
            "frontera_color": "#212529",
            "activos_color": "#0066CC",
            "optimo_color": "#C10E70",
            "watermark": {"color": "#B0B0B0", "alpha": 0.10, "fontsize": 45},
            "label": "leocaliva.com"
        }
    }
    tema_cfg = temas.get(tema, temas["bloomberg_dark"])
    plt.style.use(tema_cfg["style"])
    if tema == "modern_light":
        plt.rcParams["axes.facecolor"] = "#f0f4fa"
        plt.rcParams["figure.facecolor"] = "#f0f4fa"

    # ========== Carga de datos ==========
    retornos = []
    for ticker in tickers:
        path = None
        for root, dirs, files in os.walk(carpeta_datos_limpios):
            for file in files:
                if file.lower().startswith(ticker.lower()) and file.endswith('.csv'):
                    path = os.path.join(root, file)
        if path is None:
            print(f"⚠️ No se encontró archivo limpio para {ticker}")
            continue
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        retornos.append(np.log(df["Close"] / df["Close"].shift(1)).rename(ticker))
    data = pd.concat(retornos, axis=1).dropna()

    mean_returns = data.mean() * 252
    cov_matrix = data.cov() * 252
    num_assets = len(tickers)

    # ========== Simulación Monte Carlo ==========
    carteras = []
    for i in range(n_iter):
        pesos = np.random.random(num_assets)
        pesos /= pesos.sum()
        ret = np.sum(mean_returns * pesos)
        vol = np.sqrt(np.dot(pesos, np.dot(cov_matrix, pesos)))
        sharpe = ret / vol if vol != 0 else np.nan
        carteras.append({
            'retorno': ret,
            'volatilidad': vol,
            'sharpe': sharpe,
            'pesos': pesos.copy()
        })
    carteras = pd.DataFrame(carteras)

    # ========== Activos individuales ==========
    datosTickers = []
    for ticker in data.columns:
        ret = mean_returns[ticker]
        vol = data[ticker].std() * np.sqrt(252)
        sharpe = ret / vol if vol != 0 else np.nan
        datosTickers.append({'ticker': ticker, 'retorno': ret, 'volatilidad': vol, 'sharpe': sharpe})
    datosTickers = pd.DataFrame(datosTickers).set_index('ticker')

    # ========== Portafolio óptimo ==========
    idx_max = carteras.sharpe.idxmax()
    optimo = carteras.loc[idx_max]
    mejor_port = carteras.iloc[idx_max]['pesos']
    datosTickers['ponderacion_optima'] = mejor_port

    # ========== Frontera eficiente teórica ==========
    def portafolio_stats(pesos):
        ret = np.dot(mean_returns, pesos)
        vol = np.sqrt(np.dot(pesos, np.dot(cov_matrix, pesos)))
        return np.array([ret, vol, ret / vol if vol != 0 else np.nan])

    fronterax, fronteray = [], []
    targets = np.linspace(carteras.retorno.min(), carteras.retorno.max(), 100)
    for target in targets:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(mean_returns, x) - target}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        result = minimize(
            lambda x: portafolio_stats(x)[1],
            num_assets * [1. / num_assets,],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            fronterax.append(result.fun)
            fronteray.append(target)
        else:
            fronterax.append(np.nan)
            fronteray.append(np.nan)

    # ========== Gráfico ==========

    fig, ax = plt.subplots(figsize=(10, 7))
    
    if tema == "modern_light":
        ax.set_facecolor("#d8e3ec")
        fig.patch.set_facecolor("white")

    # Marca de agua visible en todos los temas, especialmente en Bloomberg dark
    plt.text(
        0.5, 0.54, tema_cfg["label"],
        fontsize=tema_cfg["watermark"]["fontsize"],
        color=tema_cfg["watermark"]["color"],
        alpha=tema_cfg["watermark"]["alpha"],
        ha='center', va='center',
        transform=ax.transAxes,
        weight='bold', zorder=0
    )

    # Nube de carteras
    scatter = ax.scatter(
        carteras.volatilidad, carteras.retorno, 
        c=carteras.sharpe, s=10, cmap=tema_cfg["colormap"], alpha=0.57, zorder=2
    )
    plt.colorbar(scatter, label='Sharpe Ratio', aspect=40)

    # Frontera eficiente
    ax.plot(fronterax, fronteray, '--', color=tema_cfg["frontera_color"], lw=2.4, label='Frontera eficiente', zorder=4)

    # ======= Portafolio óptimo (círculo y anotación auto-ajustada) =======
    opt_circle_color = tema_cfg["optimo_color"]
    # Ajuste dinámico para anotación:
    align_h = 'left' if optimo.volatilidad < (carteras.volatilidad.max() * 0.85) else 'right'
    align_v = 'bottom' if optimo.retorno > (carteras.retorno.min() + (carteras.retorno.max() - carteras.retorno.min())*0.35) else 'top'
    offset_x = 0.01 if align_h == 'left' else -0.012
    offset_y = 0.014 if align_v == 'bottom' else -0.016

    # Círculo óptimo: cyan claro en dark, pro en otros temas, borde blanco, tamaño más chico
    ax.scatter(
        optimo.volatilidad, optimo.retorno,
        c=opt_circle_color, alpha=0.36, s=420, label='Óptimo',
        zorder=6, edgecolor="white", linewidth=2.2
    )

    ax.text(
        optimo.volatilidad + offset_x,
        optimo.retorno + offset_y,
        f'Óptimo\nSharpe={optimo.sharpe:.2f}\nR={optimo.retorno:.2%}\nσ={optimo.volatilidad:.2%}',
        fontsize=10, c='black', ha=align_h, va=align_v, zorder=7, weight='bold',
        bbox=dict(facecolor='white', alpha=0.92, edgecolor='#CCCCCC', boxstyle='round,pad=0.35')
    )

    # Activos individuales
    for ticker in data.columns:
        vol = datosTickers.loc[ticker, 'volatilidad']
        ret = datosTickers.loc[ticker, 'retorno']
        ax.scatter(
            vol, ret, c=tema_cfg["activos_color"], s=295, zorder=8, edgecolor='white', linewidth=1.9
        )
        ax.text(
            vol, ret, ticker, fontsize=13, c='white', ha='center', va='center', zorder=9, weight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground="black")]
        )

    # Ejes en porcentaje
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Leyendas, título
    ax.legend(fontsize=12, loc='upper left')
    plt.xlabel('Volatilidad Anual', fontsize=14)
    plt.ylabel('Retorno Anual', fontsize=14)
    plt.title('Frontera eficiente (teórica y simulada) con activos individuales', fontsize=16, weight="bold")
    plt.grid(True, alpha=0.23)
    plt.tight_layout()

    out_path = os.path.join(carpeta_salida, f"frontera_montecarlo_{n_iter}iter_{tema}.png")
    plt.savefig(out_path, dpi=320)
    plt.close()
    print(f"🖼️ Gráfico frontera eficiente guardado en {out_path}")

    # ========== Exportar resultados ==========
    carteras_export = carteras.copy()
    for i, ticker in enumerate(data.columns):
        carteras_export[f"W_{ticker}"] = carteras_export['pesos'].apply(lambda x: x[i])
    carteras_export.drop(columns=['pesos'], inplace=True)
    excel_carteras = os.path.join(carpeta_salida, f"montecarlo_portafolios_{n_iter}iter_{tema}.xlsx")
    carteras_export.to_excel(excel_carteras, index=False)
    print(f"✅ Resultados de Monte Carlo exportados a {excel_carteras}")

    excel_activos = os.path.join(carpeta_salida, f"activos_individuales_{tema}.xlsx")
    datosTickers.round(5).to_excel(excel_activos)
    print(f"✅ Estadísticas de activos individuales exportadas a {excel_activos}")

    # Portafolio óptimo por consola
    print("\n📈 Portafolio de Sharpe máximo:")
    for t, w in zip(data.columns, mejor_port):
        print(f"   {t}: {w:.2%}")
    print(f"Retorno esperado: {optimo.retorno:.2%}")
    print(f"Riesgo: {optimo.volatilidad:.2%}")
    print(f"Sharpe Ratio: {optimo.sharpe:.2f}")

    return datosTickers.round(5), optimo
