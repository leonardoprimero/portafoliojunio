import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patheffects as pe
from scipy.optimize import minimize
from rich.progress import Progress



def markowitz_simulacion(
    tickers,
    carpeta_datos_limpios="DatosLimpios",
    n_iter=10000,
    carpeta_salida="Montecarlo",
    tema="bloomberg_dark",
    capital_usd=100_000,
    peso_min=0.05,
    peso_max=0.25,
    usar_benchmark=False,
    benchmark_ticker="SPY",
    benchmark_como_activo=False
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
    
    if usar_benchmark and not benchmark_como_activo and benchmark_ticker in tickers:
        cartera_tickers = [t for t in tickers if t != benchmark_ticker]
    else:
        cartera_tickers = tickers

    mean_returns = data.mean() * 252
    cov_matrix = data.cov() * 252
    num_assets = len(cartera_tickers)
    # ========== Simulación Monte Carlo con restricciones ==========
    carteras = []
    print(f"⏳ Simulación Monte Carlo ({n_iter} iteraciones)...")
    for i in range(n_iter):
        # Generación de pesos con límites
        for _ in range(500):  # Evita loops infinitos si no hay solución posible
            pesos = np.random.dirichlet(np.ones(num_assets), 1)[0]
            if np.all(pesos >= peso_min) and np.all(pesos <= peso_max):
                break
        else:
            # Si después de 500 intentos no consigue, usá pesos iguales
            pesos = np.ones(num_assets) / num_assets

        ret = np.sum(mean_returns[cartera_tickers] * pesos)
        vol = np.sqrt(np.dot(pesos, np.dot(cov_matrix.loc[cartera_tickers, cartera_tickers], pesos)))
        sharpe = ret / vol if vol != 0 else np.nan
        carteras.append({
            'retorno': ret,
            'volatilidad': vol,
            'sharpe': sharpe,
            'pesos': pesos.copy()
        })
        if n_iter >= 10 and i % (n_iter // 10) == 0 and i > 0:
            print(f"   → Monte Carlo: {i}/{n_iter} ({100*i//n_iter}%)")
    print("✅ Simulación Monte Carlo finalizada.")

    carteras = pd.DataFrame(carteras)

    # ========== Activos individuales ==========
    datosTickers = []
    for ticker in data.columns:
        ret = mean_returns[ticker]
        vol = data[ticker].std() * np.sqrt(252)
        sharpe = ret / vol if vol != 0 else np.nan
        datosTickers.append({'ticker': ticker, 'retorno': ret, 'volatilidad': vol, 'sharpe': sharpe})
    datosTickers = pd.DataFrame(datosTickers).set_index('ticker')

    # ========== Portafolios destacados ==========
    idx_max_sharpe = carteras.sharpe.idxmax()
    idx_max_ret = carteras.retorno.idxmax()
    idx_min_risk = carteras.volatilidad.idxmin()
    port_opt = carteras.loc[idx_max_sharpe]
    port_ret = carteras.loc[idx_max_ret]
    port_risk = carteras.loc[idx_min_risk]
    nombres = ["Sharpe óptimo", "Mayor retorno", "Menor riesgo"]
    indices = [idx_max_sharpe, idx_max_ret, idx_min_risk]
    carteras_top = [port_opt, port_ret, port_risk]

    # ========== Frontera eficiente teórica ==========
    def portafolio_stats(pesos):
        ret = np.dot(mean_returns[cartera_tickers], pesos)
        vol = np.sqrt(np.dot(pesos, np.dot(cov_matrix.loc[cartera_tickers, cartera_tickers], pesos)))
        return np.array([ret, vol, ret / vol if vol != 0 else np.nan])

    fronterax, fronteray = [], []
    targets = np.linspace(carteras.retorno.min(), carteras.retorno.max(), 100)
    for target in targets:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(mean_returns, x) - target}
        )
        bounds = tuple((peso_min, peso_max) for _ in range(num_assets))
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
    align_h = 'left' if port_opt.volatilidad < (carteras.volatilidad.max() * 0.85) else 'right'
    align_v = 'bottom' if port_opt.retorno > (carteras.retorno.min() + (carteras.retorno.max() - carteras.retorno.min())*0.35) else 'top'
    offset_x = 0.01 if align_h == 'left' else -0.012
    offset_y = 0.014 if align_v == 'bottom' else -0.016

    # Círculo óptimo: cyan claro en dark, pro en otros temas, borde blanco, tamaño más chico
    ax.scatter(
        port_opt.volatilidad, port_opt.retorno,
        c=opt_circle_color, alpha=0.36, s=420, label='Óptimo',
        zorder=6, edgecolor="white", linewidth=2.2
    )

    ax.text(
        port_opt.volatilidad + offset_x,
        port_opt.retorno + offset_y,
        f'Óptimo\nSharpe={port_opt.sharpe:.2f}\nR={port_opt.retorno:.2%}\nσ={port_opt.volatilidad:.2%}',
        fontsize=10, c='black', ha=align_h, va=align_v, zorder=7, weight='bold',
        bbox=dict(facecolor='white', alpha=0.92, edgecolor='#CCCCCC', boxstyle='round,pad=0.35')
    )

    # Activos individuales (portafolio)
    for ticker in cartera_tickers:
        vol = datosTickers.loc[ticker, 'volatilidad']
        ret = datosTickers.loc[ticker, 'retorno']
        ax.scatter(
            vol, ret, c=tema_cfg["activos_color"], s=295, zorder=8, edgecolor='white', linewidth=1.9
        )
        ax.text(
            vol, ret, ticker, fontsize=13, c='white', ha='center', va='center', zorder=9, weight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground="black")]
        )

    # Benchmark destacado si corresponde
    if usar_benchmark and benchmark_ticker in datosTickers.index:
        vol_bench = datosTickers.loc[benchmark_ticker, 'volatilidad']
        ret_bench = datosTickers.loc[benchmark_ticker, 'retorno']
        ax.scatter(
            vol_bench, ret_bench,
            c='#FFB800', s=400, zorder=10, edgecolor='black', linewidth=2, marker="D", label=f"Benchmark: {benchmark_ticker}"
        )
        ax.text(
            vol_bench, ret_bench,
            f"{benchmark_ticker} (benchmark)",
            fontsize=12, c='#FFB800', ha='center', va='bottom', weight='bold',
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")]
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

    # ========== Exportar resultados de todos los portafolios ==========
    carteras_export = carteras.copy()
    for i, ticker in enumerate(cartera_tickers):
        carteras_export[f"W_{ticker}"] = carteras_export["pesos"].apply(lambda x: x[i])
    carteras_export.drop(columns=['pesos'], inplace=True)
    excel_carteras = os.path.join(carpeta_salida, f"montecarlo_portafolios_{n_iter}iter_{tema}.xlsx")
    carteras_export.to_excel(excel_carteras, index=False)
    print(f"✅ Resultados de Monte Carlo exportados a {excel_carteras}")

    excel_activos = os.path.join(carpeta_salida, f"activos_individuales_{tema}.xlsx")
    datosTickers.round(5).to_excel(excel_activos)
    print(f"✅ Estadísticas de activos individuales exportadas a {excel_activos}")

    # ========== Exportar portafolios destacados (nuevo) ==========
    resumen_portafolios = []
    for nombre, port in zip(nombres, carteras_top):
        pesos = port['pesos']
        usd = pesos * capital_usd
        resumen = {
            "Tipo": nombre,
            "Retorno": port["retorno"],
            "Riesgo": port["volatilidad"],
            "Sharpe": port["sharpe"]
        }
        for t, w, u in zip(cartera_tickers, pesos, usd):
            resumen[f"{t} (%)"] = w
            resumen[f"{t} (USD)"] = u
        resumen_portafolios.append(resumen)
    df_resumen = pd.DataFrame(resumen_portafolios)
    excel_resumen = os.path.join(carpeta_salida, f"portafolios_destacados_{n_iter}iter_{tema}.xlsx")
    df_resumen.to_excel(excel_resumen, index=False)
    print(f"✅ Resumen de portafolios destacados exportado a {excel_resumen}")

    # ========== Output consola para portafolio óptimo ==========
    print("\n📈 Portafolio de Sharpe máximo:")
    for t, w in zip(data.columns, port_opt['pesos']):
        print(f"   {t}: {w:.2%}    | {w*capital_usd:,.2f} USD")
    print(f"Retorno esperado: {port_opt['retorno']:.2%}")
    print(f"Riesgo: {port_opt['volatilidad']:.2%}")
    print(f"Sharpe Ratio: {port_opt['sharpe']:.2f}")

    return datosTickers.round(5), port_opt




