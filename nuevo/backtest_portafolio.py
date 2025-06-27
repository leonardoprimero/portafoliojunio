import pandas as pd
import numpy as np
import os, glob
import matplotlib.pyplot as plt
import seaborn as sns
from marcadeagua import agregar_marca_agua

def cargar_datos(tickers, carpeta="DatosLimpios"):
    dfs = []
    for ticker in tickers:
        archivo = None
        for root, dirs, files in os.walk(carpeta):
            for file in files:
                if file.lower().startswith(ticker.lower()) and file.endswith(".csv"):
                    archivo = os.path.join(root, file)
        if archivo is None:
            print(f"❌ No se encontró el archivo para {ticker}")
            continue
        df = pd.read_csv(archivo, parse_dates=["Date"])
        df = df[["Date", "Close"]].set_index("Date")
        df.columns = [ticker]
        dfs.append(df)
    return pd.concat(dfs, axis=1).sort_index()

def backtest_profesional(
    pesos, tickers, benchmark="SPY", carpeta="DatosLimpios",
    carpeta_salida="BacktestPortafolioPro", capital=100_000
):
    os.makedirs(carpeta_salida, exist_ok=True)

    # === Cargar datos y calcular retornos ===
    data = cargar_datos(tickers + [benchmark], carpeta)
    data = data.dropna()
    retornos = np.log(data / data.shift(1)).dropna()
    ret_port = (retornos[tickers] * pesos).sum(axis=1)
    ret_bench = retornos[benchmark]

    curva_port = capital * (1 + ret_port).cumprod()
    curva_bench = capital * (1 + ret_bench).cumprod()

    # === Gráfico 1: Cumulative Returns ===
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10, 4))
    plt.plot(curva_port, label="Portafolio", lw=2.2)
    plt.plot(curva_bench, label="Benchmark", ls="--", lw=2)
    plt.ylabel('Valor acumulado')
    plt.title('Cumulative Returns')
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    agregar_marca_agua(
        fig,
        "datosgenerales/logo1.jpg",   # PNG transparente recomendado
        texto="leocaliva.com",
        reps_x=9,          # Ajustá para más o menos pattern
        reps_y=7,
        alpha_logo=0.10,
        alpha_text=0.08,
        size_factor=0.06,  # Tamaño chico
        font_size=15
    )
    plt.savefig(f"{carpeta_salida}/cumulative_returns.png", dpi=260, bbox_inches="tight")
    plt.close()

    # === Gráfico 2: Cumulative Returns (log) ===
    plt.figure(figsize=(10, 4))
    plt.plot(np.log(curva_port), label="Portafolio", lw=2.2)
    plt.plot(np.log(curva_bench), label="Benchmark", ls="--", lw=2)
    plt.ylabel('Log Valor acumulado')
    plt.title('Cumulative Returns (Log Scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{carpeta_salida}/cumulative_returns_log.png", dpi=250)
    plt.close()

    # === Gráfico 3: Rolling Volatility ===
    window = 126  # 6 meses hábiles
    rolling_vol_port = ret_port.rolling(window).std() * np.sqrt(252)
    rolling_vol_bench = ret_bench.rolling(window).std() * np.sqrt(252)
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_vol_port, label="Portafolio")
    plt.plot(rolling_vol_bench, label="Benchmark", ls="--")
    plt.ylabel('Volatilidad anualizada')
    plt.title('Rolling Volatility (6M)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{carpeta_salida}/rolling_volatility.png", dpi=250)
    plt.close()

    # === Gráfico 4: Rolling Sharpe Ratio ===
    rolling_sharpe = (ret_port.rolling(window).mean() / ret_port.rolling(window).std()) * np.sqrt(252)
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_sharpe, color='teal', label='Rolling Sharpe (6M)')
    plt.axhline(rolling_sharpe.mean(), ls='--', color='grey', label='Media')
    plt.ylabel('Sharpe Ratio')
    plt.title('Rolling Sharpe Ratio (6M)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{carpeta_salida}/rolling_sharpe.png", dpi=250)
    plt.close()

    # === Gráfico 5: Underwater Plot ===
    cum = (1 + ret_port).cumprod()
    roll_max = cum.cummax()
    drawdown = cum / roll_max - 1
    plt.figure(figsize=(10, 4))
    plt.fill_between(drawdown.index, drawdown, 0, color='tomato', alpha=0.5)
    plt.title('Underwater Plot (Drawdown)')
    plt.ylabel('Drawdown')
    plt.tight_layout()
    plt.savefig(f"{carpeta_salida}/underwater_plot.png", dpi=250)
    plt.close()

    # === Gráfico 6: Monthly Returns Heatmap ===
    df_month = ret_port.resample('M').apply(lambda x: (x + 1).prod() - 1)
    df_pivot = df_month.to_frame("ret").reset_index()
    df_pivot["Year"] = df_pivot["Date"].dt.year
    df_pivot["Month"] = df_pivot["Date"].dt.month
    tabla = df_pivot.pivot(index="Year", columns="Month", values="ret")
    plt.figure(figsize=(9, 4))
    sns.heatmap(tabla, annot=True, fmt=".1%", center=0, cmap="RdYlGn", cbar=False)
    plt.title("Monthly Returns (%)")
    plt.tight_layout()
    plt.savefig(f"{carpeta_salida}/monthly_returns_heatmap.png", dpi=250)
    plt.close()

    # === Gráfico 7: Annual Returns ===
    df_annual = ret_port.resample('Y').apply(lambda x: (x + 1).prod() - 1)
    df_annual.index = df_annual.index.year  # Pone solo el año como índice

    plt.figure(figsize=(12, 5))
    bars = plt.bar(df_annual.index.astype(str), df_annual.values, color="#1e88e5", alpha=0.87)
    plt.ylabel("Retorno anual", fontsize=15, weight="bold")
    plt.xlabel("Año", fontsize=14)
    plt.title("Annual Returns", fontsize=17, weight="bold", pad=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=13)
    plt.axhline(df_annual.mean(), color="orange", ls="--", lw=2, label="Promedio")
    plt.legend(fontsize=13, loc="best")

    # Opcional: etiqueta los valores arriba de cada barra
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.0%}", va='bottom', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{carpeta_salida}/annual_returns.png", dpi=280)
    plt.close()


    # === Gráfico 8: Return Quantiles ===
    q = pd.DataFrame({
        "Daily": ret_port,
        "Weekly": ret_port.resample('W').apply(lambda x: (x+1).prod()-1),
        "Monthly": ret_port.resample('M').apply(lambda x: (x+1).prod()-1)
    })
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=q, orient="h")
    plt.title("Return Quantiles")
    plt.tight_layout()
    plt.savefig(f"{carpeta_salida}/return_quantiles.png", dpi=250)
    plt.close()

    # ===== Exportar métricas principales =====
    met_port = calcular_metricas_portafolio(ret_port)
    met_bench = calcular_metricas_portafolio(ret_bench)
    resumen = pd.DataFrame([met_port, met_bench], index=["Portafolio óptimo", benchmark]).T
    resumen.to_excel(f"{carpeta_salida}/resumen_metricas.xlsx")
    resumen.to_csv(f"{carpeta_salida}/resumen_metricas.csv")
    print(f"🟢 Todos los gráficos y métricas guardados en: {carpeta_salida}/")

def calcular_metricas_portafolio(retornos, rf=0.00, freq=252):
    ann_ret = (1 + retornos).prod()**(freq/len(retornos)) - 1
    ann_vol = retornos.std() * np.sqrt(freq)
    sharpe = (ann_ret - rf) / ann_vol
    sortino = (ann_ret - rf) / (retornos[retornos<0].std() * np.sqrt(freq))
    dd = (retornos.add(1).cumprod() / (retornos.add(1).cumprod().cummax()) - 1)
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    return {
        "CAGR": ann_ret,
        "Vol anualizada": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Meses positivos": (retornos.resample('M').apply(lambda x: (x+1).prod()-1) > 0).mean()
    }
import pandas as pd
import numpy as np
import os
import glob
# ... tus otros imports

def buscar_archivo_portafolio(destacados_path="Montecarlo", pattern="portafolios_destacados_*.xlsx"):
    archivos = glob.glob(os.path.join(destacados_path, pattern))
    if not archivos:
        raise FileNotFoundError("No se encontró ningún archivo de portafolio destacado en la carpeta Montecarlo.")
    archivos = sorted(archivos, key=os.path.getmtime, reverse=True)
    return archivos[0]
