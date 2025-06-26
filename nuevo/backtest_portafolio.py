import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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

def analizar_portafolio_optimo(
    pesos, tickers, benchmark="SPY", capital=100_000,
    carpeta="DatosLimpios", plot=True, save=False, carpeta_salida="BacktestPortafolio"
):
    import os

    data = cargar_datos(tickers + [benchmark], carpeta)
    data = data.dropna()
    retornos = np.log(data / data.shift(1)).dropna()
    # Serie del portafolio óptimo
    pesos = np.array(pesos)
    ret_port = (retornos[tickers] * pesos).sum(axis=1)
    curva_port = capital * (1 + ret_port).cumprod()
    # Serie del benchmark
    ret_bench = retornos[benchmark]
    curva_bench = capital * (1 + ret_bench).cumprod()
    # Drawdown
    dd_port = curva_port / curva_port.cummax() - 1
    dd_bench = curva_bench / curva_bench.cummax() - 1

    # Métricas
    met_port = calcular_metricas_portafolio(ret_port)
    met_bench = calcular_metricas_portafolio(ret_bench)
    resumen = pd.DataFrame([met_port, met_bench], index=["Portafolio óptimo", benchmark]).T

    if save:
        os.makedirs(carpeta_salida, exist_ok=True)

    # GRAFICOS
    fig, axs = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    # Curva de valor
    axs[0].plot(curva_port, label="Portafolio óptimo", lw=2.3)
    axs[0].plot(curva_bench, label=benchmark, lw=2.3, ls="--")
    axs[0].set_ylabel("Valor ($)")
    axs[0].set_title("Evolución del valor del portafolio vs Benchmark")
    axs[0].legend()
    axs[0].grid(True, alpha=0.2)
    # Drawdown
    axs[1].fill_between(dd_port.index, dd_port, 0, color="tab:blue", alpha=0.35)
    axs[1].plot(dd_bench.index, dd_bench, color="orange", alpha=0.45, ls="--")
    axs[1].set_ylabel("Drawdown")
    axs[1].set_title("Drawdown (caída máxima desde el máximo histórico)")
    axs[1].grid(True, alpha=0.15)
    # Histograma de retornos diarios
    axs[2].hist(ret_port, bins=60, alpha=0.8, label="Portafolio", color="tab:blue", density=True)
    axs[2].hist(ret_bench, bins=60, alpha=0.6, label=benchmark, color="orange", density=True)
    axs[2].set_title("Histograma de retornos diarios")
    axs[2].legend()
    axs[2].grid(True, alpha=0.1)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(carpeta_salida, "resumen_backtest.png"), dpi=320)
        plt.close(fig)
    elif plot:
        plt.show()

    # Tabla resumen
    if save:
        resumen.to_excel(os.path.join(carpeta_salida, "resumen_metricas.xlsx"))
        resumen.to_csv(os.path.join(carpeta_salida, "resumen_metricas.csv"))
        curva_port.to_frame("Portafolio_optimo").join(curva_bench.to_frame(benchmark)).to_csv(
            os.path.join(carpeta_salida, "curvas_valor.csv")
        )
        dd_port.to_frame("Drawdown_portafolio").join(dd_bench.to_frame("Drawdown_bench")).to_csv(
            os.path.join(carpeta_salida, "drawdowns.csv")
        )
        print(f"🟢 Gráficos y métricas del backtest guardados en: {carpeta_salida}/")

    print("\n===== MÉTRICAS RESUMEN =====")
    print(resumen.round(4))

    return {
        "curva_port": curva_port,
        "curva_bench": curva_bench,
        "drawdown_port": dd_port,
        "drawdown_bench": dd_bench,
        "retornos_port": ret_port,
        "retornos_bench": ret_bench,
        "resumen": resumen
    }