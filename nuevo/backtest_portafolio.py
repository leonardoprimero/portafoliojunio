import pandas as pd
import numpy as np
import os, glob
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
# from marcadeagua import marca_agua_logo_central, watermark_text_fade  ESO ES PARA VER SI LE PONEMOS MARCA DE AGUA

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
    plt.savefig(f"{carpeta_salida}/cumulative_returns.png", dpi=250)
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


def buscar_archivo_portafolio(destacados_path="Montecarlo", pattern="portafolios_destacados_*.xlsx"):
    archivos = glob.glob(os.path.join(destacados_path, pattern))
    if not archivos:
        raise FileNotFoundError("No se encontró ningún archivo de portafolio destacado en la carpeta Montecarlo.")
    archivos = sorted(archivos, key=os.path.getmtime, reverse=True)
    return archivos[0]

def backtest_equal_weight(
    tickers, benchmark="SPY", carpeta="DatosLimpios",
    carpeta_salida="BackTestingPortafolioIguales", capital=100_000
):
    os.makedirs(carpeta_salida, exist_ok=True)
    n = len(tickers)
    pesos = np.array([1/n] * n)
    backtest_profesional(
        pesos=pesos,
        tickers=tickers,
        benchmark=benchmark,
        carpeta=carpeta,
        carpeta_salida=carpeta_salida,
        capital=capital
    )


def normalizar_col(col):
    # Quita acentos, espacios, guiones y convierte a minúsculas
    return (
        unicodedata.normalize('NFKD', str(col))
        .encode('ascii', 'ignore').decode('ascii')
        .replace(" ", "_").replace("-", "_")
        .lower()
    )

def leer_excel_cartera_real(
    path="datosgenerales/CarteraReal.xlsx",
    hoja_cartera=0,
    hoja_cash=1,
    hoja_cliente=2
):
    """Lee las tres hojas del Excel de la cartera real y devuelve tres DataFrames con nombres de columnas normalizados"""
    xls = pd.ExcelFile(path)
    df_cartera = pd.read_excel(xls, sheet_name=hoja_cartera)
    df_cash = pd.read_excel(xls, sheet_name=hoja_cash)
    df_cliente = pd.read_excel(xls, sheet_name=hoja_cliente)
    
    # Normalizar columnas
    df_cartera.columns = [normalizar_col(c) for c in df_cartera.columns]
    df_cash.columns    = [normalizar_col(c) for c in df_cash.columns]
    df_cliente.columns = [normalizar_col(c) for c in df_cliente.columns]
    
    return df_cartera, df_cash, df_cliente

def BackTestingReal(
    excel_path="datosgenerales/CarteraReal.xlsx",
    hoja_cartera=0,
    hoja_cash=1,
    hoja_cliente=2,
    carpeta_datos="DatosLimpios",
    benchmark="SPY"
):
    df_cartera, df_cash, df_cliente = leer_excel_cartera_real(
        path=excel_path,
        hoja_cartera=hoja_cartera,
        hoja_cash=hoja_cash,
        hoja_cliente=hoja_cliente
    )

    # Ahora todo está normalizado y usás los nombres nuevos
    print(df_cartera.columns)
    print(df_cash.columns)
    print(df_cliente.columns)
   
   # === Extraer info de la hoja cartera ===
    tickers = df_cartera['ticker'].dropna().tolist()
    # Definir fecha de inicio del backtest (mínima entre todos los movimientos)
    fecha_inicio = pd.to_datetime(df_cartera['fecha_de_accion'].min())

    # Preferencia: si hay montos, usá esos; si no, usá peso (%)
    if 'monto_usd' in df_cartera.columns and df_cartera['monto_usd'].notnull().any():
        montos = df_cartera['monto_usd'].fillna(0).astype(float).values
        capital = montos.sum()
        pesos = montos / capital if capital > 0 else np.array([1/len(montos)]*len(montos))
    elif 'peso' in df_cartera.columns and df_cartera['peso'].notnull().any():
        pesos = df_cartera['peso'].fillna(0).astype(float).values / 100
        capital = 1
    else:
        raise Exception("Debés completar al menos una columna: 'Monto USD' o 'Peso (%)' en el Excel.")

    # Filtrar activos y pesos > 0
    tickers_filtrados = [t for t, p in zip(tickers, pesos) if str(t) and p > 0]
    pesos_filtrados = np.array([p for t, p in zip(tickers, pesos) if str(t) and p > 0])

    # === Nombre de la carpeta de salida personalizado ===
    nombre = str(df_cliente['nombre'].iloc[0]).strip()
    apellido = str(df_cliente['apellido'].iloc[0]).strip()
    carpeta_salida = f"BackTesting_{nombre}_{apellido}"
    os.makedirs(carpeta_salida, exist_ok=True)

    # === Cargar precios históricos (usa tu función actual) ===
    data = cargar_datos(tickers_filtrados + [benchmark], carpeta=carpeta_datos)
    data = data[data.index >= fecha_inicio].dropna()

    # === Calcular retornos ===
    retornos = np.log(data / data.shift(1)).dropna()
    ret_port = (retornos[tickers_filtrados] * pesos_filtrados).sum(axis=1)
    ret_bench = retornos[benchmark]

    curva_port = capital * (1 + ret_port).cumprod()
    curva_bench = capital * (1 + ret_bench).cumprod()

    # === Graficar (igual que backtest_profesional) ===
    # === Gráfico 1: Cumulative Returns ===
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10, 4))
    plt.plot(curva_port, label="Portafolio", lw=2.2)
    plt.plot(curva_bench, label="Benchmark", ls="--", lw=2)
    plt.ylabel('Valor acumulado')
    plt.title('Cumulative Returns')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{carpeta_salida}/cumulative_returns.png", dpi=250)
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
    print(f"🟢 Backtest real para {nombre} {apellido} guardado en: {carpeta_salida}/")

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

    # ...seguí con el resto de los gráficos igual que en backtest_profesional...

def buscar_cliente_por_dni_email(dni=None, email=None, carpeta_clientes="datosgenerales/Clientes"):
    excels_clientes = glob.glob(os.path.join(carpeta_clientes, "*.xlsx"))
    for excel_path in excels_clientes:
        try:
            _, _, df_cliente = leer_excel_cartera_real(path=excel_path)
            if dni:
                dni_archivo = str(df_cliente["dni"].iloc[0]).strip()
                if dni_archivo == str(dni):
                    return excel_path
            elif email:
                email_archivo = str(df_cliente["email"].iloc[0]).strip().lower()
                if email_archivo == email.lower().strip():
                    return excel_path
        except Exception as e:
            print(f"⚠️ Error leyendo {excel_path}: {e}")
    return None


