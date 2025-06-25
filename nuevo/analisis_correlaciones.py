import os
import pandas as pd
import numpy as np
from graficos_correlacion import (
    plot_clustermap_correlacion,
    plot_clustered_heatmap_sin_dendrograma,
    plot_rolling_correlation_lines
)

# ----------- CONFIGURACIÓN BENCHMARK Y FILTRO ----------- #
USAR_BENCHMARK = True        # True: el benchmark NO entra en matriz/rolling, solo se usa para análisis especial
BENCHMARK_TICKER = "SPY"    # Cambiá por el ticker que quieras (por ej. "QQQ", "^GSPC", "MERV", etc)
BENCHMARK_COMO_ACTIVO = False  # True: el benchmark también entra como un activo más en matrices/rolling
# -------------

def calcular_matriz_correlacion(
    carpeta_datos_limpios="DatosLimpios",
    carpeta_salida="Correlaciones",
    metodo="pearson",
    tema="bloomberg_dark",
    extension_salida="xlsx",
    generar_clustermap=True,
    mostrar_dendrograma=True
):
    os.makedirs(carpeta_salida, exist_ok=True)
    dfs = []
    tickers_ok = []
    errores = []

    print(f"\n🔍 Analizando archivos en {carpeta_datos_limpios} (subcarpetas incluidas)...")
    for root, dirs, files in os.walk(carpeta_datos_limpios):
        for archivo in files:
            if not archivo.endswith(".csv"):
                continue
            ticker = archivo.replace(".csv", "")
            path = os.path.join(root, archivo)
            try:
                df = pd.read_csv(path)
                if df.empty or "Close" not in df.columns:
                    errores.append(f"⚠️ {ticker} está vacío o le falta columna \'Close\'.")
                    continue
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                df = df.dropna(subset=["Date"])
                df.set_index("Date", inplace=True)
                serie = df["Close"].pct_change().rename(ticker)
                if serie.dropna().empty:
                    errores.append(f"⚠️ {ticker} no tiene retornos válidos (solo NaN o un único dato).")
                    continue
                dfs.append(serie)
                tickers_ok.append(ticker)
            except Exception as e:
                errores.append(f"❌ Error leyendo {ticker}: {e}")
                
            if USAR_BENCHMARK and not BENCHMARK_COMO_ACTIVO:
                if BENCHMARK_TICKER in tickers_ok:
                    tickers_ok.remove(BENCHMARK_TICKER)

    if not dfs:
        print("❌ No se pudo calcular la matriz de correlaciones: no hay activos válidos.")
        for err in errores:
            print(err)
        return None

    df_retornos = pd.concat(dfs, axis=1)
    df_retornos = df_retornos.loc[:, tickers_ok]
    df_retornos = df_retornos.dropna(how='all', axis=0)
    
    if USAR_BENCHMARK and not BENCHMARK_COMO_ACTIVO:
        if BENCHMARK_TICKER in df_retornos.columns:
            df_retornos = df_retornos.drop(columns=[BENCHMARK_TICKER])

    if df_retornos.shape[1] < 2:
        print("❌ Se necesitan al menos dos activos válidos para calcular correlación.")
        for err in errores:
            print(err)
        return None

    # Calcular y CLUSTERIZAR la matriz de correlaciones
    matriz = df_retornos.corr(method=metodo)
    from scipy.cluster.hierarchy import linkage, dendrogram
    linkage_matrix = linkage(matriz, method="average")
    dendro = dendrogram(linkage_matrix, labels=matriz.index, no_plot=True)
    idx = dendro["leaves"]
    matriz_cluster = matriz.iloc[idx, :].iloc[:, idx]

    # Guardar SOLO la matriz clusterizada
    nombre_archivo = f"matriz_correlacion_{metodo}_clusterizada.{extension_salida}"
    path_archivo = os.path.join(carpeta_salida, nombre_archivo)
    if extension_salida == "xlsx":
        matriz_cluster.to_excel(path_archivo)
    else:
        matriz_cluster.to_csv(path_archivo)

    print(f"\n✅ Matriz de correlaciones clusterizada ({metodo}) guardada en: {path_archivo}")

    # SOLO graficar el heatmap clusterizado (con o sin dendrograma)
    if generar_clustermap:
        if mostrar_dendrograma:
            plot_clustermap_correlacion(matriz, carpeta_salida, metodo, tema, mostrar_dendrograma=True)
        else:
            plot_clustered_heatmap_sin_dendrograma(matriz, carpeta_salida, metodo, tema)

    if errores:
        print("\n🟡 Advertencias:")
        for err in errores:
            print(err)
    print("\n🏁 Análisis de correlaciones terminado.")

    return matriz_cluster

def calcular_correlaciones_rolling(
    carpeta_datos_limpios="DatosLimpios",
    carpeta_salida="CorrelacionesRolling",
    metodo="pearson",
    ventana=60,
    tema="bloomberg_dark",
    top_n_pares_mas_volatiles=5
):
    import os
    import pandas as pd
    import numpy as np

    os.makedirs(carpeta_salida, exist_ok=True)
    dfs = []
    tickers_ok = []
    errores = []

    print(f"\n🔍 Analizando archivos en {carpeta_datos_limpios} (subcarpetas incluidas) para correlaciones rolling...")
    for root, dirs, files in os.walk(carpeta_datos_limpios):
        for archivo in files:
            if not archivo.endswith(".csv"):
                continue
            ticker = archivo.replace(".csv", "")
            path = os.path.join(root, archivo)
            try:
                df = pd.read_csv(path)
                if df.empty or "Close" not in df.columns:
                    errores.append(f"⚠️ {ticker} está vacío o le falta columna 'Close'.")
                    continue
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                df = df.dropna(subset=["Date"])
                df.set_index("Date", inplace=True)
                serie = df["Close"].pct_change().rename(ticker)
                if serie.dropna().empty:
                    errores.append(f"⚠️ {ticker} no tiene retornos válidos (solo NaN o un único dato).")
                    continue
                dfs.append(serie)
                tickers_ok.append(ticker)
            except Exception as e:
                errores.append(f"❌ Error leyendo {ticker}: {e}")
                
            if USAR_BENCHMARK and not BENCHMARK_COMO_ACTIVO:
                if BENCHMARK_TICKER in tickers_ok:
                    tickers_ok.remove(BENCHMARK_TICKER)

    if not dfs:
        print("❌ No se pudo calcular correlaciones rolling: no hay activos válidos.")
        for err in errores:
            print(err)
        return None

    df_retornos = pd.concat(dfs, axis=1)
    df_retornos = df_retornos.loc[:, tickers_ok]
    df_retornos = df_retornos.dropna(how='all', axis=0)
    
    if USAR_BENCHMARK and not BENCHMARK_COMO_ACTIVO:
        if BENCHMARK_TICKER in df_retornos.columns:
            df_retornos = df_retornos.drop(columns=[BENCHMARK_TICKER])

    if len(df_retornos) < ventana:
        print(f"❌ No hay suficientes datos ({len(df_retornos)} filas) para calcular correlaciones rolling con ventana de {ventana} días.")
        for err in errores:
            print(err)
        return None

    if df_retornos.shape[1] < 2:
        print("❌ Se necesitan al menos dos activos válidos para calcular correlación rolling.")
        for err in errores:
            print(err)
        return None

    print(f"Calculando correlaciones rolling con ventana de {ventana} días...")
    
    # Calcular correlaciones rolling para cada par de activos
    rolling_correlations = {}
    for i in range(len(tickers_ok)):
        for j in range(i + 1, len(tickers_ok)):
            ticker1 = tickers_ok[i]
            ticker2 = tickers_ok[j]
            pair_name = f"{ticker1}-{ticker2}"
            if ticker1 in df_retornos.columns and ticker2 in df_retornos.columns:
                rolling_corr = df_retornos[ticker1].rolling(window=ventana).corr(df_retornos[ticker2])
                rolling_correlations[pair_name] = rolling_corr
            else:
                print(f"Advertencia: No se encontraron datos para el par {pair_name}. Saltando.")

    if not rolling_correlations:
        print("❌ No se generaron correlaciones rolling para ningún par.")
        return None

    df_rolling_correlations = pd.DataFrame(rolling_correlations)
    df_rolling_correlations = df_rolling_correlations.dropna(how="all", axis=0).dropna(how="all", axis=1)

    # Guardar CSV con todas las correlaciones rolling
    nombre_archivo_rolling = f"correlaciones_rolling_{metodo}_{ventana}d.csv"
    path_archivo_rolling = os.path.join(carpeta_salida, nombre_archivo_rolling)
    df_rolling_correlations.to_csv(path_archivo_rolling)
    print(f"✅ Correlaciones rolling guardadas en: {path_archivo_rolling}")

    # Guardar resumen estadístico de cada par
    resumen = df_rolling_correlations.describe().T[["mean", "std", "min", "max"]]
    resumen_path = os.path.join(carpeta_salida, f"resumen_rolling_correlaciones_{metodo}_{ventana}d.csv")
    resumen.to_csv(resumen_path)
    print(f"📈 Resumen estadístico guardado en: {resumen_path}")

    # Graficar SOLO los N pares más volátiles
    if top_n_pares_mas_volatiles > 0 and len(df_rolling_correlations.columns) > top_n_pares_mas_volatiles:
        variaciones = df_rolling_correlations.std().sort_values(ascending=False)
        pares_top = variaciones.head(top_n_pares_mas_volatiles).index.tolist()
        from graficos_correlacion import plot_rolling_correlation_lines
        plot_rolling_correlation_lines(
            df_rolling_correlations, carpeta_salida, metodo, ventana, tema,
            pares_a_graficar=pares_top, plot_idx="topvar"
        )
        print(f"🖼️ Gráfico top {top_n_pares_mas_volatiles} pares volátiles guardado.")

    if errores:
        print("\n🟡 Advertencias durante el cálculo de correlaciones rolling:")
        for err in errores:
            print(err)
    print("\n🏁 Análisis de correlaciones rolling terminado.")

    return df_rolling_correlations


def graficar_pares_rolling_especificos(
    carpeta_salida="CorrelacionesRolling",
    metodo="pearson",
    ventana=60,
    tema="bloomberg_dark",
    pares_especificos=None  # lista de strings tipo "AAPL-MSFT"
):
    
    path_archivo_rolling = os.path.join(carpeta_salida, f"correlaciones_rolling_{metodo}_{ventana}d.csv")
    if not os.path.exists(path_archivo_rolling):
        print(f"❌ No existe el archivo de correlaciones rolling. Corré primero el análisis completo.")
        return

    df_rolling_correlations = pd.read_csv(path_archivo_rolling, index_col=0, parse_dates=True)

    if pares_especificos is None or len(pares_especificos) == 0:
        # Recomendar tres pares con mayor volatilidad
        top3 = df_rolling_correlations.std().sort_values(ascending=False).head(3).index.tolist()
        print("💡 Te interesaría ver la correlación entre algunos activos?")
        print(f"Te recomiendo estos tres pares por ser los más volátiles:\n  {', '.join(top3)}")
        print("Agregalos en 'pares_especificos' en tu main si querés graficarlos. El programa termina acá.")
        return

    pares_disponibles = df_rolling_correlations.columns.tolist()
    pares_graficados = 0

    for idx, par in enumerate(pares_especificos):
        if par in pares_disponibles:
            plot_rolling_correlation_lines(
                df_rolling_correlations, carpeta_salida, metodo, ventana, tema,
                pares_a_graficar=[par], plot_idx=idx
            )
            pares_graficados += 1
        else:
            print(f"⚠️ No se encontró el par {par} en el CSV, revisá los nombres (ejemplo: AAPL-MSFT).")
    if pares_graficados == 0:
        print("No se graficó ningún par. ¿Seguro que escribiste bien los nombres?")

    return