import os
import pandas as pd
from graficos_correlacion import (
    plot_clustermap_correlacion,
    plot_clustered_heatmap_sin_dendrograma
)

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

    if not dfs:
        print("❌ No se pudo calcular la matriz de correlaciones: no hay activos válidos.")
        for err in errores:
            print(err)
        return None

    df_retornos = pd.concat(dfs, axis=1)
    df_retornos = df_retornos.loc[:, tickers_ok]
    df_retornos = df_retornos.dropna(how='all', axis=0)

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
