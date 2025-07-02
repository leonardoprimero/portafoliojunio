import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from graficos_correlacion import (
    plot_clustermap_correlacion,
    plot_clustered_heatmap_sin_dendrograma,
    plot_rolling_correlation_lines
)
from aaaconfig_usuario import cartera_para_correlacion


# ----------- CONFIGURACIÓN BENCHMARK Y FILTRO ----------- #
USAR_BENCHMARK = True        # True: el benchmark NO entra en matriz/rolling, solo se usa para análisis especial
BENCHMARK_TICKER = "SPY"    # Cambiá por el ticker que quieras (por ej. "QQQ", "^GSPC", "MERV", etc)
BENCHMARK_COMO_ACTIVO = False  # True: el benchmark también entra como un activo más en matrices/rolling
# -------------

def cargar_tickers_de_cartera(nombre_hoja, path_excel="carteras_recomendadas.xlsx"):
    if nombre_hoja.lower() == "todos":
        TICKERS_TASA = ["^IRX", "^FVX", "^TNX", "^TYX"]
        from aaaconfig_usuario import tickers
        return [t for t in tickers if t not in TICKERS_TASA]
    else:
        # Buscar hoja que matchee el nombre ignorando mayús/minús y paréntesis
        xl = pd.ExcelFile(path_excel)
        target = nombre_hoja.lower().strip()
        matching = [s for s in xl.sheet_names if s.lower().strip().startswith(target)]
        if not matching:
            raise ValueError(f"No se encontró ninguna hoja que empiece con '{nombre_hoja}' en {path_excel}. Las hojas disponibles son: {xl.sheet_names}")
        df = xl.parse(matching[0])
        return df['ticker'].tolist() if 'ticker' in df.columns else df.iloc[:, 0].tolist()
    
    
tickers_filtrados = cargar_tickers_de_cartera(cartera_para_correlacion)
print("Tickers filtrados para correlación:", tickers_filtrados)

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
            ticker = archivo.replace(".csv", "").strip().upper()
            if ticker not in tickers_filtrados:
                continue  # Solo procesa los tickers de la cartera elegida
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

    # --- BLOQUE ROBUSTO: limpiar matriz antes de clusterizar ---
    distancias = 1 - matriz

    # Limpia filas y columnas con todos NaN
    distancias = distancias.dropna(axis=0, how='all').dropna(axis=1, how='all')
    # Reemplaza infinitos por NaN y limpia de nuevo
    distancias = distancias.replace([np.inf, -np.inf], np.nan)
    distancias = distancias.dropna(axis=0, how='any').dropna(axis=1, how='any')

    if distancias.shape[0] > 1 and distancias.shape[1] > 1 and np.isfinite(distancias.values).all():
        from scipy.cluster.hierarchy import linkage, dendrogram
        linkage_matrix = linkage(distancias, method="average")
        # Usá distancias.index como labels, ¡no matriz.index!
        dendro = dendrogram(linkage_matrix, labels=distancias.index, no_plot=True)
        idx = dendro["leaves"]
        # Reordená solo la parte relevante de la matriz original
        matriz_cluster = matriz.loc[distancias.index, distancias.index].iloc[idx, :].iloc[:, idx]
    else:
        print("⚠️ No se puede hacer clustering: matriz con NaN, infinitos o menos de dos activos con datos válidos.")
        matriz_cluster = matriz  # o None, según prefieras
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
            if ticker not in tickers_filtrados:
                continue  # Solo procesa los tickers de la cartera elegida
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

def calcular_metricas_resumen_correlacion(matriz):
    valores = matriz.values.flatten()
    valores = valores[~np.isnan(valores)]
    valores = valores[valores != 1]  # excluimos la diagonal
    resumen = {
        "Media": np.mean(valores),
        "Desviacion Std": np.std(valores),
        "% pares > 0.75": np.mean(valores > 0.75) * 100,
        "% pares < -0.5": np.mean(valores < -0.5) * 100,
        "Máximo": np.max(valores),
        "Mínimo": np.min(valores)
    }
    return resumen

def guardar_metricas_resumen(resumen, carpeta_salida, nombre="resumen_metricas_correlacion.csv"):
    os.makedirs(carpeta_salida, exist_ok=True)
    df = pd.DataFrame(resumen, index=[0])
    path = os.path.join(carpeta_salida, nombre)
    df.to_csv(path, index=False)
    print(f"📊 Métricas resumen guardadas en {path}")


def calcular_estabilidad_rolling(df_rolling_correlations):
    resumen = df_rolling_correlations.describe().T
    resumen = resumen[["mean", "std"]].dropna()
    resumen["coef_estabilidad"] = resumen["std"] / resumen["mean"].abs()
    resumen = resumen.sort_values("coef_estabilidad", ascending=False)
    return resumen

def guardar_estabilidad_rolling(resumen_estabilidad, carpeta_salida, metodo, ventana):
    os.makedirs(carpeta_salida, exist_ok=True)
    path = os.path.join(carpeta_salida, f"estabilidad_rolling_{metodo}_{ventana}d.csv")
    resumen_estabilidad.to_csv(path)
    print(f"📈 Estabilidad de correlaciones rolling guardada en {path}")


def ranking_correlaciones_extremas(matriz, top_n=10):
    df = matriz.stack().reset_index()
    df.columns = ["Activo 1", "Activo 2", "Correlacion"]
    df = df[df["Activo 1"] != df["Activo 2"]]  # excluir la diagonal
    df = df.dropna()
    top_pos = df.sort_values("Correlacion", ascending=False).head(top_n)
    top_neg = df.sort_values("Correlacion", ascending=True).head(top_n)
    return top_pos, top_neg

def guardar_rankings_extremos(top_pos, top_neg, carpeta_salida):
    os.makedirs(carpeta_salida, exist_ok=True)
    top_pos.to_csv(os.path.join(carpeta_salida, "pares_mas_correlacionados.csv"), index=False)
    top_neg.to_csv(os.path.join(carpeta_salida, "pares_menos_correlacionados.csv"), index=False)
    print("📌 Rankings extremos guardados")

def ejecutar_pca_sobre_retornos(
    carpeta_datos_limpios="DatosLimpios", 
    carpeta_salida="PCA", 
    n_componentes=2, 
    tema="default"
):
    from aaaconfig_usuario import cartera_para_correlacion
    # Importá la función y usá la misma lógica de tickers filtrados
    tickers_filtrados = cargar_tickers_de_cartera(cartera_para_correlacion)
    tickers_filtrados = [str(t).strip().upper() for t in tickers_filtrados]

    os.makedirs(carpeta_salida, exist_ok=True)
    dfs = []
    tickers_ok = []

    for root, _, files in os.walk(carpeta_datos_limpios):
        for archivo in files:
            if archivo.endswith(".csv"):
                ticker = archivo.replace(".csv", "").strip().upper()
                if ticker not in tickers_filtrados:
                    continue
                path = os.path.join(root, archivo)
                df = pd.read_csv(path)
                if "Close" not in df.columns:
                    continue
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
                retornos = df["Close"].pct_change().dropna()
                if retornos.empty:
                    continue
                dfs.append(retornos.rename(ticker))
                tickers_ok.append(ticker)

    if not dfs:
        print("❌ No se pudo calcular el PCA: no hay activos válidos.")
        return

    df_retornos = pd.concat(dfs, axis=1)
    df_retornos = df_retornos.loc[:, tickers_ok]

    # Hacé dropna como en la correlación (filas y columnas con al menos un dato)
    df_retornos = df_retornos.dropna(how='all', axis=0)
    # Drop filas donde falte algún dato, para que sea rectangular
    df_retornos = df_retornos.dropna(axis=0, how='any')

    # Standardize
    X = (df_retornos - df_retornos.mean()) / df_retornos.std()

    if X.shape[1] < 2 or X.shape[0] < 2:
        print("⚠️ No hay suficientes activos ni datos válidos para aplicar PCA.")
        print("Shape de X:", X.shape)
        return

    # PCA
    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X.T)
    explained = pca.explained_variance_ratio_ * 100

    # Gráfico profesional
    plt.figure(figsize=(12, 10))
    plt.style.use("default")
    ax = plt.gca()
    ax.set_facecolor("whitesmoke")
    ax.grid(True, linestyle="--", alpha=0.3)

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], s=120, alpha=0.85, color="royalblue", edgecolors="black", linewidths=0.5)

    for i, nombre in enumerate(X.columns):
        plt.text(X_pca[i, 0] + 1.8, X_pca[i, 1] + 1.8, nombre, fontsize=10, weight="bold", color="black")

    plt.xlabel(f"Componente Principal 1 ({explained[0]:.1f}%)", fontsize=13)
    plt.ylabel(f"Componente Principal 2 ({explained[1]:.1f}%)", fontsize=13)
    plt.title("PCA de Retornos Estandarizados\n(Matriz de Correlación entre Activos)", fontsize=16, weight="bold")
    plt.tight_layout()

    out_path = os.path.join(carpeta_salida, "pca_retorno_2D.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"📊 PCA guardado en: {out_path}")