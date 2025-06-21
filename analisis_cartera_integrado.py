"""
ANALISIS CUANTITATIVO INTEGRADO DE CARTERAS DE INVERSIÓN
"""
import os
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from pypfopt import expected_returns, risk_models, plotting #, objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
# from sklearn.decomposition import PCA # Para análisis PCA futuro
# from scipy.cluster.hierarchy import linkage, fcluster, dendrogram # Para clustering futuro
# from scipy import stats # Para estadísticas más avanzadas

# ==============================================================================
# --- CONFIGURACIÓN GENERAL DEL SCRIPT ---
# ==============================================================================

# --- Configuración de Activos y Fechas ---
# Lista de tickers a analizar. Ejemplo: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY']
LISTA_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
# Ticker del benchmark para comparaciones (ej. 'SPY', '^GSPC')
BENCHMARK_TICKER = 'SPY'
# Fecha de inicio para la descarga de datos (YYYY-MM-DD)
FECHA_INICIO_DATOS = '2020-01-01'
# Fecha de fin para la descarga de datos (YYYY-MM-DD). Si es None, se usará la fecha actual.
FECHA_FIN_DATOS = datetime.datetime.now().strftime('%Y-%m-%d')

# --- Configuración de Capital y Cartera (Opcional para análisis inicial) ---
# Capital total disponible para la inversión
CAPITAL_TOTAL_CARTERA = 100000  # Ejemplo: 100,000 unidades monetarias
# (Opcional) Asignación inicial de pesos/porcentajes si se quiere analizar una cartera predefinida.
# Si es None, se procederá directamente a la optimización.
# Ejemplo: {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.25, 'TSLA': 0.15, 'NVDA': 0.10}
ASIGNACION_INICIAL_PESOS = None

# --- Configuración de Gráficos ---
# Tema visual para los gráficos: "light", "dark", "vintage", "normal"
TEMA_GRAFICOS = "dark"
# Escala para los gráficos de precios: "linear" o "log"
ESCALA_PRECIOS_GRAFICOS = "log"
# Paleta de colores para Seaborn (ej. "viridis", "magma", "Set2", "husl")
PALETA_COLORES_GRAFICOS = "viridis"

# --- Configuración de Análisis Técnico por Activo ---
# Periodos para las Medias Móviles Simples (SMA)
SMA_CORTA_PERIODO = 20
SMA_MEDIA_PERIODO = 50
SMA_LARGA_PERIODO = 200
# Activar/Desactivar indicadores y sus periodos
ACTIVAR_RSI = True
RSI_PERIODO = 14
ACTIVAR_MACD = True
MACD_FAST_PERIODO = 12
MACD_SLOW_PERIODO = 26
MACD_SIGNAL_PERIODO = 9
ACTIVAR_BBANDS = True
BBANDS_PERIODO = 20
BBANDS_STD_DEV = 2
ACTIVAR_ANALISIS_VOLUMEN = True

# --- Configuración de Optimización de Cartera ---
# Método de optimización:
# "max_sharpe" (Maximizar Ratio de Sharpe)
# "min_volatility" (Minimizar Volatilidad)
# "efficient_risk" (Minimizar Volatilidad para un Retorno Objetivo)
# "efficient_return" (Maximizar Retorno para una Volatilidad Objetivo)
METODO_OPTIMIZACION = "max_sharpe"
# Retorno objetivo (solo para "efficient_risk")
RETORNO_OBJETIVO_OPTIMIZACION = 0.20 # Ejemplo: 20% anual
# Volatilidad objetivo (solo para "efficient_return")
VOLATILIDAD_OBJETIVO_OPTIMIZACION = 0.15 # Ejemplo: 15% anual
# Tasa libre de riesgo para cálculo de Sharpe Ratio
TASA_LIBRE_RIESGO = 0.02 # Ejemplo: 2%

# Parámetros para Simulación Monte Carlo (si se activa)
ACTIVAR_MONTE_CARLO = True
NUM_SIMULACIONES_MC = 10000

# --- Configuración de Cache de Datos ---
# Activar/Desactivar el uso de datos cacheados localmente para evitar descargas repetidas
USAR_CACHE_DATOS = True # Cambiar a False para forzar la descarga siempre

# --- Configuración de Salidas (Reportes) ---
# Nombre base para los archivos de salida (PDF y Excel)
NOMBRE_BASE_REPORTE = "Reporte_Cartera_Integrado"
# Carpeta donde se guardarán los reportes generados
CARPETA_REPORTES = "reportes_generados"
# Carpeta para datos descargados/cacheados (opcional)
CARPETA_DATOS_ENTRADA = "datos_entrada"
# Carpeta para gráficos temporales (se puede limpiar después)
CARPETA_GRAFICOS_TEMP = os.path.join(CARPETA_REPORTES, "temp_graficos")

# ==============================================================================
# --- ESTRUCTURA DE CARPETAS ---
# ==============================================================================
def crear_carpetas_necesarias():
    """Crea las carpetas necesarias si no existen."""
    os.makedirs(CARPETA_REPORTES, exist_ok=True)
    os.makedirs(CARPETA_DATOS_ENTRADA, exist_ok=True)
    os.makedirs(CARPETA_GRAFICOS_TEMP, exist_ok=True)

    # Subcarpetas para gráficos
    os.makedirs(os.path.join(CARPETA_GRAFICOS_TEMP, "activos_individuales"), exist_ok=True)
    os.makedirs(os.path.join(CARPETA_GRAFICOS_TEMP, "analisis_cartera"), exist_ok=True)
    print("Carpetas necesarias verificadas/creadas.")

# ==============================================================================
# --- FUNCIONES AUXILIARES (Se irán añadiendo más) ---
# ==============================================================================
def configurar_estilo_graficos():
    """Configura el estilo de Matplotlib y Seaborn según TEMA_GRAFICOS."""
    if TEMA_GRAFICOS == "dark":
        plt.style.use('dark_background')
        sns.set_palette(PALETA_COLORES_GRAFICOS)
        # Puedes añadir más configuraciones específicas para el tema oscuro
        plt.rcParams.update({
            'figure.facecolor': '#282c34',
            'axes.facecolor': '#282c34',
            'text.color': '#abb2bf',
            'axes.labelcolor': '#abb2bf',
            'xtick.color': '#abb2bf',
            'ytick.color': '#abb2bf',
            'grid.color': '#4b5263',
            'legend.facecolor': '#3c4049',
            'legend.edgecolor': '#4b5263'
        })
    elif TEMA_GRAFICOS == "light":
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(PALETA_COLORES_GRAFICOS)
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'text.color': '#333333',
            'axes.labelcolor': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'grid.color': '#cccccc',
            'legend.facecolor': 'white',
            'legend.edgecolor': '#cccccc'
        })
    elif TEMA_GRAFICOS == "vintage":
        plt.style.use('seaborn-v0_8-pastel')
        sns.set_palette("YlGnBu") # Ejemplo de paleta vintage
        # ... más configuraciones ...
    else: # "normal" o por defecto
        plt.style.use('seaborn-v0_8-whitegrid') # O 'default'
        sns.set_palette(PALETA_COLORES_GRAFICOS)

    plt.rcParams['figure.figsize'] = (10, 6) # Tamaño base para figuras
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    # Para asegurar que los caracteres se muestren bien en PDF
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    print(f"Estilo de gráficos configurado a: {TEMA_GRAFICOS} con paleta {PALETA_COLORES_GRAFICOS}")

# ==============================================================================
# --- MÓDULO 1: OBTENCIÓN DE DATOS ---
# ==============================================================================
def obtener_datos_activos(tickers_solicitados, fecha_inicio, fecha_fin, usar_cache=True, carpeta_cache="datos_entrada"):
    """
    Descarga o carga desde caché los datos de precios históricos para una lista de tickers.

    Args:
        tickers_solicitados (list): Lista de strings con los tickers.
        fecha_inicio (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        fecha_fin (str): Fecha de fin en formato 'YYYY-MM-DD'.
        usar_cache (bool): Si es True, intenta cargar desde CSV local antes de descargar.
                           Guarda los datos descargados en CSV si no existen.
        carpeta_cache (str): Ruta a la carpeta para guardar/cargar archivos CSV de caché.

    Returns:
        dict: Un diccionario donde las claves son los tickers y los valores son
              DataFrames con 'Adj Close' y 'Volume' para ese ticker.
              Retorna un diccionario vacío si no se pueden obtener datos.
    """
    print(f"\n--- Iniciando Módulo de Obtención de Datos para: {', '.join(tickers_solicitados)} ---")
    datos_por_ticker = {}

    for ticker in tickers_solicitados:
        archivo_cache_ticker = os.path.join(carpeta_cache, f"{ticker}_{fecha_inicio}_a_{fecha_fin}.csv")
        df_ticker_data = None

        if usar_cache and os.path.exists(archivo_cache_ticker):
            try:
                print(f"Cargando datos para {ticker} desde caché: {archivo_cache_ticker}")
                df_ticker_data = pd.read_csv(archivo_cache_ticker, index_col='Date', parse_dates=True)
                if df_ticker_data.empty:
                    print(f"Advertencia: Archivo de caché para {ticker} está vacío. Se intentará descargar.")
                    df_ticker_data = None
                elif 'Adj Close' not in df_ticker_data.columns or 'Volume' not in df_ticker_data.columns:
                    print(f"Advertencia: Archivo de caché para {ticker} no contiene 'Adj Close' o 'Volume'. Se intentará descargar.")
                    df_ticker_data = None
            except Exception as e:
                print(f"Error al cargar datos de caché para {ticker} ({archivo_cache_ticker}): {e}. Se intentará descargar.")
                df_ticker_data = None

        if df_ticker_data is None:
            print(f"Descargando datos para {ticker} desde Yahoo Finance ({fecha_inicio} a {fecha_fin})...")
            try:
                data = yf.download(ticker, start=fecha_inicio, end=fecha_fin, progress=False, auto_adjust=False) # auto_adjust=False para obtener Open, High, Low, Close, Adj Close, Volume
                if data.empty:
                    print(f"❌ No se encontraron datos para {ticker} en el rango especificado.")
                    continue

                # Seleccionar solo 'Adj Close' y 'Volume'. yfinance a veces devuelve 'Adj Close' con minúscula.
                cols_a_usar = []
                if 'Adj Close' in data.columns:
                    cols_a_usar.append('Adj Close')
                elif 'adj close' in data.columns: # Manejar posible minúscula
                    data.rename(columns={'adj close': 'Adj Close'}, inplace=True)
                    cols_a_usar.append('Adj Close')
                else:
                    print(f"Advertencia: No se encontró 'Adj Close' para {ticker}. Usando 'Close'.")
                    if 'Close' in data.columns:
                         data.rename(columns={'Close': 'Adj Close'}, inplace=True) # Tratar 'Close' como 'Adj Close'
                         cols_a_usar.append('Adj Close')
                    else:
                        print(f"Error: No se encontró ni 'Adj Close' ni 'Close' para {ticker}.")
                        continue

                if 'Volume' in data.columns:
                    cols_a_usar.append('Volume')
                elif 'volume' in data.columns: # Manejar posible minúscula
                     data.rename(columns={'volume': 'Volume'}, inplace=True)
                     cols_a_usar.append('Volume')
                else:
                    print(f"Advertencia: No se encontró 'Volume' para {ticker}. Se asignará 0.")
                    data['Volume'] = 0 # Crear columna de volumen con ceros
                    cols_a_usar.append('Volume')

                df_ticker_data = data[cols_a_usar].copy()

                # Interpolar valores faltantes antes de guardar en caché
                df_ticker_data.interpolate(method='linear', axis=0, inplace=True)
                df_ticker_data.fillna(method='bfill', inplace=True) # Rellenar los NaN iniciales que la interpolación no cubre
                df_ticker_data.fillna(method='ffill', inplace=True) # Rellenar los NaN finales

                if usar_cache:
                    try:
                        os.makedirs(carpeta_cache, exist_ok=True)
                        df_ticker_data.to_csv(archivo_cache_ticker)
                        print(f"Datos para {ticker} guardados en caché: {archivo_cache_ticker}")
                    except Exception as e:
                        print(f"Error al guardar datos en caché para {ticker}: {e}")

            except Exception as e:
                print(f"⚠️ Error al descargar datos para {ticker}: {e}")
                continue

        if df_ticker_data is not None and not df_ticker_data.empty:
            # Asegurar que las columnas se llamen 'Adj Close' y 'Volume'
            if 'Adj Close' not in df_ticker_data.columns and 'Close' in df_ticker_data.columns: # Si se renombró Close a Adj Close
                pass # ya está hecho
            elif 'Adj Close' not in df_ticker_data.columns:
                 print(f"Error Crítico: Falta 'Adj Close' en datos finales de {ticker}")
                 continue
            if 'Volume' not in df_ticker_data.columns:
                 print(f"Error Crítico: Falta 'Volume' en datos finales de {ticker}")
                 continue

            datos_por_ticker[ticker] = df_ticker_data
        else:
            print(f"No se pudieron obtener datos para {ticker}.")

    if not datos_por_ticker:
        print("❌ No se pudieron obtener datos para ninguno de los tickers solicitados.")
        return {}

    print(f"✅ Datos obtenidos y procesados para {len(datos_por_ticker)} tickers.")
    print("--- Módulo de Obtención de Datos Finalizado ---")
    return datos_por_ticker

# ==============================================================================
# --- MÓDULO 2: ANÁLISIS POR ACTIVO INDIVIDUAL ---
# ==============================================================================
def calcular_indicadores_tecnicos(df_activo):
    """
    Calcula indicadores técnicos para un DataFrame de activo.
    El DataFrame debe tener columnas 'Adj Close' y 'Volume'.
    """
    df = df_activo.copy()
    precio_col = 'Adj Close' # Asegurarse de usar el nombre correcto de columna
    vol_col = 'Volume'

    # Retornos
    df['RetornoDiario'] = df[precio_col].pct_change()

    # Medias Móviles
    df[f'SMA_{SMA_CORTA_PERIODO}'] = df[precio_col].rolling(window=SMA_CORTA_PERIODO).mean()
    df[f'SMA_{SMA_MEDIA_PERIODO}'] = df[precio_col].rolling(window=SMA_MEDIA_PERIODO).mean()
    df[f'SMA_{SMA_LARGA_PERIODO}'] = df[precio_col].rolling(window=SMA_LARGA_PERIODO).mean()

    if ACTIVAR_RSI:
        delta = df[precio_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIODO).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIODO).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    if ACTIVAR_MACD:
        ema_fast = df[precio_col].ewm(span=MACD_FAST_PERIOD, adjust=False).mean()
        ema_slow = df[precio_col].ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    if ACTIVAR_BBANDS:
        df['BB_Middle'] = df[precio_col].rolling(window=BBANDS_PERIODO).mean()
        std_dev = df[precio_col].rolling(window=BBANDS_PERIODO).std()
        df['BB_Upper'] = df['BB_Middle'] + (std_dev * BBANDS_STD_DEV)
        df['BB_Lower'] = df['BB_Middle'] - (std_dev * BBANDS_STD_DEV)

    # El volumen ya está en df[vol_col]
    if ACTIVAR_ANALISIS_VOLUMEN and vol_col not in df.columns:
        print(f"Advertencia: Columna '{vol_col}' no encontrada para análisis de volumen.")
        # Podríamos añadir una columna de ceros si es estrictamente necesario más adelante
        # pero es mejor que la obtención de datos la provea.

    return df

def generar_graficos_activo(df_indicadores, ticker):
    """
    Genera y guarda los gráficos para un activo individual.
    """
    print(f"  Generando gráficos para {ticker}...")
    path_graficos_activo = os.path.join(CARPETA_GRAFICOS_TEMP, "activos_individuales")
    precio_col = 'Adj Close'
    vol_col = 'Volume'

    # 1. Gráfico de Precios, SMAs y Bandas de Bollinger
    fig, ax1 = plt.subplots(figsize=(12, 7))
    color_precio = 'tab:blue'
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Precio', color=color_precio)
    ax1.plot(df_indicadores.index, df_indicadores[precio_col], color=color_precio, label='Precio Adj.', alpha=0.8, linewidth=1.5)
    ax1.plot(df_indicadores.index, df_indicadores[f'SMA_{SMA_CORTA_PERIODO}'], color='orange', label=f'SMA {SMA_CORTA_PERIODO}d', alpha=0.7, linewidth=1)
    ax1.plot(df_indicadores.index, df_indicadores[f'SMA_{SMA_MEDIA_PERIODO}'], color='purple', label=f'SMA {SMA_MEDIA_PERIODO}d', alpha=0.7, linewidth=1)
    ax1.plot(df_indicadores.index, df_indicadores[f'SMA_{SMA_LARGA_PERIODO}'], color='green', label=f'SMA {SMA_LARGA_PERIODO}d', alpha=0.7, linewidth=1.5)

    if ACTIVAR_BBANDS and 'BB_Upper' in df_indicadores.columns:
        ax1.plot(df_indicadores.index, df_indicadores['BB_Upper'], color='gray', linestyle='--', alpha=0.5, label='BB Superior')
        ax1.plot(df_indicadores.index, df_indicadores['BB_Middle'], color='gray', linestyle=':', alpha=0.6, label='BB Media')
        ax1.plot(df_indicadores.index, df_indicadores['BB_Lower'], color='gray', linestyle='--', alpha=0.5, label='BB Inferior')
        ax1.fill_between(df_indicadores.index, df_indicadores['BB_Lower'], df_indicadores['BB_Upper'], color='gray', alpha=0.1)

    ax1.tick_params(axis='y', labelcolor=color_precio)
    ax1.legend(loc='upper left')
    ax1.set_title(f'{ticker} - Análisis de Precio y Volumen', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.3)
    if ESCALA_PRECIOS_GRAFICOS == 'log':
        ax1.set_yscale('log')

    # 2. Gráfico de Volumen (si está activado)
    if ACTIVAR_ANALISIS_VOLUMEN and vol_col in df_indicadores.columns:
        ax2 = ax1.twinx() # Compartir el mismo eje x
        color_volumen = 'tab:red'
        ax2.set_ylabel('Volumen', color=color_volumen)
        ax2.bar(df_indicadores.index, df_indicadores[vol_col], color=color_volumen, alpha=0.3, width=1.0, label='Volumen')
        ax2.tick_params(axis='y', labelcolor=color_volumen)
        ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig(os.path.join(path_graficos_activo, f"{ticker}_precio_volumen.png"), dpi=300)
    plt.close(fig)

    # 3. Gráfico de RSI (si está activado)
    if ACTIVAR_RSI and 'RSI' in df_indicadores.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_indicadores.index, df_indicadores['RSI'], label='RSI', color='teal', linewidth=1.5)
        ax.axhline(70, linestyle='--', alpha=0.5, color='red', label='Sobrecompra (70)')
        ax.axhline(30, linestyle='--', alpha=0.5, color='green', label='Sobreventa (30)')
        ax.set_title(f'{ticker} - Índice de Fuerza Relativa (RSI)', fontsize=16)
        ax.set_ylabel('RSI')
        ax.set_xlabel('Fecha')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.tight_layout()
        plt.savefig(os.path.join(path_graficos_activo, f"{ticker}_rsi.png"), dpi=300)
        plt.close(fig)

    # 4. Gráfico de MACD (si está activado)
    if ACTIVAR_MACD and 'MACD' in df_indicadores.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_indicadores.index, df_indicadores['MACD'], label='MACD', color='blue', linewidth=1.5)
        ax.plot(df_indicadores.index, df_indicadores['MACD_Signal'], label='Señal', color='red', linestyle='--', linewidth=1.5)
        ax.bar(df_indicadores.index, df_indicadores['MACD_Hist'], label='Histograma', color='gray', alpha=0.5)
        ax.axhline(0, linestyle='--', alpha=0.5, color='black')
        ax.set_title(f'{ticker} - MACD', fontsize=16)
        ax.set_ylabel('Valor MACD')
        ax.set_xlabel('Fecha')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.tight_layout()
        plt.savefig(os.path.join(path_graficos_activo, f"{ticker}_macd.png"), dpi=300)
        plt.close(fig)

    # 5. Histograma de Retornos Diarios
    if 'RetornoDiario' in df_indicadores.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df_indicadores['RetornoDiario'].dropna(), kde=True, ax=ax, bins=50)
        media_retornos = df_indicadores['RetornoDiario'].mean()
        std_retornos = df_indicadores['RetornoDiario'].std()
        ax.axvline(media_retornos, color='red', linestyle='--', label=f'Media: {media_retornos:.4f}')
        ax.axvline(media_retornos + std_retornos, color='green', linestyle='--', label=f'+1 Std Dev')
        ax.axvline(media_retornos - std_retornos, color='green', linestyle='--', label=f'-1 Std Dev')
        ax.set_title(f'{ticker} - Distribución de Retornos Diarios', fontsize=16)
        ax.set_xlabel('Retorno Diario')
        ax.set_ylabel('Frecuencia')
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(path_graficos_activo, f"{ticker}_distribucion_retornos.png"), dpi=300)
        plt.close(fig)

    print(f"  Gráficos para {ticker} guardados en {path_graficos_activo}")


def analizar_activos_individuales(datos_por_ticker):
    """
    Realiza el análisis técnico y genera gráficos para cada activo.
    """
    print("\n--- Iniciando Módulo de Análisis por Activo Individual ---")
    resultados_analisis_individual = {}

    for ticker, df_activo_original in datos_por_ticker.items():
        if ticker == BENCHMARK_TICKER: # No analizar el benchmark como un activo individual aquí
            continue

        print(f"Analizando activo: {ticker}...")
        if df_activo_original.empty or 'Adj Close' not in df_activo_original.columns:
            print(f"  No hay datos suficientes o falta 'Adj Close' para {ticker}. Saltando análisis individual.")
            continue

        df_con_indicadores = calcular_indicadores_tecnicos(df_activo_original)

        # Calcular estadísticas descriptivas básicas
        stats = {
            'retorno_diario_medio': df_con_indicadores['RetornoDiario'].mean(),
            'volatilidad_diaria': df_con_indicadores['RetornoDiario'].std(),
            'retorno_anualizado_estimado': ((1 + df_con_indicadores['RetornoDiario'].mean())**252) - 1, # Asumiendo 252 días de trading
            'volatilidad_anualizada_estimada': df_con_indicadores['RetornoDiario'].std() * np.sqrt(252)
        }
        # Añadir últimos valores de indicadores
        stats['precio_actual'] = df_con_indicadores['Adj Close'].iloc[-1] if not df_con_indicadores['Adj Close'].empty else np.nan
        stats[f'sma_corta_actual'] = df_con_indicadores[f'SMA_{SMA_CORTA_PERIODO}'].iloc[-1] if not df_con_indicadores[f'SMA_{SMA_CORTA_PERIODO}'].empty else np.nan
        # ... (añadir más indicadores actuales si se desea)

        resultados_analisis_individual[ticker] = {
            'dataframe_con_indicadores': df_con_indicadores,
            'estadisticas': pd.Series(stats) # Convertir a Series para mejor visualización luego
        }

        generar_graficos_activo(df_con_indicadores, ticker)
        print(f"  Análisis para {ticker} completado.")

    print("--- Módulo de Análisis por Activo Individual Finalizado ---")
    return resultados_analisis_individual

# ==============================================================================
# --- MÓDULO 3: ANÁLISIS Y OPTIMIZACIÓN DE CARTERA ---
# ==============================================================================
def calcular_retornos_diarios_cartera(df_precios_activos):
    """Calcula los retornos diarios para un DataFrame de precios de activos."""
    return df_precios_activos.pct_change().dropna()

def generar_matriz_correlacion(df_retornos_activos, ticker_list):
    """Genera y guarda el heatmap de la matriz de correlación."""
    print("  Generando matriz de correlación...")
    path_graficos_cartera = os.path.join(CARPETA_GRAFICOS_TEMP, "analisis_cartera")

    matriz_corr = df_retornos_activos.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_corr, annot=True, cmap=PALETA_COLORES_GRAFICOS, fmt=".2f", linewidths=.5)
    plt.title('Matriz de Correlación de Retornos Diarios de Activos', fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    ruta_guardado = os.path.join(path_graficos_cartera, "matriz_correlacion.png")
    plt.savefig(ruta_guardado, dpi=300)
    plt.close()
    print(f"  Matriz de correlación guardada en: {ruta_guardado}")
    return matriz_corr, ruta_guardado


def optimizar_cartera_pypfopt(df_precios_activos):
    """
    Optimiza la cartera utilizando PyPortfolioOpt según la configuración global.
    """
    print("  Iniciando optimización de cartera con PyPortfolioOpt...")

    # 1. Calcular retornos esperados y matriz de covarianza
    # mu = expected_returns.mean_historical_return(df_precios_activos, compounding=False, frequency=252) # Retorno simple anualizado
    mu = expected_returns.ema_historical_return(df_precios_activos, compounding=False, frequency=252, span=SMA_LARGA_PERIODO) # EMA para retornos esperados, más reactivo
    S = risk_models.CovarianceShrinkage(df_precios_activos, frequency=252).ledoit_wolf()
    # S = risk_models.sample_cov(df_precios_activos, frequency=252) # Covarianza simple

    # 2. Inicializar EfficientFrontier
    ef = EfficientFrontier(mu, S)
    # ef.add_objective(objective_functions.L2_reg) # Regularización para evitar pesos extremos, opcional

    # 3. Aplicar el método de optimización
    try:
        if METODO_OPTIMIZACION == "max_sharpe":
            print("    Optimizando para Máximo Ratio de Sharpe...")
            ef.max_sharpe(risk_free_rate=TASA_LIBRE_RIESGO)
        elif METODO_OPTIMIZACION == "min_volatility":
            print("    Optimizando para Mínima Volatilidad...")
            ef.min_volatility()
        elif METODO_OPTIMIZACION == "efficient_risk":
            print(f"    Optimizando para Retorno Objetivo: {RETORNO_OBJETIVO_OPTIMIZACION*100:.2f}%...")
            ef.efficient_risk(target_return=RETORNO_OBJETIVO_OPTIMIZACION, risk_free_rate=TASA_LIBRE_RIESGO)
        elif METODO_OPTIMIZACION == "efficient_return":
            print(f"    Optimizando para Volatilidad Objetivo: {VOLATILIDAD_OBJETIVO_OPTIMIZACION*100:.2f}%...")
            ef.efficient_return(target_volatility=VOLATILIDAD_OBJETIVO_OPTIMIZACION, risk_free_rate=TASA_LIBRE_RIESGO)
        else:
            print(f"    Método de optimización '{METODO_OPTIMIZACION}' no reconocido. Usando Máximo Ratio de Sharpe por defecto.")
            ef.max_sharpe(risk_free_rate=TASA_LIBRE_RIESGO)

        pesos_optimos_raw = ef.clean_weights() # Limpia pesos muy pequeños y normaliza
        rendimiento_esperado, volatilidad_anual, sharpe_ratio = ef.portfolio_performance(verbose=False, risk_free_rate=TASA_LIBRE_RIESGO)

        print("    Optimización completada.")
        print(f"    Rendimiento Esperado Anual: {rendimiento_esperado*100:.2f}%")
        print(f"    Volatilidad Anual: {volatilidad_anual*100:.2f}%")
        print(f"    Ratio de Sharpe: {sharpe_ratio:.2f}")

        return pesos_optimos_raw, rendimiento_esperado, volatilidad_anual, sharpe_ratio, mu, S

    except Exception as e:
        print(f"    Error durante la optimización ({METODO_OPTIMIZACION}): {e}")
        print("    Intentando con Máximo Ratio de Sharpe como fallback...")
        try:
            ef_fallback = EfficientFrontier(mu, S)
            ef_fallback.max_sharpe(risk_free_rate=TASA_LIBRE_RIESGO)
            pesos_optimos_raw = ef_fallback.clean_weights()
            rendimiento_esperado, volatilidad_anual, sharpe_ratio = ef_fallback.portfolio_performance(verbose=False, risk_free_rate=TASA_LIBRE_RIESGO)
            print("    Optimización fallback (Max Sharpe) completada.")
            return pesos_optimos_raw, rendimiento_esperado, volatilidad_anual, sharpe_ratio, mu, S
        except Exception as e_fallback:
            print(f"    Error durante la optimización fallback: {e_fallback}")
            return None, None, None, None, mu, S


def simulacion_monte_carlo_cartera(df_retornos_activos, num_simulaciones):
    """
    Realiza una simulación Monte Carlo para generar carteras aleatorias.
    """
    print(f"  Iniciando Simulación Monte Carlo con {num_simulaciones} carteras...")
    num_activos = df_retornos_activos.shape[1]
    resultados_mc = np.zeros((3 + num_activos, num_simulaciones)) # 3 para Ret, Vol, Sharpe + num_activos para pesos

    retornos_medios_anualizados = df_retornos_activos.mean() * 252
    matriz_cov_anualizada = df_retornos_activos.cov() * 252

    for i in range(num_simulaciones):
        # Generar pesos aleatorios
        pesos = np.random.random(num_activos)
        pesos /= np.sum(pesos) # Normalizar

        # Calcular retorno y volatilidad de la cartera
        retorno_cartera = np.sum(retornos_medios_anualizados * pesos)
        volatilidad_cartera = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov_anualizada, pesos)))

        # Guardar resultados
        resultados_mc[0,i] = retorno_cartera
        resultados_mc[1,i] = volatilidad_cartera
        resultados_mc[2,i] = (retorno_cartera - TASA_LIBRE_RIESGO) / volatilidad_cartera if volatilidad_cartera !=0 else 0
        for j in range(num_activos):
            resultados_mc[3+j,i] = pesos[j]

    # Convertir a DataFrame
    columnas_df_mc = ['Retorno', 'Volatilidad', 'SharpeRatio'] + [ticker for ticker in df_retornos_activos.columns]
    df_resultados_mc = pd.DataFrame(resultados_mc.T, columns=columnas_df_mc)

    # Encontrar la cartera con el máximo Sharpe Ratio de la simulación
    idx_max_sharpe_mc = df_resultados_mc['SharpeRatio'].idxmax()
    cartera_optima_mc = df_resultados_mc.iloc[idx_max_sharpe_mc]

    print("  Simulación Monte Carlo completada.")
    print(f"  Mejor Cartera MC - Retorno: {cartera_optima_mc['Retorno']*100:.2f}%, Vol: {cartera_optima_mc['Volatilidad']*100:.2f}%, Sharpe: {cartera_optima_mc['SharpeRatio']:.2f}")
    return df_resultados_mc, cartera_optima_mc

def generar_graficos_cartera(ef_mu, ef_S, pesos_optimos, df_resultados_mc=None, cartera_optima_mc=None, df_retornos_optima=None, df_retornos_benchmark=None):
    """Genera y guarda los gráficos relacionados con el análisis de cartera."""
    print("  Generando gráficos de análisis de cartera...")
    path_graficos_cartera = os.path.join(CARPETA_GRAFICOS_TEMP, "analisis_cartera")

    # 1. Frontera Eficiente y Portafolio Óptimo (PyPortfolioOpt)
    fig, ax = plt.subplots(figsize=(10, 7))
    ef_plot = EfficientFrontier(ef_mu, ef_S) # Recrear para plotear

    # Plotear la frontera eficiente
    plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=False, eh_color=PALETA_COLORES_GRAFICOS) # eh_color no es un param valido, se usará el global
    ax.set_title('Frontera Eficiente y Portafolio Óptimo', fontsize=16)

    # Plotear el portafolio óptimo de PyPortfolioOpt
    rend_opt, vol_opt, _ = ef_plot.portfolio_performance(risk_free_rate=TASA_LIBRE_RIESGO, weights=pesos_optimos)
    ax.scatter(vol_opt, rend_opt, marker='*', s=200, c='red', label='Portafolio Óptimo (PyOpt)', zorder=5)

    # 2. Si hay resultados de Monte Carlo, plotearlos
    if df_resultados_mc is not None and not df_resultados_mc.empty:
        ax.scatter(df_resultados_mc['Volatilidad'], df_resultados_mc['Retorno'], c=df_resultados_mc['SharpeRatio'], cmap='viridis', s=10, alpha=0.3, label='Carteras Monte Carlo')
        if cartera_optima_mc is not None:
            ax.scatter(cartera_optima_mc['Volatilidad'], cartera_optima_mc['Retorno'], marker='X', s=200, c='orange', label='Portafolio Óptimo (MC)', zorder=5)

    ax.set_xlabel('Volatilidad Anualizada')
    ax.set_ylabel('Retorno Anualizado Esperado')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    ruta_ef = os.path.join(path_graficos_cartera, "frontera_eficiente.png")
    plt.savefig(ruta_ef, dpi=300)
    plt.close(fig)
    print(f"  Gráfico de Frontera Eficiente guardado en: {ruta_ef}")

    # 3. Composición de la Cartera Óptima (Gráfico de Torta)
    if pesos_optimos:
        pesos_df = pd.Series(pesos_optimos).sort_values(ascending=False)
        pesos_significativos = pesos_df[pesos_df > 0.001] # Filtrar pesos muy pequeños para claridad

        if not pesos_significativos.empty:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(pesos_significativos, labels=pesos_significativos.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette(PALETA_COLORES_GRAFICOS, len(pesos_significativos)))
            ax.axis('equal') # Para que el gráfico sea circular
            plt.title('Composición de la Cartera Óptima', fontsize=16)
            plt.tight_layout()
            ruta_composicion = os.path.join(path_graficos_cartera, "composicion_cartera_optima.png")
            plt.savefig(ruta_composicion, dpi=300)
            plt.close(fig)
            print(f"  Gráfico de Composición de Cartera guardado en: {ruta_composicion}")
        else:
            print("  No hay pesos significativos para graficar la composición de la cartera.")


    # 4. Evolución Histórica Simulada de la Cartera Óptima vs Benchmark
    if df_retornos_optima is not None:
        evolucion_optima = (1 + df_retornos_optima).cumprod()

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(evolucion_optima.index, evolucion_optima, label='Cartera Óptima Simulada', linewidth=2)

        if df_retornos_benchmark is not None:
            evolucion_benchmark = (1 + df_retornos_benchmark).cumprod()
            ax.plot(evolucion_benchmark.index, evolucion_benchmark.iloc[:,0], label=f'Benchmark ({BENCHMARK_TICKER})', linestyle='--', linewidth=2)

        plt.title('Evolución Histórica Simulada de la Cartera Óptima vs. Benchmark', fontsize=16)
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Acumulado (Base 1)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        ruta_evolucion = os.path.join(path_graficos_cartera, "evolucion_cartera_vs_benchmark.png")
        plt.savefig(ruta_evolucion, dpi=300)
        plt.close(fig)
        print(f"  Gráfico de Evolución de Cartera guardado en: {ruta_evolucion}")


def analizar_y_optimizar_cartera(df_precios_activos, df_precios_benchmark=None):
    """
    Orquesta el análisis y la optimización de la cartera.
    """
    print("\n--- Iniciando Módulo de Análisis y Optimización de Cartera ---")
    if df_precios_activos.empty or len(df_precios_activos.columns) < 2:
        print("  No hay suficientes activos (se requieren al menos 2) o datos para el análisis de cartera. Saltando.")
        return None

    # 1. Calcular retornos diarios de los activos
    df_retornos_activos = calcular_retornos_diarios_cartera(df_precios_activos)
    if df_retornos_activos.empty:
        print("  No se pudieron calcular los retornos de los activos. Saltando análisis de cartera.")
        return None

    # 2. Generar Matriz de Correlación
    matriz_corr, _ = generar_matriz_correlacion(df_retornos_activos, list(df_precios_activos.columns))

    # 3. Optimizar Cartera con PyPortfolioOpt
    resultados_optimizacion = optimizar_cartera_pypfopt(df_precios_activos)
    pesos_optimos, rend_esp_opt, vol_anual_opt, sharpe_opt, ef_mu, ef_S = resultados_optimizacion

    if pesos_optimos is None:
        print("  Fallo en la optimización de la cartera. No se pueden generar más resultados para la cartera.")
        return {'matriz_correlacion': matriz_corr}


    # 4. Simulación Monte Carlo (Opcional)
    df_mc_resultados = None
    cartera_mc_optima = None
    if ACTIVAR_MONTE_CARLO:
        df_mc_resultados, cartera_mc_optima = simulacion_monte_carlo_cartera(df_retornos_activos, NUM_SIMULACIONES_MC)

    # 5. Calcular retornos de la cartera óptima para simulación histórica
    df_retornos_cartera_optima = None
    if pesos_optimos:
        # Asegurarse de que los pesos_optimos (dict) y df_retornos_activos (DataFrame) tengan los mismos tickers
        # y en el mismo orden para la multiplicación.
        pesos_serie = pd.Series(pesos_optimos)
        # Alinear los retornos con los pesos (importante si algunos tickers no están en pesos_optimos o viceversa)
        retornos_alineados, pesos_alineados = df_retornos_activos.align(pesos_serie, axis=1, join='inner')
        df_retornos_cartera_optima = (retornos_alineados * pesos_alineados).sum(axis=1)

    # 6. Calcular retornos del benchmark
    df_retornos_benchmark_data = None
    if df_precios_benchmark is not None and not df_precios_benchmark.empty:
        df_retornos_benchmark_data = df_precios_benchmark.pct_change().dropna()

    # 7. Generar Gráficos de Cartera
    generar_graficos_cartera(
        ef_mu=ef_mu,
        ef_S=ef_S,
        pesos_optimos=pesos_optimos,
        df_resultados_mc=df_mc_resultados,
        cartera_optima_mc=cartera_mc_optima,
        df_retornos_optima=df_retornos_cartera_optima,
        df_retornos_benchmark=df_retornos_benchmark_data
    )

    resultados_finales_cartera = {
        'matriz_correlacion': matriz_corr,
        'pesos_optimos': pesos_optimos,
        'rendimiento_esperado_optimo': rend_esp_opt,
        'volatilidad_anual_optima': vol_anual_opt,
        'sharpe_ratio_optimo': sharpe_opt,
        'dataframe_monte_carlo': df_mc_resultados, # Puede ser None
        'cartera_optima_monte_carlo': cartera_mc_optima, # Puede ser None
        'retornos_diarios_cartera_optima': df_retornos_cartera_optima # Puede ser None
    }

    print("--- Módulo de Análisis y Optimización de Cartera Finalizado ---")
    return resultados_finales_cartera


# ==============================================================================
# --- MÓDULO 4: GENERACIÓN DE REPORTE PDF ---
# ==============================================================================
class ReportePDF(FPDF):
    def __init__(self, orientation='P', unit='mm', format='A4', nombre_autor="Equipo de Análisis"):
        super().__init__(orientation, unit, format)
        self.nombre_autor = nombre_autor
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        # Registrar fuentes (asegúrate de que las rutas a los .ttf sean correctas o usa fuentes estándar)
        # Si no tienes DejaVuSans, FPDF usará una fuente por defecto como Arial/Helvetica.
        try:
            # Intentar añadir fuentes personalizadas si existen (ej. en una carpeta 'fonts')
            # self.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
            # self.add_font('DejaVu', 'B', 'fonts/DejaVuSans-Bold.ttf', uni=True)
            self.set_font('Arial', '', 10) # Fuente por defecto si DejaVu no está
        except RuntimeError:
            print("Advertencia: Fuentes DejaVu no encontradas. Usando fuentes estándar de FPDF.")
            self.set_font('Arial', '', 10)


    def header(self):
        if self.page_no() == 1: # No header en la portada
            return
        self.set_font('Arial', 'B', 10)
        # self.set_font('DejaVu', 'B', 10) # Si usas DejaVu
        w = self.get_string_width(f"Reporte de Análisis de Cartera - {datetime.datetime.now().strftime('%Y-%m-%d')}") + 6
        self.set_x((self.w - w) / 2)
        self.cell(w, 9, f"Reporte de Análisis de Cartera - {datetime.datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'C')
        self.ln(5)

    def footer(self):
        if self.page_no() == 1: # No footer en la portada
            return
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        # self.set_font('DejaVu', 'I', 8) # Si usas DejaVu
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}} | {self.nombre_autor}', 0, 0, 'C')

    def chapter_title(self, title_text, level=1):
        self.ln(8)
        if level == 1:
            self.set_font('Arial', 'B', 16)
            # self.set_font('DejaVu', 'B', 16)
            self.set_fill_color(200, 220, 255) # Color de fondo para títulos principales
            self.cell(0, 10, title_text, 0, 1, 'L', fill=True)
        elif level == 2:
            self.set_font('Arial', 'B', 14)
            # self.set_font('DejaVu', 'B', 14)
            self.cell(0, 8, title_text, 0, 1, 'L')
        else: # level 3 o más
            self.set_font('Arial', 'B', 12)
            # self.set_font('DejaVu', 'B', 12)
            self.cell(0, 7, title_text, 0, 1, 'L')
        self.ln(4)
        self.set_font('Arial', '', 10) # Reset a fuente normal
        # self.set_font('DejaVu', '', 10)

    def chapter_body(self, body_text, text_align='L'):
        self.set_font('Arial', '', 10)
        # self.set_font('DejaVu', '', 10)
        if isinstance(body_text, list): # Si es una lista de strings (párrafos)
            for paragraph in body_text:
                self.multi_cell(0, 6, paragraph, 0, text_align)
                self.ln(2)
        else:
            self.multi_cell(0, 6, body_text, 0, text_align)
        self.ln(4)

    def add_image_from_path(self, image_path, caption="", width_pct=0.8):
        if not os.path.exists(image_path):
            self.chapter_body(f"[Error: Imagen no encontrada en {image_path}]")
            return

        page_width = self.w - self.l_margin - self.r_margin
        img_width = page_width * width_pct

        try:
            # Obtener dimensiones de la imagen para mantener la proporción
            from PIL import Image as PILImage
            with PILImage.open(image_path) as img:
                aspect_ratio = img.height / img.width
                img_height = img_width * aspect_ratio

            # Centrar imagen
            x_pos = (self.w - img_width) / 2
            self.image(image_path, x=x_pos, w=img_width, h=img_height)
            self.ln(img_height + 2) # Espacio después de la imagen

            if caption:
                self.set_font('Arial', 'I', 9)
                # self.set_font('DejaVu', 'I', 9)
                self.multi_cell(0, 5, caption, 0, 'C')
                self.ln(4)
        except ImportError:
            self.chapter_body("[Nota: PIL/Pillow no está instalado, no se puede obtener aspect ratio de imagen. Usando ancho fijo.]")
            self.image(image_path, w=img_width) # Usar solo ancho si PIL no está
            self.ln(5)


    def add_table_from_dataframe(self, df, title=""):
        if title:
            self.chapter_title(title, level=3)

        if df is None or df.empty:
            self.chapter_body("No hay datos disponibles para esta tabla.")
            return

        self.set_font('Arial', 'B', 8)
        # self.set_font('DejaVu', 'B', 8)

        # Ancho de columnas (aproximado, se puede mejorar)
        num_cols = len(df.columns) + (1 if df.index.name or df.index.nlevels > 1 else 0)
        available_width = self.w - self.l_margin - self.r_margin
        col_width = available_width / num_cols if num_cols > 0 else available_width
        line_height = 6

        # Encabezados
        if df.index.name:
            self.cell(col_width, line_height, str(df.index.name), 1, 0, 'C', fill=True)
        elif df.index.nlevels > 1: # MultiIndex
             for i, name in enumerate(df.index.names):
                self.cell(col_width/df.index.nlevels if df.index.nlevels else col_width, line_height, str(name), 1, 0, 'C', fill=True)

        for col_name in df.columns:
            self.cell(col_width, line_height, str(col_name), 1, 0, 'C', fill=True)
        self.ln(line_height)

        # Datos
        self.set_font('Arial', '', 7)
        # self.set_font('DejaVu', '', 7)
        for index, row in df.iterrows():
            if isinstance(index, tuple): # MultiIndex
                 for idx_part in index:
                    self.cell(col_width/len(index) if len(index) else col_width, line_height, str(idx_part)[:15], 1, 0, 'L') # Truncar texto largo
            else:
                self.cell(col_width, line_height, str(index)[:15], 1, 0, 'L') # Truncar

            for val in row:
                if isinstance(val, float):
                    self.cell(col_width, line_height, f"{val:.4f}", 1, 0, 'R')
                else:
                    self.cell(col_width, line_height, str(val)[:15], 1, 0, 'L') # Truncar
            self.ln(line_height)
        self.ln(4)

def generar_reporte_pdf(
    nombre_archivo_salida,
    lista_tickers_analizados,
    fecha_inicio_str,
    fecha_fin_str,
    resultados_analisis_individuales,
    resultados_opt_cartera
    ):
    """
    Genera el reporte PDF completo.
    """
    print("\n--- Iniciando Módulo de Generación de Reporte PDF ---")
    pdf = ReportePDF(nombre_autor="Jules IA") # Puedes poner tu nombre o el del script
    pdf.alias_nb_pages() # Para tener el total de páginas en el footer {nb}

    # --- 1. PORTADA ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    # pdf.set_font('DejaVu', 'B', 24)
    pdf.cell(0, 20, "Análisis Cuantitativo Integrado de Cartera", 0, 1, 'C')
    pdf.ln(10)

    pdf.set_font('Arial', 'B', 14)
    # pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, "Activos Analizados:", 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    # pdf.set_font('DejaVu', '', 10)
    pdf.multi_cell(0, 6, ", ".join(lista_tickers_analizados), 0, 'L')
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 14)
    # pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, "Periodo de Análisis:", 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    # pdf.set_font('DejaVu', '', 10)
    pdf.cell(0, 6, f"Desde: {fecha_inicio_str} Hasta: {fecha_fin_str}", 0, 1, 'L')
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 14)
    # pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, "Fecha de Generación del Reporte:", 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    # pdf.set_font('DejaVu', '', 10)
    pdf.cell(0, 6, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0, 1, 'L')
    pdf.ln(15)

    pdf.set_font('Arial', 'I', 10)
    # pdf.set_font('DejaVu', 'I', 10)
    pdf.multi_cell(0, 6, "Este reporte fue generado automáticamente y presenta un análisis cuantitativo basado en datos históricos. No constituye recomendación de inversión.", 0, 'C')


    # --- 2. RESUMEN EJECUTIVO (si hay datos de cartera) ---
    if resultados_opt_cartera and resultados_opt_cartera.get('pesos_optimos'):
        pdf.add_page()
        pdf.chapter_title("Resumen Ejecutivo", level=1)

        res_pesos = resultados_opt_cartera['pesos_optimos']
        res_rend = resultados_opt_cartera.get('rendimiento_esperado_optimo', 0)
        res_vol = resultados_opt_cartera.get('volatilidad_anual_optima', 0)
        res_sharpe = resultados_opt_cartera.get('sharpe_ratio_optimo', 0)

        summary_text = [
            f"Se ha realizado un análisis cuantitativo para una cartera compuesta por los activos: {', '.join(lista_tickers_analizados)}.",
            f"El periodo de datos históricos considerado es desde {fecha_inicio_str} hasta {fecha_fin_str}.",
            f"El método de optimización de cartera utilizado fue: '{METODO_OPTIMIZACION}'.",
            "Resultados Clave de la Cartera Optimizada:",
            f"  - Rendimiento Anualizado Esperado: {res_rend*100:.2f}%",
            f"  - Volatilidad Anualizada Esperada: {res_vol*100:.2f}%",
            f"  - Ratio de Sharpe Esperado: {res_sharpe:.2f}",
            "La composición de la cartera óptima (pesos significativos) es la siguiente:"
        ]
        pdf.chapter_body(summary_text)

        pesos_significativos_df = pd.DataFrame.from_dict(
            {t: p for t, p in res_pesos.items() if p > 0.001},
            orient='index', columns=['Peso']
        )
        pesos_significativos_df['Peso'] = pesos_significativos_df['Peso'].apply(lambda x: f"{x*100:.2f}%")
        pesos_significativos_df.index.name = "Activo"
        if not pesos_significativos_df.empty:
            pdf.add_table_from_dataframe(pesos_significativos_df.sort_values(by="Peso", ascending=False))
        else:
            pdf.chapter_body("No se encontraron pesos significativos para la cartera óptima.")

    # --- 3. ANÁLISIS POR ACTIVO INDIVIDUAL ---
    if resultados_analisis_individuales:
        pdf.add_page()
        pdf.chapter_title("Análisis por Activo Individual", level=1)

        for ticker, data_activo in resultados_analisis_individuales.items():
            pdf.chapter_title(f"Activo: {ticker}", level=2)

            # Mostrar algunas estadísticas clave del activo
            stats_df = data_activo['estadisticas'].to_frame(name='Valor')
            # Formatear algunos valores para mejor lectura en PDF
            if 'retorno_anualizado_estimado' in stats_df.index:
                stats_df.loc['retorno_anualizado_estimado', 'Valor'] = f"{stats_df.loc['retorno_anualizado_estimado', 'Valor']*100:.2f}%"
            if 'volatilidad_anualizada_estimada' in stats_df.index:
                 stats_df.loc['volatilidad_anualizada_estimada', 'Valor'] = f"{stats_df.loc['volatilidad_anualizada_estimada', 'Valor']*100:.2f}%"
            if 'precio_actual' in stats_df.index:
                 stats_df.loc['precio_actual', 'Valor'] = f"{stats_df.loc['precio_actual', 'Valor']:.2f}"

            pdf.add_table_from_dataframe(stats_df)

            # Añadir gráficos del activo
            graficos_a_incluir = [
                (f"{ticker}_precio_volumen.png", f"Gráfico de Precio, SMAs y Volumen para {ticker}"),
                (f"{ticker}_rsi.png", f"Índice de Fuerza Relativa (RSI) para {ticker}" if ACTIVAR_RSI else None),
                (f"{ticker}_macd.png", f"MACD para {ticker}" if ACTIVAR_MACD else None),
                (f"{ticker}_distribucion_retornos.png", f"Distribución de Retornos Diarios para {ticker}")
            ]
            for img_file, caption in graficos_a_incluir:
                if caption is None: continue # Si el indicador no estaba activo, el gráfico no existe
                path_completo_img = os.path.join(CARPETA_GRAFICOS_TEMP, "activos_individuales", img_file)
                if os.path.exists(path_completo_img):
                    pdf.add_image_from_path(path_completo_img, caption=caption, width_pct=0.9)
                else:
                    pdf.chapter_body(f"[Gráfico {img_file} no encontrado]", text_align='C')
            pdf.add_page() # Nueva página para el siguiente activo o sección

    # --- 4. ANÁLISIS DE CARTERA ---
    if resultados_opt_cartera:
        # (La página ya se añadió si hay análisis individual, o se añade aquí si no lo hubo)
        if not resultados_analisis_individuales: pdf.add_page()

        pdf.chapter_title("Análisis y Optimización de Cartera", level=1)

        # Matriz de Correlación
        pdf.chapter_title("Matriz de Correlación", level=2)
        path_matriz_corr = os.path.join(CARPETA_GRAFICOS_TEMP, "analisis_cartera", "matriz_correlacion.png")
        pdf.add_image_from_path(path_matriz_corr, caption="Heatmap de Correlación de Retornos Diarios", width_pct=0.7)

        # Optimización de Cartera
        pdf.chapter_title("Resultados de la Optimización", level=2)
        pdf.chapter_body(f"Método de Optimización Aplicado: {METODO_OPTIMIZACION}")

        path_frontera_ef = os.path.join(CARPETA_GRAFICOS_TEMP, "analisis_cartera", "frontera_eficiente.png")
        pdf.add_image_from_path(path_frontera_ef, caption="Frontera Eficiente y Portafolio(s) Óptimo(s)", width_pct=0.9)

        # Tabla de Métricas de Cartera Optimizada
        metricas_opt_df = pd.DataFrame({
            'Métrica': ['Rendimiento Anualizado Esperado', 'Volatilidad Anualizada Esperada', 'Ratio de Sharpe Esperado'],
            'Valor': [
                f"{resultados_opt_cartera.get('rendimiento_esperado_optimo', 0)*100:.2f}%",
                f"{resultados_opt_cartera.get('volatilidad_anual_optima', 0)*100:.2f}%",
                f"{resultados_opt_cartera.get('sharpe_ratio_optimo', 0):.2f}"
            ]
        }).set_index('Métrica')
        pdf.add_table_from_dataframe(metricas_opt_df, title="Métricas de la Cartera Optimizada (PyPortfolioOpt)")

        # Composición de Cartera
        path_composicion = os.path.join(CARPETA_GRAFICOS_TEMP, "analisis_cartera", "composicion_cartera_optima.png")
        pdf.add_image_from_path(path_composicion, caption="Composición de la Cartera Óptima (Pesos)", width_pct=0.6)

        # Evolución Histórica
        path_evolucion = os.path.join(CARPETA_GRAFICOS_TEMP, "analisis_cartera", "evolucion_cartera_vs_benchmark.png")
        pdf.add_image_from_path(path_evolucion, caption="Evolución Histórica Simulada de la Cartera Óptima vs. Benchmark", width_pct=0.9)

        # (Opcional) Resultados de Monte Carlo
        if ACTIVAR_MONTE_CARLO and resultados_opt_cartera.get('cartera_optima_monte_carlo') is not None:
            pdf.add_page()
            pdf.chapter_title("Resultados de Simulación Monte Carlo", level=2)
            mc_opt = resultados_opt_cartera['cartera_optima_monte_carlo']
            mc_text = [
                f"Se realizaron {NUM_SIMULACIONES_MC} simulaciones de Monte Carlo.",
                "La cartera con el Ratio de Sharpe más alto encontrada en la simulación tiene las siguientes métricas:",
                f"  - Rendimiento Anualizado: {mc_opt['Retorno']*100:.2f}%",
                f"  - Volatilidad Anualizada: {mc_opt['Volatilidad']*100:.2f}%",
                f"  - Ratio de Sharpe: {mc_opt['SharpeRatio']:.2f}"
            ]
            pdf.chapter_body(mc_text)

            pesos_mc_dict = {t: mc_opt[t] for t in lista_tickers_analizados if t in mc_opt and mc_opt[t] > 0.001}
            if pesos_mc_dict:
                pesos_mc_df = pd.DataFrame.from_dict(pesos_mc_dict, orient='index', columns=['Peso'])
                pesos_mc_df['Peso'] = pesos_mc_df['Peso'].apply(lambda x: f"{x*100:.2f}%")
                pesos_mc_df.index.name = "Activo"
                pdf.add_table_from_dataframe(pesos_mc_df.sort_values(by="Peso", ascending=False), title="Composición Cartera Óptima (Monte Carlo)")


    # --- GUARDAR PDF ---
    try:
        pdf.output(nombre_archivo_salida, 'F')
        print(f"✅ Reporte PDF generado exitosamente: {nombre_archivo_salida}")
    except Exception as e:
        print(f"❌ Error al generar el PDF: {e}")

    print("--- Módulo de Generación de Reporte PDF Finalizado ---")


# ==============================================================================
# --- MÓDULO 5: GENERACIÓN DE REPORTE EXCEL ---
# ==============================================================================
def generar_reporte_excel(
    nombre_archivo_salida,
    lista_tickers_analizados, # Lista de tickers que realmente tienen datos y fueron analizados
    resultados_analisis_individuales, # Dict con 'dataframe_con_indicadores' y 'estadisticas'
    resultados_opt_cartera, # Dict con resultados de optimización
    df_activos_precios_para_cartera, # DataFrame con precios ajustados para el análisis de cartera
    df_retornos_activos_cartera # DataFrame con retornos diarios de activos para cartera
    ):
    """
    Genera un reporte Excel con múltiples hojas detallando el análisis.
    """
    print("\n--- Iniciando Módulo de Generación de Reporte Excel ---")

    try:
        with pd.ExcelWriter(nombre_archivo_salida, engine='openpyxl') as writer:

            # --- Hoja 1: Resumen General ---
            print("  Creando Hoja: Resumen General...")
            sheet_name_resumen = "Resumen General"

            # Estadísticas individuales
            stats_individuales_list = []
            if resultados_analisis_individuales:
                for ticker in lista_tickers_analizados: # Iterar en el orden original si es posible
                    if ticker in resultados_analisis_individuales:
                        stats_activo = resultados_analisis_individuales[ticker]['estadisticas'].copy()
                        stats_activo.name = ticker # El nombre de la Serie será el ticker
                        stats_individuales_list.append(stats_activo)

            if stats_individuales_list:
                df_resumen_individual = pd.DataFrame(stats_individuales_list)
                df_resumen_individual.index.name = "Activo"
                df_resumen_individual.to_excel(writer, sheet_name=sheet_name_resumen, startrow=0, startcol=0)
                current_row = len(df_resumen_individual) + 3 # Espacio para la siguiente tabla
            else:
                pd.DataFrame({"Info": ["No hay estadísticas individuales disponibles"]}).to_excel(writer, sheet_name=sheet_name_resumen, startrow=0, startcol=0)
                current_row = 3

            # Métricas y Pesos de Cartera Optimizada
            if resultados_opt_cartera and resultados_opt_cartera.get('pesos_optimos'):
                df_metricas_cartera = pd.DataFrame({
                    'Métrica': ['Rendimiento Anualizado Esperado', 'Volatilidad Anualizada Esperada', 'Ratio de Sharpe Esperado'],
                    'Valor': [
                        resultados_opt_cartera.get('rendimiento_esperado_optimo', np.nan),
                        resultados_opt_cartera.get('volatilidad_anual_optima', np.nan),
                        resultados_opt_cartera.get('sharpe_ratio_optimo', np.nan)
                    ]
                }).set_index('Métrica')
                df_metricas_cartera.to_excel(writer, sheet_name=sheet_name_resumen, startrow=current_row, startcol=0)
                current_row += len(df_metricas_cartera) + 2

                df_pesos_optimos = pd.Series(resultados_opt_cartera['pesos_optimos'], name="Peso Óptimo").to_frame()
                df_pesos_optimos.index.name = "Activo"
                df_pesos_optimos[df_pesos_optimos["Peso Óptimo"] > 1e-5].sort_values(by="Peso Óptimo", ascending=False).to_excel(
                    writer, sheet_name=sheet_name_resumen, startrow=current_row, startcol=0
                )
            else:
                 pd.DataFrame({"Info": ["No hay resultados de optimización de cartera disponibles"]}).to_excel(
                     writer, sheet_name=sheet_name_resumen, startrow=current_row, startcol=0
                 )

            # --- Hojas por Activo Individual ---
            if resultados_analisis_individuales:
                for ticker in lista_tickers_analizados:
                    if ticker in resultados_analisis_individuales:
                        print(f"  Creando Hoja para activo: {ticker}...")
                        df_activo_completo = resultados_analisis_individuales[ticker]['dataframe_con_indicadores']
                        df_activo_completo.to_excel(writer, sheet_name=f"Datos_{ticker}", index=True)

            # --- Hoja: Datos de Cartera ---
            print("  Creando Hoja: Datos de Cartera...")
            sheet_name_datos_cartera = "Datos Cartera"
            start_datos_cartera_row = 0
            if df_activos_precios_para_cartera is not None and not df_activos_precios_para_cartera.empty:
                df_activos_precios_para_cartera.to_excel(writer, sheet_name=sheet_name_datos_cartera, startrow=start_datos_cartera_row, startcol=0, sheet_name_prefix="Precios_")
                start_datos_cartera_row += len(df_activos_precios_para_cartera) + 3
            else:
                pd.DataFrame({"Info": ["No hay datos de precios para la cartera"]}).to_excel(writer, sheet_name=sheet_name_datos_cartera, startrow=start_datos_cartera_row, startcol=0)
                start_datos_cartera_row += 3

            if df_retornos_activos_cartera is not None and not df_retornos_activos_cartera.empty:
                 df_retornos_activos_cartera.to_excel(writer, sheet_name=sheet_name_datos_cartera, startrow=start_datos_cartera_row, startcol=0, sheet_name_prefix="Retornos_")
                 start_datos_cartera_row += len(df_retornos_activos_cartera) + 3
            else:
                pd.DataFrame({"Info": ["No hay datos de retornos para la cartera"]}).to_excel(writer, sheet_name=sheet_name_datos_cartera, startrow=start_datos_cartera_row, startcol=0)
                start_datos_cartera_row += 3

            if resultados_opt_cartera and 'matriz_correlacion' in resultados_opt_cartera:
                mat_corr = resultados_opt_cartera['matriz_correlacion']
                if mat_corr is not None and not mat_corr.empty:
                    mat_corr.to_excel(writer, sheet_name=sheet_name_datos_cartera, startrow=start_datos_cartera_row, startcol=0, sheet_name_prefix="Correlacion_")
            else:
                 pd.DataFrame({"Info": ["No hay matriz de correlación disponible"]}).to_excel(writer, sheet_name=sheet_name_datos_cartera, startrow=start_datos_cartera_row, startcol=0)


            # --- Hoja: Optimización Monte Carlo (Opcional) ---
            if ACTIVAR_MONTE_CARLO and resultados_opt_cartera and resultados_opt_cartera.get('dataframe_monte_carlo') is not None:
                print("  Creando Hoja: Optimización Monte Carlo...")
                df_mc = resultados_opt_cartera['dataframe_monte_carlo']
                if not df_mc.empty:
                    # Guardar una selección (ej. top 10 por Sharpe) para no hacer el archivo muy grande
                    df_mc.nlargest(100, 'SharpeRatio').to_excel(writer, sheet_name="MonteCarlo_Top100_Sharpe", index=False)

                    cartera_opt_mc = resultados_opt_cartera.get('cartera_optima_monte_carlo')
                    if cartera_opt_mc is not None:
                        pd.DataFrame(cartera_opt_mc).to_excel(writer, sheet_name="MonteCarlo_Optima", header=True)
                else:
                    pd.DataFrame({"Info": ["No hay resultados de Monte Carlo disponibles"]}).to_excel(writer, sheet_name="MonteCarlo_Info")

        print(f"✅ Reporte Excel generado exitosamente: {nombre_archivo_salida}")

    except Exception as e:
        print(f"❌ Error al generar el reporte Excel: {e}")

    print("--- Módulo de Generación de Reporte Excel Finalizado ---")

# ==============================================================================
# --- FUNCIÓN PRINCIPAL (MAIN) ---
# ==============================================================================
def main():
    """Función principal que orquesta el análisis."""
    print("Iniciando el Análisis Cuantitativo Integrado de Carteras...")

    # 1. Crear carpetas y configurar estilo
    crear_carpetas_necesarias()
    configurar_estilo_graficos()

    # 2. Obtener datos
    # Combinar la lista de tickers principales con el benchmark para descargarlos todos juntos
    todos_los_tickers_a_obtener = LISTA_TICKERS[:]
    if BENCHMARK_TICKER and BENCHMARK_TICKER not in todos_los_tickers_a_obtener:
        todos_los_tickers_a_obtener.append(BENCHMARK_TICKER)

    datos_historicos_todos = obtener_datos_activos(
        tickers_solicitados=todos_los_tickers_a_obtener,
        fecha_inicio=FECHA_INICIO_DATOS,
        fecha_fin=FECHA_FIN_DATOS,
        usar_cache=USAR_CACHE_DATOS,
        carpeta_cache=CARPETA_DATOS_ENTRADA
    )

    if not datos_historicos_todos: # Si el diccionario está vacío
        print("Error crítico: No se pudieron obtener datos para ningún activo. El análisis no puede continuar.")
        return

    # Separar los datos del benchmark de los datos de los activos
    datos_activos_analisis = {
        ticker: df for ticker, df in datos_historicos_todos.items() if ticker in LISTA_TICKERS and ticker != BENCHMARK_TICKER
    }
    df_benchmark_data = datos_historicos_todos.get(BENCHMARK_TICKER)

    # Validar que tengamos datos para los tickers principales
    if not datos_activos_analisis:
        print("Error crítico: No se pudieron obtener datos para los activos principales de la LISTA_TICKERS. El análisis no puede continuar.")
        return

    print(f"\nDatos cargados para {len(datos_activos_analisis)} activos principales.")
    if df_benchmark_data is not None:
        print(f"Datos cargados para el benchmark {BENCHMARK_TICKER}.")
    elif BENCHMARK_TICKER:
        print(f"Advertencia: No se pudieron cargar datos para el benchmark {BENCHMARK_TICKER}.")


    # 3. Análisis por activo
    resultados_individuales = analizar_activos_individuales(datos_activos_analisis)

    if not resultados_individuales:
        print("Advertencia: No se generaron resultados del análisis individual de activos.")
        # Considerar si continuar o no. Por ahora, continuamos.

    # Extraer DataFrames de precios ajustados para el módulo de cartera
    # Esto es necesario porque el módulo de cartera espera un solo DataFrame con precios ajustados
    lista_df_precios_ajustados = []
    for ticker in LISTA_TICKERS: # Iterar sobre la lista original para mantener el orden deseado
        if ticker in datos_activos_analisis and 'Adj Close' in datos_activos_analisis[ticker].columns:
            df_precio_ticker = datos_activos_analisis[ticker][['Adj Close']].copy()
            df_precio_ticker.rename(columns={'Adj Close': ticker}, inplace=True)
            lista_df_precios_ajustados.append(df_precio_ticker)

    df_activos_precios_para_cartera = pd.DataFrame()
    if lista_df_precios_ajustados:
        df_activos_precios_para_cartera = pd.concat(lista_df_precios_ajustados, axis=1)
        df_activos_precios_para_cartera.sort_index(inplace=True)
        # Interpolar y rellenar nuevamente por si acaso la concatenación introduce NaNs
        df_activos_precios_para_cartera.interpolate(method='linear', axis=0, inplace=True)
        df_activos_precios_para_cartera.fillna(method='bfill', inplace=True)
        df_activos_precios_para_cartera.fillna(method='ffill', inplace=True)


    df_benchmark_precios_adj = None
    if df_benchmark_data is not None and 'Adj Close' in df_benchmark_data.columns:
        df_benchmark_precios_adj = df_benchmark_data[['Adj Close']].copy()
        df_benchmark_precios_adj.rename(columns={'Adj Close': BENCHMARK_TICKER}, inplace=True)
        df_benchmark_precios_adj.sort_index(inplace=True)
        df_benchmark_precios_adj.interpolate(method='linear', axis=0, inplace=True)
        df_benchmark_precios_adj.fillna(method='bfill', inplace=True)
        df_benchmark_precios_adj.fillna(method='ffill', inplace=True)


    # 4. Análisis y optimización de cartera
    resultados_cartera = None
    # Determinar la lista real de tickers para los que se obtuvieron datos y se analizaron individualmente
    tickers_analizados_validos = list(resultados_individuales.keys()) if resultados_individuales else []

    if df_activos_precios_para_cartera.empty or len(df_activos_precios_para_cartera.columns) < 2:
        print("No hay suficientes datos de precios de activos (se requieren al menos 2 tickers válidos) para el análisis de cartera. Saltando esta sección.")
    else:
        print("\n--- DataFrame de Precios Ajustados para Análisis de Cartera (usando tickers válidos) ---")
        # Asegurarse de que df_activos_precios_para_cartera solo contenga tickers válidos
        df_activos_precios_para_cartera_valida = df_activos_precios_para_cartera[tickers_analizados_validos]
        print(df_activos_precios_para_cartera_valida.head())

        if df_benchmark_precios_adj is not None:
            print("\n--- DataFrame de Precios Ajustados para Benchmark ---")
            print(df_benchmark_precios_adj.head())

        resultados_cartera = analizar_y_optimizar_cartera(
            df_activos_precios_para_cartera_valida,
            df_benchmark_precios_adj
        )
        if resultados_cartera:
            print("\n--- Resultados de Optimización de Cartera ---")
            if resultados_cartera.get('pesos_optimos'):
                print("Pesos Óptimos:")
                for ticker_opt, peso_opt in resultados_cartera['pesos_optimos'].items():
                    if peso_opt > 1e-4: # Mostrar solo pesos significativos
                         print(f"  {ticker_opt}: {peso_opt*100:.2f}%")
                print(f"Rendimiento Esperado: {resultados_cartera.get('rendimiento_esperado_optimo', 0)*100:.2f}%")
                print(f"Volatilidad Anual: {resultados_cartera.get('volatilidad_anual_optima', 0)*100:.2f}%")
                print(f"Ratio de Sharpe: {resultados_cartera.get('sharpe_ratio_optimo', 0):.2f}")

    # 5. Generar reporte PDF
    nombre_pdf = os.path.join(CARPETA_REPORTES, f"{NOMBRE_BASE_REPORTE}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    generar_reporte_pdf(
        nombre_archivo_salida=nombre_pdf,
        lista_tickers_analizados=tickers_analizados_validos,
        fecha_inicio_str=FECHA_INICIO_DATOS,
        fecha_fin_str=FECHA_FIN_DATOS,
        resultados_analisis_individuales=resultados_individuales,
        resultados_opt_cartera=resultados_cartera
    )

    # 6. Generar reporte Excel
    nombre_excel = os.path.join(CARPETA_REPORTES, f"{NOMBRE_BASE_REPORTE}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    # Necesitamos los retornos diarios de los activos para la hoja "Datos Cartera"
    df_retornos_activos_cartera_excel = None
    if not df_activos_precios_para_cartera_valida.empty: # df_activos_precios_para_cartera_valida ya está filtrada
        df_retornos_activos_cartera_excel = calcular_retornos_diarios_cartera(df_activos_precios_para_cartera_valida)

    generar_reporte_excel(
        nombre_archivo_salida=nombre_excel,
        lista_tickers_analizados=tickers_analizados_validos,
        resultados_analisis_individuales=resultados_individuales,
        resultados_opt_cartera=resultados_cartera,
        df_activos_precios_para_cartera=df_activos_precios_para_cartera_valida, # Pasar el DF de precios ya filtrado
        df_retornos_activos_cartera=df_retornos_activos_cartera_excel
    )

    print("Análisis completado.")
    print(f"Los reportes se han guardado en la carpeta: {CARPETA_REPORTES}")

if __name__ == "__main__":
    # Validar configuración básica antes de empezar
    if not LISTA_TICKERS:
        print("Error: La lista de tickers (LISTA_TICKERS) no puede estar vacía.")
    elif not FECHA_INICIO_DATOS:
        print("Error: La fecha de inicio (FECHA_INICIO_DATOS) no puede estar vacía.")
    else:
        main()
