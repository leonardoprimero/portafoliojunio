import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Reimplementar funciones aquí mismo en lugar de importarlas de analisis_cartera

def calcular_sharpe_ratio(retornos_diarios):
    media = retornos_diarios.mean()
    desviacion = retornos_diarios.std()
    if desviacion == 0:
        return np.nan
    return media / desviacion * np.sqrt(252)

def calcular_volatilidad(retornos_diarios):
    return retornos_diarios.std() * np.sqrt(252)

def calcular_retorno_total(cumulative_return):
    return cumulative_return.iloc[-1] - 1 if not cumulative_return.empty else np.nan

RUTA_RETORNOS = "RetornoDiarioAcumulado"
RUTA_SECTORES = "datosgenerales/sectores.csv"

IGNORAR_ARCHIVOS = ["Comparativo.xlsx", "ratios_completos.xlsx"]

def obtener_archivos_validos(ruta):
    return [f for f in os.listdir(ruta)
            if f.endswith(".xlsx") and f not in IGNORAR_ARCHIVOS and not f.startswith("~$")]

def cargar_retornos_con_sector():
    archivos = obtener_archivos_validos(RUTA_RETORNOS)
    lista_metricas = []

    for archivo in tqdm(archivos, desc="Ejecución en progreso"):
        try:
            path_archivo = os.path.join(RUTA_RETORNOS, archivo)
            xls = pd.ExcelFile(path_archivo)

            # Leer segunda hoja si existe
            if len(xls.sheet_names) < 2:
                print(f"⚠️ {archivo} ignorado: no tiene segunda hoja.")
                continue

            df = pd.read_excel(xls, sheet_name=1)

            # Normalizar nombres de columnas
            df.columns = [col.strip().replace(" ", "_").capitalize() for col in df.columns]

            if 'Cumulative_return' not in df.columns:
                print(f"⚠️ {archivo} ignorado: no tiene columna 'Cumulative_return'.")
                continue

            df['Daily_return'] = df['Cumulative_return'].pct_change()
            retorno = calcular_retorno_total(df['Cumulative_return'])
            volatilidad = calcular_volatilidad(df['Daily_return'])
            sharpe = calcular_sharpe_ratio(df['Daily_return'])

            ticker = archivo.replace(".xlsx", "")
            lista_metricas.append({
                'ticker': ticker,
                'retorno': retorno,
                'volatilidad': volatilidad,
                'sharpe': sharpe
            })

        except Exception as e:
            print(f"❌ Error procesando {archivo}: {e}")

    if not lista_metricas:
        raise ValueError("❌ No se generaron métricas de ningún archivo. Verifica tus archivos en la carpeta 'RetornosDiarios'.")

    df_metricas = pd.DataFrame(lista_metricas)

    # Cargar sectores
    try:
        df_sectores = pd.read_csv(RUTA_SECTORES)
        df_sectores.columns = [col.strip().lower() for col in df_sectores.columns]
        df_metricas = pd.merge(df_metricas, df_sectores, on='ticker', how='left')

    except Exception as e:
        print(f"⚠️ No se pudo cargar el archivo de sectores: {e}")
        df_metricas['sector'] = np.nan

    return df_metricas

def ejecutar_selector_activos():
    print("📊 Iniciando selección de activos desde retornos acumulados...")
    df_retornos = cargar_retornos_con_sector()

    df_validos = df_retornos.dropna(subset=['retorno', 'volatilidad', 'sharpe'])

    print(f"🔢 Tickers disponibles: {len(df_retornos)}")
    print("🔍 Vista previa de activos con métricas válidas:")
    print(df_validos.head(20))
    print(f"✅ Total con métricas válidas: {len(df_validos)}")

    if len(df_validos) < 10:
        print("⚠️ No hay suficientes activos para generar carteras.")

    # Ejemplo: sugerencia de activos por alta volatilidad
    sugerencia = df_validos.sort_values(by="volatilidad", ascending=False).head(3)
    print("💡 Te interesaría ver la correlación entre algunos activos?")
    print("Te recomiendo estos tres pares por ser los más volátiles:")
    for i in range(len(sugerencia) - 1):
        print(f"  {sugerencia.iloc[i]['ticker']}-{sugerencia.iloc[i+1]['ticker']}")

def seleccionar_activos():
    ejecutar_selector_activos()
