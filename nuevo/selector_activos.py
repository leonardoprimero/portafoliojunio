import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from aaaconfig_usuario import tasa_libre_riesgo_ticker


# --- CONFIGURACIÓN ---
RUTA_RETORNOS = "RetornoDiarioAcumulado"
RUTA_SECTORES = "datosgenerales/sectores.csv"
CARPETA_TASAS = "TasasLibresRiesgo"
IGNORAR_ARCHIVOS = ["Comparativo.xlsx", "ratios_completos.xlsx"]

# --- FUNCIONES DE CÁLCULO ---

def cargar_tasa_libre_riesgo_diaria(ticker, carpeta):
    path = os.path.join(carpeta, f"{ticker}_analisis.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de tasa libre de riesgo en: {path}")
    
    df = pd.read_excel(path, sheet_name="Serie", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    tasa_diaria = (df["Yield"] / 100) / 252
    tasa_diaria.name = "R_f"
    return tasa_diaria

def calcular_sharpe_ratio(retornos_diarios, tasa_libre_diaria=None):
    if retornos_diarios.empty:
        return np.nan

    if tasa_libre_diaria is not None and not tasa_libre_diaria.empty:
        df_combinado = pd.concat([retornos_diarios, tasa_libre_diaria], axis=1).fillna(method='ffill')
        tasa_alineada = df_combinado.loc[retornos_diarios.index, 'R_f'].fillna(method='bfill')
        if tasa_alineada.isnull().all():
            excesos = retornos_diarios
        else:
            excesos = retornos_diarios - tasa_alineada
    else:
        excesos = retornos_diarios

    media_exceso = excesos.mean(skipna=True)
    desviacion_exceso = excesos.std(skipna=True)

    if desviacion_exceso == 0 or pd.isna(desviacion_exceso) or desviacion_exceso < 1e-9: # Evitar división por cero
        return np.nan
        
    return (media_exceso / desviacion_exceso) * np.sqrt(252)

def calcular_volatilidad(retornos_diarios):
    if retornos_diarios.empty:
        return np.nan
    return retornos_diarios.std() * np.sqrt(252)

def calcular_retorno_total(cumulative_return):
    if cumulative_return.empty or len(cumulative_return.dropna()) < 2:
        return np.nan
    # Usar el último valor válido
    last_valid_value = cumulative_return.dropna().iloc[-1]
    return last_valid_value - 1

# --- FUNCIONES PRINCIPALES DEL SCRIPT ---

def obtener_archivos_validos(ruta):
    return [f for f in os.listdir(ruta)
            if f.endswith(".xlsx") and f not in IGNORAR_ARCHIVOS and not f.startswith("~$")]

def cargar_metricas_de_activos():
    archivos = obtener_archivos_validos(RUTA_RETORNOS)
    lista_metricas = []

    tasa_libre_diaria = None
    if tasa_libre_riesgo_ticker:
        try:
            tasa_libre_diaria = cargar_tasa_libre_riesgo_diaria(tasa_libre_riesgo_ticker, CARPETA_TASAS)
            print(f"🔵 Usando tasa libre de riesgo '{tasa_libre_riesgo_ticker}' para el cálculo del Sharpe Ratio.")
        except Exception as e:
            print(f"⚠️ No se pudo cargar la tasa libre de riesgo. El Sharpe Ratio se calculará sin ajuste. Error: {e}")

    for archivo in tqdm(archivos, desc="Calculando métricas de activos"):
        ticker = archivo.replace(".xlsx", "")
        try:
            path_archivo = os.path.join(RUTA_RETORNOS, archivo)
            # Leer la segunda hoja (índice 1) que contiene la serie de retornos
            df = pd.read_excel(path_archivo, sheet_name=1, index_col=0, parse_dates=True)
            
            df.columns = [col.strip().replace(" ", "_").capitalize() for col in df.columns]

            if 'Cumulative_return' not in df.columns:
                print(f"⚠️ Archivo '{archivo}' ignorado: no tiene columna 'Cumulative_return'.")
                continue
            
            # --- PASO CRÍTICO: ASEGURAR ORDEN CRONOLÓGICO ---
            df = df.sort_index()

            # Limpiar la columna de retornos acumulados de valores no numéricos y nulos
            df['Cumulative_return'] = pd.to_numeric(df['Cumulative_return'], errors='coerce')
            df = df.dropna(subset=['Cumulative_return'])

            if len(df) < 2:
                print(f"⚠️ Archivo '{archivo}' ignorado: no tiene suficientes datos válidos para calcular retornos.")
                continue

            # Calcular métricas
            daily_returns = df['Cumulative_return'].pct_change().dropna()
            
            retorno = calcular_retorno_total(df['Cumulative_return'])
            volatilidad = calcular_volatilidad(daily_returns)
            sharpe = calcular_sharpe_ratio(daily_returns, tasa_libre_diaria)
            if tasa_libre_diaria is not None and not tasa_libre_diaria.empty:
                print(f"🔹 {ticker}: Sharpe ajustado usando {tasa_libre_riesgo_ticker}")
            else:
                print(f"◽ {ticker}: Sharpe clásico (sin ajuste de tasa)")
            ticker = archivo.replace(".xlsx", "")
            lista_metricas.append({
                'ticker': ticker,
                'retorno': retorno,
                'volatilidad': volatilidad,
                'sharpe': sharpe
            })

        except Exception as e:
            print(f"❌ Error procesando el archivo '{archivo}': {e}")

    if not lista_metricas:
        raise ValueError("❌ No se pudieron generar métricas para ningún archivo.")

    df_metricas = pd.DataFrame(lista_metricas)

    try:
        df_sectores = pd.read_csv(RUTA_SECTORES)
        df_sectores.columns = [col.strip().lower() for col in df_sectores.columns]
        df_metricas = pd.merge(df_metricas, df_sectores, on='ticker', how='left')
    except Exception as e:
        print(f"⚠️ No se pudo cargar el archivo de sectores: {e}")
        df_metricas['sector'] = 'Desconocido'

    return df_metricas

def generar_carteras_y_guardar_excel(df):
    output_path = "carteras_recomendadas.xlsx"
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

    df_por_sector = df.dropna(subset=['sector', 'sharpe']).copy()
    if not df_por_sector.empty:
        carteras_sector = df_por_sector.groupby('sector').apply(lambda x: x.nlargest(3, 'sharpe')).reset_index(drop=True)
        if not carteras_sector.empty:
            carteras_sector.to_excel(writer, sheet_name="Mejores_Por_Sector", index=False)

    perfiles = {
        "Conservador (Baja Volatilidad)": lambda x: x.nsmallest(12, 'volatilidad'),
        "Moderado (Balanceado)": lambda x: x.nlargest(10, 'sharpe'),
        "Agresivo (Alto Retorno)": lambda x: x.nlargest(9, 'retorno'),
        "Mixta (Diversificada)": lambda x: pd.concat([
            x.nlargest(4, 'sharpe'), x.nsmallest(4, 'volatilidad'), x.nlargest(4, 'retorno')
        ]).drop_duplicates().head(12)
    }

    for nombre, filtro in perfiles.items():
        cartera = filtro(df.dropna(subset=['retorno', 'volatilidad', 'sharpe']))
        if not cartera.empty:
            cartera['peso_%'] = round(100 / len(cartera), 2)
            cartera.to_excel(writer, sheet_name=nombre, index=False)

    writer.close()
    print(f"✅ Archivo '{output_path}' generado exitosamente.")

def ejecutar_selector_activos():
    print("📊 Iniciando el selector de activos...")
    try:
        df_metricas = cargar_metricas_de_activos()
        df_validos = df_metricas.dropna(subset=['retorno', 'volatilidad', 'sharpe']).copy()

        print(f"\n🔢 Se procesaron {len(df_metricas)} tickers.")
        print(f"✅ Se obtuvieron {len(df_validos)} activos con métricas válidas.")
        
        if len(df_validos) < 10:
            print("\n⚠️ No hay suficientes activos con métricas válidas para generar carteras diversificadas.")
            return

        print("\n🔍 Vista previa de los activos con mejores métricas (ordenados por Sharpe):")
        print(df_validos.sort_values(by="sharpe", ascending=False).head(10))

        generar_carteras_y_guardar_excel(df_validos)

    except Exception as e:
        print(f"\n❌ Ocurrió un error crítico durante la ejecución: {e}")

if __name__ == "__main__":
    ejecutar_selector_activos()
