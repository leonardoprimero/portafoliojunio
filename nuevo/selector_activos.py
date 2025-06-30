import pandas as pd
import os
from aaaconfig_usuario import tickers_portafolio, nivel_volatilidad_cliente, sectores_cliente, max_activos_por_sector

def cargar_retornos_con_sector(
    carpeta_retornos="RetornoDiarioAcumulado",
    path_sectores="datosgenerales/sectores.csv"
):
    datos = []
    for archivo in os.listdir(carpeta_retornos):
        if archivo.endswith(".xlsx"):
            path_archivo = os.path.join(carpeta_retornos, archivo)
            try:
                df = pd.read_excel(path_archivo)
                if not df.empty:
                    ultima_fila = df.iloc[-1]
                    ticker = archivo.replace(".xlsx", "")
                    datos.append({
                        "ticker": ticker,
                        "retorno": ultima_fila.get("Cumulative_Return_lineal"),
                        "volatilidad": ultima_fila.get("volatilidad"),
                        "sharpe": ultima_fila.get("sharpe")
                    })
            except Exception as e:
                print(f"❌ Error leyendo {archivo}: {e}")

    df_metricas = pd.DataFrame(datos)

    # Asegurar consistencia en los nombres de columnas
    df_sectores = pd.read_csv(path_sectores)
    df_sectores.columns = df_sectores.columns.str.strip().str.lower()
    if 'ticker' not in df_sectores.columns:
        if 'Ticker' in df_sectores.columns:
            df_sectores.rename(columns={"Ticker": "ticker"}, inplace=True)
        else:
            raise KeyError("La columna 'ticker' no existe en el archivo de sectores.")

    df = df_metricas.merge(df_sectores, on="ticker", how="left")
    return df

def seleccionar_activos(df_metricas, max_vol, sectores, max_por_sector):
    seleccion = []
    sectores_unicos = df_metricas['sector'].dropna().unique()
    for sector in sectores_unicos:
        if sectores and sector not in sectores:
            continue
        df_sector = df_metricas[df_metricas['sector'] == sector]
        df_sector = df_sector[df_sector['volatilidad'] <= max_vol]
        df_sector = df_sector.sort_values(by='sharpe', ascending=False)
        seleccion.extend(df_sector.head(max_por_sector)['ticker'].tolist())
    return seleccion

def generar_recomendaciones(df_metricas, n_recomendaciones=5, min_activos=8, max_activos=12, min_diferencia=6):
    from itertools import combinations
    import random

    base = seleccionar_activos(df_metricas, nivel_volatilidad_cliente, sectores_cliente, max_activos_por_sector)
    base = list(set(base))
    if len(base) < min_activos:
        print("⚠️ No hay suficientes activos para generar carteras.")
        return []

    random.shuffle(base)
    recomendaciones = []
    intentos = 0
    max_intentos = 1000

    while len(recomendaciones) < n_recomendaciones and intentos < max_intentos:
        intentos += 1
        cartera = sorted(random.sample(base, k=random.randint(min_activos, min(len(base), max_activos))))
        es_distinta = all(len(set(cartera).symmetric_difference(set(c))) >= min_diferencia for c in recomendaciones)
        if es_distinta:
            recomendaciones.append(cartera)

    return recomendaciones

def ejecutar_selector_activos():
    print("📊 Iniciando selección de activos desde retornos acumulados...")
    df_retornos = cargar_retornos_con_sector()
    print(f"🔢 Tickers disponibles: {df_retornos['ticker'].nunique()}")

    print("🔍 Vista previa de activos con métricas válidas:")
    df_metricas_válidas = df_retornos.dropna(subset=['retorno', 'volatilidad', 'sharpe'])
    print(df_metricas_válidas[['ticker', 'retorno', 'volatilidad', 'sharpe', 'sector']])
    print(f"✅ Total con métricas válidas: {len(df_metricas_válidas)}")

    recomendaciones = generar_recomendaciones(df_metricas_válidas)

    for i, cartera in enumerate(recomendaciones):
        df_cartera = df_metricas_válidas[df_metricas_válidas['ticker'].isin(cartera)]
        path = f"activos_seleccionados_opcion_{i+1}.csv"
        df_cartera.to_csv(path, index=False)
        print(f"✅ Cartera sugerida {i+1} guardada en {path}")
        print(df_cartera[['ticker', 'retorno', 'volatilidad', 'sharpe', 'sector']])
        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    ejecutar_selector_activos()
