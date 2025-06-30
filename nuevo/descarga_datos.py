import os
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from tiingo import TiingoClient
from datetime import datetime

# Claves API
ALPHAVANTAGE_API_KEY = "R31F84NCALC9ZG9D"
TIINGO_API_KEY = "79b1d025eeef21a98156177c746892c792cbb145"

# Columnas estandarizadas
COLUMNS = ["Date", "Close", "High", "Low", "Open", "Volume"]


def descargar_datos(tickers, start_date, end_date, proveedor):
    proveedor = proveedor.lower()
    carpeta_base = os.path.join("DatosCrudos", proveedor)
    os.makedirs(carpeta_base, exist_ok=True)

    print(f"\n📥 Proveedor: {proveedor.upper()}\n📁 Carpeta: {carpeta_base}\n🕒 Fechas: {start_date} a {end_date}\n🎯 Tickers: {', '.join(tickers)}")

    if proveedor == "yahoo":
        tickers_fallidos = []

        for ticker in tickers:
            try:
                print(f"\n🔽 Procesando {ticker} desde Yahoo...")
                df = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False, timeout=30)

                if df.empty:
                    print(f"⚠️ No se encontraron datos para {ticker}. Saltando.")
                    tickers_fallidos.append(ticker)
                    continue

                df.reset_index(inplace=True)
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                df.columns = pd.Index(df.columns).str.replace(r"[^\w]", "", regex=True)

                if 'AdjClose' in df.columns and 'Close' not in df.columns:
                    df.rename(columns={'AdjClose': 'Close'}, inplace=True)

                guardar_csv_limpio(df, ticker, carpeta_base)

            except Exception as e:
                print(f"❌ Error con {ticker} (Yahoo): {e}")
                tickers_fallidos.append(ticker)

        if tickers_fallidos:
            print("\n🚫 Tickers que fallaron en la descarga:")
            for t in tickers_fallidos:
                print(f" - {t}")
        else:
            print("\n✅ Todos los tickers se descargaron correctamente.")


    elif proveedor == "tiingo":
        config = { 'session': True, 'api_key': TIINGO_API_KEY }
        client = TiingoClient(config)
        for ticker in tickers:
            try:
                print(f"\n🔽 Procesando {ticker} desde Tiingo...")
                df = client.get_dataframe(ticker, frequency='daily', startDate=start_date, endDate=end_date)
                df.reset_index(inplace=True)
                df.rename(columns={
                    'date': 'Date',
                    'close': 'Close',
                    'high': 'High',
                    'low': 'Low',
                    'open': 'Open',
                    'volume': 'Volume'
                }, inplace=True)

                guardar_csv_limpio(df, ticker, carpeta_base)

            except Exception as e:
                print(f"❌ Error con {ticker} (Tiingo): {e}")

    else:
        print("❌ Proveedor no soportado.")


def guardar_csv_limpio(df, ticker, carpeta):
    cleaned_df = pd.DataFrame()
    for col in COLUMNS:
        cleaned_df[col] = df[col] if col in df.columns else pd.NA

    ruta = os.path.join(carpeta, f"{ticker}.csv")
    cleaned_df.to_csv(ruta, index=False)
    print(f"✅ {ticker} guardado en {ruta}")



def limpiar_datos_crudos(origen="DatosCrudos", destino="DatosLimpios"):
    os.makedirs(destino, exist_ok=True)
    print(f"\n🧹 Iniciando limpieza de datos en '{origen}'...")

    for root, _, files in os.walk(origen):
        for file in files:
            if not file.endswith(".csv"):
                continue

            path_csv = os.path.join(root, file)
            proveedor = os.path.basename(os.path.dirname(path_csv))
            ticker = os.path.splitext(file)[0]
            destino_prov = os.path.join(destino, proveedor)
            os.makedirs(destino_prov, exist_ok=True)

            try:
                df = pd.read_csv(path_csv, encoding="utf-8")
                df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

                posibles_fechas = ['date', 'fecha']
                col_fecha = next((col for col in posibles_fechas if col in df.columns), None)
                if not col_fecha:
                    raise ValueError("No se encontró columna de fecha.")

                df.rename(columns={col_fecha: 'Date'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
                df = df.sort_values('Date').dropna(subset=['Date'])

                columnas_estandar = {
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }
                df.rename(columns=columnas_estandar, inplace=True)

                df_limpio = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df_limpio.to_csv(os.path.join(destino_prov, f"{ticker}.csv"), index=False)
                print(f"✅ {file} limpiado correctamente.")

            except Exception as e:
                print(f"❌ Error limpiando {path_csv}: {e}")

    print("\n🏁 Limpieza completada.")
