import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# ---------------------- CONFIGURACIÓN ----------------------
TICKERS = ['ETHA','IBIT','GLD','FXI','SMH','SPXL','XLU','CIBR','URA','IEUR','IBB','IVW','IVE','XLC','XLB','XLI','XLK','XLRE','XLF','XLP','XLV','XLY','VEA']  # <- Cambiá por los que quieras 
START_DATE = '2015-01-01'
END_DATE = '2025-06-17'
CARPETA_SALIDA = './datospython1'

# -----------------------------------------------------------
os.makedirs(CARPETA_SALIDA, exist_ok=True)
print(f"📥 Descargando datos desde Yahoo Finance...\n")

for ticker in TICKERS:
    try:
        print(f"🔽 Descargando {ticker}...")
        data = yf.download(ticker, start=START_DATE, end=END_DATE)
        
        if data.empty:
            print(f"⚠️  No se encontraron datos para {ticker}.")
            continue

        # Guardar en formato CSV crudo
        ruta_salida = os.path.join(CARPETA_SALIDA, f"{ticker.upper()}.csv")
        data.to_csv(ruta_salida)
        print(f"✅ Guardado: {ruta_salida}")

    except Exception as e:
        print(f"❌ Error descargando {ticker}: {e}")

print("\n✅ Descarga finalizada.")
