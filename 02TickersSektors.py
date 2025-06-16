import os
import pandas as pd
import yfinance as yf

# Crear carpeta si no existe
carpeta = './datospython1'
os.makedirs(carpeta, exist_ok=True)

# 1. Descargar la lista base del S&P 500
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
df = pd.read_html(url, header=0)[0]
df_tickers = df[['Symbol', 'Security', 'GICS Sector']]
df_tickers.columns = ['Ticker', 'Nombre', 'Sector']

# 2. Tickers extra que quieras sumar (ejemplo: MELI)
tickers_extra = ["MELI"]  # Podés poner más, separados por coma

# 3. Si no está en la tabla, lo busca en Yahoo Finance y lo agrega
for t in tickers_extra:
    if t not in df_tickers['Ticker'].values:
        print(f"⏳ Buscando {t} en Yahoo Finance...")
        try:
            ticker_yf = yf.Ticker(t)
            info = ticker_yf.info

            nombre = info.get("longName") or info.get("shortName") or t
            sector = info.get("sector") or "Sin sector"
            print(f"  ➜ Encontrado: {nombre} | Sector: {sector}")

            df_tickers = pd.concat(
                [df_tickers, pd.DataFrame([{"Ticker": t, "Nombre": nombre, "Sector": sector}])],
                ignore_index=True)
        except Exception as e:
            print(f"  ⚠️ No se pudo obtener info para {t}: {e}")

# 4. Guardar en la carpeta correcta y con el nombre requerido
csv_path = os.path.join(carpeta, 'sectores.csv')
df_tickers.to_csv(csv_path, index=False)
print(f"\n✅ CSV generado en: {csv_path}")
print(df_tickers.tail(10))
