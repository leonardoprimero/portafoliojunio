import os
import pandas as pd
import yfinance as yf
from aaaconfig_usuario import tickers as ALL_TICKERS

CARPETA_SALIDA = './datosgenerales'
ARCHIVO_CSV = 'sectores.csv'
ARCHIVO_RISKFREE = 'sectores_riskfree.csv'

TICKERS_RISKFREE = {
    "^IRX": "Treasury 13 Week Bill Yield (3 months)",
    "^FVX": "Treasury 5 Year Note Yield (5 years)",
    "^TNX": "Treasury 10 Year Note Yield (10 years)",
    "^TYX": "Treasury 30 Year Bond Yield (30 years)"
}
TICKERS_EXTRA = [t for t in ALL_TICKERS if t not in TICKERS_RISKFREE]

def crear_carpeta(carpeta):
    os.makedirs(carpeta, exist_ok=True)

def obtener_lista_sp500():
    print("📥 Descargando lista del S&P 500...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(url, header=0)[0]
        df = df[['Symbol', 'Security', 'GICS Sector']]
        df.columns = ['Ticker', 'Nombre', 'Sector']
        return df
    except Exception as e:
        print(f"❌ Error descargando S&P 500: {e}")
        return pd.DataFrame(columns=['Ticker', 'Nombre', 'Sector'])

def buscar_ticker_extra(ticker):
    print(f"🔎 Buscando {ticker} en Yahoo Finance...")
    try:
        info = yf.Ticker(ticker).info
        nombre = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector") or "Sin sector"
        print(f"  ➕ Agregado: {nombre} | Sector: {sector}")
        return {"Ticker": ticker, "Nombre": nombre, "Sector": sector}
    except Exception as e:
        print(f"  ⚠️ No se pudo obtener info de {ticker}: {e}")
        return None

def guardar_csv(df, carpeta, nombre_archivo):
    ruta = os.path.join(carpeta, nombre_archivo)
    df.to_csv(ruta, index=False, encoding='utf-8-sig')
    print(f"\n✅ CSV generado: {ruta}")

def main():
    crear_carpeta(CARPETA_SALIDA)

    df_tickers = obtener_lista_sp500()
    riskfree_rows = []

    for ticker in ALL_TICKERS:
        if ticker in TICKERS_RISKFREE:
            # Es una tasa libre de riesgo: nombre y sector especial
            riskfree_rows.append({
                "Ticker": ticker,
                "Nombre": TICKERS_RISKFREE[ticker],
                "Sector": "Risk Free Rate"
            })
        elif ticker not in df_tickers['Ticker'].values:
            nuevo = buscar_ticker_extra(ticker)
            if nuevo:
                df_tickers = pd.concat([df_tickers, pd.DataFrame([nuevo])], ignore_index=True)

    df_tickers = df_tickers[~df_tickers["Ticker"].isin(TICKERS_RISKFREE.keys())]
    df_tickers.sort_values(by='Ticker', inplace=True)
    guardar_csv(df_tickers, CARPETA_SALIDA, ARCHIVO_CSV)

    if riskfree_rows:
        df_riskfree = pd.DataFrame(riskfree_rows)
        guardar_csv(df_riskfree, CARPETA_SALIDA, ARCHIVO_RISKFREE)

    print(df_tickers.tail(10))
    if riskfree_rows:
        print("Tickers Risk Free:")
        print(df_riskfree)

if __name__ == "__main__":
    main()
