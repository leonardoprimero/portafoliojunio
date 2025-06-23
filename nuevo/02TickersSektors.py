import os
import pandas as pd
import yfinance as yf

CARPETA_SALIDA = './datosgenerales'
ARCHIVO_CSV = 'sectores.csv'
TICKERS_EXTRA = ["MELI"]  # Podés sumar más

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

    for ticker in TICKERS_EXTRA:
        if ticker not in df_tickers['Ticker'].values:
            nuevo = buscar_ticker_extra(ticker)
            if nuevo:
                df_tickers = pd.concat([df_tickers, pd.DataFrame([nuevo])], ignore_index=True)

    df_tickers.sort_values(by='Ticker', inplace=True)  # opcional
    guardar_csv(df_tickers, CARPETA_SALIDA, ARCHIVO_CSV)

    print(df_tickers.tail(10))

if __name__ == "__main__":
    main()
