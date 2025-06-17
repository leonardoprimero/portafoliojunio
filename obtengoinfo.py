import yfinance as yf
import wikipedia
import requests
import pandas as pd
import os

CARPETA_SALIDA = './FundamentalActivos'
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# Carpeta con archivos de tickers
CARPETA_TICKERS = './datospython1'
tickers = [os.path.splitext(f)[0] for f in os.listdir(CARPETA_TICKERS)
           if f.lower().endswith(('.csv', '.xlsx')) and os.path.splitext(f)[0].isupper() and os.path.splitext(f)[0] != "SECTORES"]


informes = []
errores = []

for t in tickers:
    try:
        print(f'⏳ Procesando {t}...')
        ticker = yf.Ticker(t)
        info = ticker.info

        nombre = info.get('longName', t)
        sector = info.get('sector', 'Desconocido')
        industria = info.get('industry', 'Desconocida')
        pais = info.get('country', '')
        ciudad = info.get('city', '')

        # Buscar en Wikipedia
        wikipedia.set_lang("en")
        try:
            resumen = wikipedia.summary(nombre, sentences=5)
        except:
            try:
                # Reintento sin punto, con fallback
                resumen = wikipedia.summary(nombre.replace(".", ""), sentences=5)
            except:
                resumen = "Descripción no disponible."

        # Logo desde Clearbit (experimental)
        dominio = info.get("website", "").replace("https://", "").replace("www.", "").split("/")[0]
        try:
            logo_url = f"https://logo.clearbit.com/{dominio}" if dominio else ""
            # Testeo rápido de validez
            _ = requests.get(logo_url, timeout=3)
        except:
            logo_url = ""

        informes.append({
            'Ticker': t,
            'Nombre': nombre,
            'Sector': sector,
            'Industria': industria,
            'País': pais,
            'Ciudad': ciudad,
            'Descripción': resumen,
            'Logo URL': logo_url,
            'Market Cap': info.get('marketCap'),
            'P/E Ratio': info.get('trailingPE'),
            'Dividend Yield': info.get('dividendYield'),
            'Ingresos': info.get('totalRevenue'),
            'Ingresos Netos': info.get('netIncomeToCommon'),
            'Beta': info.get('beta'),
        })
    except Exception as e:
        print(f'⚠️ Error con {t}: {e}')
        errores.append({'Ticker': t, 'Error': str(e)})

# Guardar como CSV para usar luego
pd.DataFrame(informes).to_csv(os.path.join(CARPETA_SALIDA, "informe_empresas.csv"), index=False)
print("✅ Archivo CSV guardado: informe_empresas.csv")




if errores:
    pd.DataFrame(errores).to_csv(os.path.join(CARPETA_SALIDA, "errores_wikipedia_clearbit.csv"), index=False)
    print("⚠️ Errores guardados en errores_wikipedia_clearbit.csv")

print(f"🟢 Informes exitosos: {len(informes)}")
print(f"🔴 Errores: {len(errores)}")
