# === PROCURAR TENER XlsxWriter   pip install XlsxWriter ===

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# === CONFIGURACIÓN ===
'''sectores = {
    'Tecnología': ['AAPL', 'MSFT', 'NVDA', 'GOOGL'],
    'Salud': ['JNJ', 'PFE', 'MRK', 'UNH'],
    'Finanzas': ['JPM', 'BAC', 'WFC', 'C'],
    'Energía': ['XOM', 'CVX', 'COP', 'SLB'],
    'Benchmark':['SPY']
}'''

sectores = {
    'Tecnología': ['AAPL', 'MSFT', 'NVDA', 'GOOGL'],
    'Salud': ['JNJ', 'PFE'],
    'Finanzas': ['JPM', 'WFC', 'C', 'GS'],
    'Energía': ['XOM', 'CVX', 'COP'],
    'Consumo discrecional': ['AMZN', 'TSLA', 'MCD'],
    'Consumo básico': ['PG', 'KO', 'WMT'],
    'Industriales': ['HON', 'UPS', 'CAT'],
    'Utilities': [ 'SO', 'EXC'],
    'Materiales': [ 'APD', 'ECL'],
    'Comunicaciones': ['T', 'DIS'],
    'Real Estate': ['PLD', 'AMT'],
    'Benchmark': ['SPY']
    }

fecha_inicio = '2015-01-01'
fecha_fin = '2025-06-16'

# Ruta donde se guardarán los archivos
carpeta_destino = os.path.join(os.path.dirname(__file__), 'datospython1')

# Crear carpeta si no existe
os.makedirs(carpeta_destino, exist_ok=True)

# === DESCARGA Y GUARDADO ===
for sector, activos in sectores.items():
    for ticker in activos:
        print(f"Descargando {ticker} ({sector})...")
        try:
            data = yf.download(ticker, start=fecha_inicio, end=fecha_fin)

            if data.empty:
                print(f"❌ Sin datos para {ticker}.")
                continue

            precios = data[['Close']].copy()
            precios.reset_index(inplace=True)
            precios.columns = ['Fecha', 'Precio_Cierre']
            precios['Fecha'] = precios['Fecha'].dt.date

            retornos = precios.copy()
            retornos['Retorno_Diario'] = retornos['Precio_Cierre'].pct_change()
            retornos = retornos.dropna(subset=['Retorno_Diario'])

            archivo_excel = os.path.join(carpeta_destino, f"{ticker}.xlsx")

            with pd.ExcelWriter(archivo_excel, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as writer:
                precios.to_excel(writer, sheet_name='Precios_Cierre', index=False)
                retornos[['Fecha', 'Retorno_Diario']].to_excel(writer, sheet_name='Retornos_Diarios', index=False)

                # Formato bonito
                workbook = writer.book
                formato_moneda = workbook.add_format({'num_format': '$#,##0.00'})
                formato_porcentaje = workbook.add_format({'num_format': '0.00%'})
                formato_fecha = workbook.add_format({'num_format': 'yyyy-mm-dd'})

                hoja_precios = writer.sheets['Precios_Cierre']
                hoja_retornos = writer.sheets['Retornos_Diarios']

                hoja_precios.set_column('A:A', 12, formato_fecha)
                hoja_precios.set_column('B:B', 15, formato_moneda)

                hoja_retornos.set_column('A:A', 12, formato_fecha)
                hoja_retornos.set_column('B:B', 18, formato_porcentaje)

            print(f"✅ Guardado en: {archivo_excel}")

        except Exception as e:
            print(f"⚠️ Error con {ticker}: {e}")
