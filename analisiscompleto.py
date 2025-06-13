# Script extendido: genera PDF con portada estilizada y Excel con datos, gráficos y colores modernos

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl.drawing.image import Image
from openpyxl import load_workbook
from matplotlib.backends.backend_pdf import PdfPages

# Paleta moderna
sns.set_palette("Set2")
plt.style.use('seaborn-v0_8-whitegrid')

# Configuración inicial
carpeta = './datospython1'
archivos = [f for f in os.listdir(carpeta) if f.endswith('.xlsx') or f.endswith('.xls')]
tickers = [f.replace('.xlsx', '').replace('.xls', '') for f in archivos]

# Diccionarios
resultados = {}
dataframes = {}
os.makedirs('./graficos_temp', exist_ok=True)
pdf = PdfPages('reporte_analisis.pdf')

# Portada del PDF con mejor estética
portada = plt.figure(figsize=(11, 8.5))
portada.clf()
portada.patch.set_facecolor('#f0f0f5')
titulo = "📊 Informe de Análisis Financiero"
subtitulo = f"Activos analizados: {', '.join(tickers)}"
descripcion = (
    "\nEste informe contiene:\n"
    "• Precio histórico con medias móviles (21d, 63d, 252d)\n"
    "• Retornos diarios\n"
    "• Histograma de retornos\n"
    "• Comentarios automáticos por activo\n\n"
    "📚 Guía rápida:\n"
    "✔️ Precio > SMA252 → tendencia alcista\n"
    "✔️ Cruces de medias móviles = posibles señales\n"
    "✔️ Retornos diarios = volatilidad\n"
    "✔️ Histograma = distribución de rendimientos\n\n"
    "⚠️ Nota: Este análisis es solo con fines educativos y no constituye asesoramiento financiero."
)
portada.text(0.5, 0.9, titulo, fontsize=24, ha='center', va='top', weight='bold', color='#333366')
portada.text(0.5, 0.82, subtitulo, fontsize=14, ha='center', va='top', style='italic')
portada.text(0.05, 0.7, descripcion, fontsize=11, ha='left', va='top')
pdf.savefig(portada)
plt.close(portada)

# Análisis por activo
for archivo in archivos:
    nombre = archivo.replace('.xlsx', '').replace('.xls', '')
    ruta = os.path.join(carpeta, archivo)
    df = pd.read_excel(ruta)

    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df.set_index('Fecha', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        continue

    if 'Precio_Cierre' in df.columns:
        df.rename(columns={'Precio_Cierre': 'Precio'}, inplace=True)
    elif 'Adj Close' in df.columns:
        df.rename(columns={'Adj Close': 'Precio'}, inplace=True)
    elif 'Close' in df.columns:
        df.rename(columns={'Close': 'Precio'}, inplace=True)
    else:
        continue

    df['Retorno'] = df['Precio'].pct_change()
    df['RetornoSemanal'] = df['Retorno'].rolling(window=5).mean()
    df['SMA_21'] = df['Precio'].rolling(window=21).mean()
    df['SMA_63'] = df['Precio'].rolling(window=63).mean()
    df['SMA_252'] = df['Precio'].rolling(window=252).mean()

    retorno_diario = df['Retorno'].mean()
    retorno_semanal = df['Retorno'].rolling(5).mean().mean()
    retorno_mensual = df['Retorno'].rolling(21).mean().mean()
    retorno_anual = (1 + retorno_diario) ** 252 - 1

    vol_diaria = df['Retorno'].std()
    vol_semanal = df['Retorno'].rolling(5).std().mean()
    vol_mensual = df['Retorno'].rolling(21).std().mean()

    stats = {
        'retorno_diario_prom': retorno_diario,
        'retorno_semanal_prom': retorno_semanal,
        'retorno_mensual_prom': retorno_mensual,
        'retorno_anual': retorno_anual,
        'volatilidad_diaria': vol_diaria,
        'volatilidad_semanal': vol_semanal,
        'volatilidad_mensual': vol_mensual,
        'precio_actual': df['Precio'].iloc[-1],
        'sma_21': df['SMA_21'].iloc[-1],
        'sma_63': df['SMA_63'].iloc[-1],
        'sma_252': df['SMA_252'].iloc[-1]
    }

    resultados[nombre] = stats
    dataframes[nombre] = df

    # Gráfico de precios
    fig, ax = plt.subplots(figsize=(10, 5))
    df['Precio'].plot(ax=ax, label='Precio', linewidth=1.2)
    df['SMA_21'].plot(ax=ax, label='SMA 21d')
    df['SMA_63'].plot(ax=ax, label='SMA 63d')
    df['SMA_252'].plot(ax=ax, label='SMA 252d')
    ax.set_title(f'{nombre} - Precio con Medias Móviles')
    ax.set_ylabel("Precio de cierre (USD)")
    ax.set_xlabel("Fecha")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'./graficos_temp/{nombre}_precio.png')
    pdf.savefig(fig)
    plt.close(fig)

    # Comentario
    comentario = (
        f"📌 {nombre}\n\n"
        f"Retorno anual estimado: {retorno_anual*100:.2f}%\n"
        f"Volatilidad mensual: {vol_mensual*100:.2f}%\n"
        f"Precio actual: {df['Precio'].iloc[-1]:.2f} USD\n"
        f"SMA 252d: {df['SMA_252'].iloc[-1]:.2f}\n"
        f"Situación: {'por encima' if df['Precio'].iloc[-1] > df['SMA_252'].iloc[-1] else 'por debajo'} de la tendencia anual."
    )
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    ax.text(0, 1, comentario, fontsize=10, va='top', ha='left', wrap=True, family='monospace')
    pdf.savefig(fig)
    plt.close(fig)

    # Retornos diarios
    fig, ax = plt.subplots(figsize=(10, 4))
    df['Retorno'].plot(ax=ax, color='orange')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title(f'{nombre} - Retornos Diarios')
    ax.set_ylabel("Retorno")
    ax.set_xlabel("Fecha")
    fig.tight_layout()
    fig.savefig(f'./graficos_temp/{nombre}_retorno.png')
    pdf.savefig(fig)
    plt.close(fig)

    # Histograma
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df['Retorno'].dropna(), bins=50, kde=True, ax=ax, color='teal')
    mu = df['Retorno'].mean()
    sigma = df['Retorno'].std()
    ax.axvline(mu, color='red', linestyle='--', label=f'Media: {mu:.4f}')
    ax.axvline(mu + sigma, color='green', linestyle='--', label=f'+1σ: {mu + sigma:.4f}')
    ax.axvline(mu - sigma, color='green', linestyle='--', label=f'-1σ: {mu - sigma:.4f}')
    ax.set_title(f'{nombre} - Histograma de Retornos')
    ax.set_xlabel("Retorno diario")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'./graficos_temp/{nombre}_histograma.png')
    pdf.savefig(fig)
    plt.close(fig)

pdf.close()

# Generar Excel
resumen = pd.DataFrame(resultados).T
with pd.ExcelWriter('analisis_activos.xlsx', engine='openpyxl') as writer:
    resumen.to_excel(writer, sheet_name='Resumen')

    # Gráfico comparativo
    plt.figure(figsize=(8, 5))
    resumen['retorno_anual'].sort_values().plot(kind='barh', color='slateblue')
    plt.title('Comparación de Retornos Anuales')
    plt.xlabel('Retorno anual estimado')
    plt.tight_layout()
    plt.savefig('./graficos_temp/retornos_anuales.png')
    plt.close()

    writer.book = load_workbook('analisis_activos.xlsx')
    ws = writer.book['Resumen']
    img = Image('./graficos_temp/retornos_anuales.png')
    img.anchor = 'J2'
    ws.add_image(img)

    for nombre, df in dataframes.items():
        df.to_excel(writer, sheet_name=nombre)
        ws = writer.book[nombre]
        fila = len(df) + 5
        for i, tipo in enumerate(['precio', 'retorno', 'histograma']):
            path = f'./graficos_temp/{nombre}_{tipo}.png'
            if os.path.exists(path):
                img = Image(path)
                img.anchor = f'A{fila + i * 20}'
                ws.add_image(img)

    writer.book.save('analisis_activos.xlsx')

import shutil
shutil.rmtree('./graficos_temp')

print("✅ Excel + PDF generados con portada estilizada, gráficos modernos y datos completos.")
