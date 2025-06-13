#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

CARPETA_DATOS = './datospython1'
CARPETA_SALIDA = './graficos_temp'
BENCHMARK = 'SPY'
ROLLING_WINDOW = 60
MIN_DIAS = 200  # Descarta archivos que tengan pocos días de datos

os.makedirs(CARPETA_SALIDA, exist_ok=True)

def leer_archivo(ruta):
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    try:
        if ruta.endswith('.csv'):
            df = pd.read_csv(ruta, index_col=0, parse_dates=True)
        elif ruta.endswith('.xlsx') or ruta.endswith('.xls'):
            df = pd.read_excel(ruta, index_col=0, parse_dates=True)
        else:
            print(f"⚠️ Formato no reconocido: {ruta}")
            return None, None
        for c in ['adj close', 'close', 'precio_cierre', 'Adj Close', 'Close', 'Precio_Cierre']:
            for col in df.columns:
                if col.strip().lower() == c.lower():
                    df_filtrado = df[[col]].rename(columns={col: nombre})
                    return df_filtrado, col
        print(f"⚠️ No se encontró columna válida en {ruta}")
        return None, None
    except Exception as e:
        print(f"⚠️ Error leyendo {ruta}: {e}")
        return None, None

# --- LECTURA Y FILTRO DE ARCHIVOS ---
dataframes = []
plantilla_cols, plantilla_tipo = None, None
fechas_x_activo = {}

archivos = [f for f in os.listdir(CARPETA_DATOS) if not f.startswith('.')]

for archivo in archivos:
    if archivo.split('.')[0].upper() == BENCHMARK.upper():
        continue
    ruta = os.path.join(CARPETA_DATOS, archivo)
    df, col = leer_archivo(ruta)
    if df is not None:
        if len(df) < MIN_DIAS:
            print(f"⚠️ {archivo} descartado: solo {len(df)} días de datos.")
            continue
        dataframes.append(df)
        fechas_x_activo[df.columns[0]] = (df.index.min(), df.index.max(), len(df))
        if plantilla_cols is None and col is not None:
            plantilla_cols = col
            plantilla_tipo = os.path.splitext(archivo)[1]

if len(dataframes) == 0:
    print("❌ No se encontraron archivos válidos.")
    exit()

# --- MERGE DE ACTIVOS: SOLO LOS DÍAS COMUNES ---
df_merged = dataframes[0]
for df in dataframes[1:]:
    df_merged = df_merged.join(df, how='inner')

# --- BENCHMARK ---
archivo_bench = os.path.join(CARPETA_DATOS, f"{BENCHMARK}{plantilla_tipo or '.csv'}")
df_bench = None

if os.path.exists(archivo_bench):
    print(f"✔️ Benchmark {BENCHMARK} encontrado en carpeta. Verificando formato...")
    df_bench, col_bench = leer_archivo(archivo_bench)
    if df_bench is None:
        try:
            if archivo_bench.endswith('.xlsx') or archivo_bench.endswith('.xls'):
                tmp = pd.read_excel(archivo_bench, index_col=0, parse_dates=True)
            else:
                tmp = pd.read_csv(archivo_bench, index_col=0, parse_dates=True)
            col_bench = None
            for c in ['Adj Close', 'Close', BENCHMARK, 'Precio_Cierre']:
                if c in tmp.columns:
                    col_bench = c
                    break
            if col_bench:
                df_bench = tmp[[col_bench]].rename(columns={col_bench: BENCHMARK})
            else:
                raise Exception("No se encontró columna de cierre válida en el benchmark")
        except Exception as e:
            print(f"❌ Imposible adaptar benchmark: {e}")
            exit()
else:
    print(f"🔽 Descargando {BENCHMARK} ({df_merged.index.min().date()} a {df_merged.index.max().date()}) y guardando como los otros archivos...")
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Instalá yfinance: pip install yfinance")
    data_yf = yf.download(BENCHMARK, start=df_merged.index.min(), end=df_merged.index.max())
    col_yf = None
    for c in [plantilla_cols, 'Adj Close', 'Close']:
        if c and c in data_yf.columns:
            col_yf = c
            break
    if not col_yf:
        raise Exception(f"No se encontró columna de cierre en datos de Yahoo para {BENCHMARK}")
    df_bench = data_yf[[col_yf]].rename(columns={col_yf: BENCHMARK})
    if plantilla_tipo in ['.xlsx', '.xls']:
        df_bench.to_excel(archivo_bench)
    else:
        df_bench.to_csv(archivo_bench)
    print(f"✔️ Benchmark guardado como {archivo_bench}")

# --- MERGE BENCHMARK SOLO EN FECHAS QUE EXISTEN EN EL MERGE ---
df_bench = df_bench[df_bench.index.isin(df_merged.index)]
fechas_x_activo[BENCHMARK] = (df_bench.index.min(), df_bench.index.max(), len(df_bench))
df_merged = df_merged.join(df_bench, how='inner')

if df_merged.shape[0] < 20:
    print("❌ No hay suficientes fechas comunes. Debug rápido:")
    for activo, (fi, ff, N) in fechas_x_activo.items():
        print(f"  - {activo:8}: {fi} a {ff} ({N} días)")
    print("  --> Sugerencia: revisá tus archivos, algunos pueden tener pocos días o fechas desfasadas.")
    exit()

print(f"📅 Fechas disponibles tras limpieza: {df_merged.index.min()} a {df_merged.index.max()} ({len(df_merged)} días)")
print(f"🟦 Activos encontrados: {list(df_merged.columns)}")

# -- ANÁLISIS QUANT --
retornos = np.log(df_merged / df_merged.shift(1)).dropna()
print("🔎 Retornos shape:", retornos.shape)
if retornos.empty:
    print("❌ No hay retornos calculados (puede que haya sólo un día de solapamiento).")
    exit()

cor_matrix_pearson = retornos.corr()
cor_matrix_spearman = retornos.corr(method='spearman')

from sklearn.decomposition import PCA
retornos_for_pca = retornos.dropna(axis=1, how='any')
retornos_for_pca.columns = [str(col) for col in retornos_for_pca.columns]

if retornos_for_pca.shape[0] == 0:
    print("❗️ATENCIÓN: No hay datos suficientes para hacer PCA (la matriz de retornos quedó vacía).")
    explained_var = np.array([])
    components = pd.DataFrame()
else:
    pca = PCA()
    pca.fit(retornos_for_pca)
    explained_var = pca.explained_variance_ratio_
    components = pd.DataFrame(
        pca.components_,
        columns=retornos_for_pca.columns,
        index=[f'PC{i+1}' for i in range(pca.components_.shape[0])]
    )

corrs = cor_matrix_pearson.where(~np.eye(cor_matrix_pearson.shape[0], dtype=bool)).abs().unstack().sort_values(ascending=False)
top_pairs = []
for (a1, a2) in corrs.index:
    if a1 != a2 and (a2, a1) not in top_pairs:
        top_pairs.append((a1, a2))
    if len(top_pairs) == 2:
        break
rolling_corrs = {}
for a1, a2 in top_pairs:
    rolling_corrs[f'{a1}-{a2}'] = retornos[a1].rolling(ROLLING_WINDOW).corr(retornos[a2])

def calcular_beta(activo, bench, retornos):
    cov = retornos[[activo, bench]].cov().iloc[0, 1]
    var = retornos[bench].var()
    return cov / var if var != 0 else np.nan
betas = {}
for activo in retornos.columns:
    if activo != BENCHMARK:
        betas[activo] = calcular_beta(activo, BENCHMARK, retornos)

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
linkage_matrix = linkage(1 - cor_matrix_pearson.abs(), method='ward')
cluster_labels = fcluster(linkage_matrix, t=4, criterion='maxclust')
cluster_df = pd.DataFrame({'Activo': cor_matrix_pearson.columns, 'Cluster': cluster_labels})

# Exportar a Excel multihoja
excel_path = os.path.join(CARPETA_SALIDA, 'analisis_completo_portafolio.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    cor_matrix_pearson.round(4).to_excel(writer, sheet_name="Correlacion_Pearson")
    cor_matrix_spearman.round(4).to_excel(writer, sheet_name="Correlacion_Spearman")
    if explained_var.size > 0:
        pd.DataFrame(explained_var, columns=['Varianza Explicada']).to_excel(writer, sheet_name="PCA_Varianza")
        components.T.round(4).to_excel(writer, sheet_name="PCA_Loadings")
    pd.DataFrame(betas.items(), columns=['Activo', f"Beta_{BENCHMARK}"]).to_excel(writer, sheet_name="Betas")
    cluster_df.to_excel(writer, sheet_name="Clusters", index=False)
    for nombre, serie in rolling_corrs.items():
        serie.to_frame().to_excel(writer, sheet_name=f"Rolling_{nombre[:20]}")

print(f"\n✅ Archivo Excel completo generado: {excel_path}")

# Heatmap de correlación
plt.figure(figsize=(12, 10))
sns.set(style="white", font_scale=1.1)
mask = np.triu(np.ones_like(cor_matrix_pearson, dtype=bool))
sns.heatmap(cor_matrix_pearson, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=.5, cbar_kws={"shrink": .8, 'label': 'Correlación'})
plt.title("Matriz de Correlación entre Activos", fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
img_path = os.path.join(CARPETA_SALIDA, 'heatmap_correlacion.png')
plt.savefig(img_path)
plt.close()

# Dendrograma de clustering
plt.figure(figsize=(12, 5))
dendrogram(linkage_matrix, labels=cor_matrix_pearson.columns, leaf_rotation=45)
plt.title('Dendrograma de Clustering de Activos')
plt.tight_layout()
dendro_path = os.path.join(CARPETA_SALIDA, 'dendrograma_clustering.png')
plt.savefig(dendro_path)
plt.close()

# Varianza explicada (PCA)
if explained_var.size > 0:
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(explained_var)+1), explained_var*100)
    plt.xlabel('Componente Principal')
    plt.ylabel('% de Varianza Explicada')
    plt.title('Varianza Explicada por PCA')
    plt.tight_layout()
    pca_path = os.path.join(CARPETA_SALIDA, 'pca_varianza.png')
    plt.savefig(pca_path)
    plt.close()
else:
    pca_path = None

# Rolling Correlation plots
for nombre, serie in rolling_corrs.items():
    plt.figure(figsize=(10, 4))
    plt.plot(serie.index, serie, label=nombre)
    plt.axhline(0, color='gray', ls='--')
    plt.title(f'Correlación Rolling 60 días: {nombre}')
    plt.ylabel('Correlación')
    plt.xlabel('Fecha')
    plt.legend()
    plt.tight_layout()
    roll_path = os.path.join(CARPETA_SALIDA, f'rollingcorr_{nombre}.png')
    plt.savefig(roll_path)
    plt.close()

# PDF profesional con todo
from fpdf import FPDF

pdf_path = os.path.join(CARPETA_SALIDA, "informe_quant_financiero_leocaliva.pdf")
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "Informe Quant Financiero", ln=True, align="C")

pdf.set_font("Arial", '', 12)
pdf.cell(0, 10, f"Autor: Leonardo Caliva", ln=True, align="C")
pdf.cell(0, 10, f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y')}", ln=True, align="C")
pdf.cell(0, 10, "Sitio web: https://leocaliva.com", ln=True, align="C")
pdf.ln(5)

if os.path.exists(img_path):
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Matriz de Correlación Pearson", ln=True)
    pdf.image(img_path, x=15, w=180)
    pdf.ln(10)

if os.path.exists(dendro_path):
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Clustering jerárquico de activos", ln=True)
    pdf.image(dendro_path, x=15, w=180)

if pca_path and os.path.exists(pca_path):
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Varianza Explicada por PCA", ln=True)
    pdf.image(pca_path, x=30, w=150)

for nombre, serie in rolling_corrs.items():
    roll_path = os.path.join(CARPETA_SALIDA, f'rollingcorr_{nombre}.png')
    if os.path.exists(roll_path):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Correlación Rolling: {nombre}", ln=True)
        pdf.image(roll_path, x=15, w=180)

pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, f"Betas respecto a {BENCHMARK}", ln=True)
pdf.set_font("Arial", '', 11)
for activo, beta in betas.items():
    pdf.cell(0, 8, f"{activo}: Beta = {beta:.3f}", ln=True)  # Usar 'Beta' (texto) para evitar error unicode

pdf.add_page()
pdf.set_font("Arial", 'I', 10)
pdf.multi_cell(0, 8, f"""
Este informe integra análisis de correlación de Pearson y Spearman, clusters jerárquicos, rolling correlations dinámicas, análisis PCA y exposición beta respecto al benchmark {BENCHMARK}.
Generado automáticamente para Leonardo Caliva. 
Contacto y más info en https://leocaliva.com
""")
pdf.output(pdf_path)

print(f"\n📄 PDF profesional generado en: {pdf_path}")
