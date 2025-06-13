#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# correlaciones.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import numpy as np

# ------------------------ CONFIG ------------------------
CARPETA_DATOS = './datospython1'
CARPETA_SALIDA = './graficos_temp'
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ------------------------ LECTURA DE DATOS ------------------------
dataframes = {}
for archivo in os.listdir(CARPETA_DATOS):
    ruta = os.path.join(CARPETA_DATOS, archivo)
    nombre_activo = os.path.splitext(archivo)[0]

    try:
        if archivo.endswith('.csv'):
            df = pd.read_csv(ruta, index_col=0, parse_dates=True)
        elif archivo.endswith('.xlsx') or archivo.endswith('.xls'):
            df = pd.read_excel(ruta, index_col=0, parse_dates=True)
        else:
            continue

        # Buscar columnas de precios
        columnas_validas = [col for col in df.columns if col.lower() in ['adj close', 'close', 'precio_cierre']]
        if not columnas_validas:
            continue

        col = columnas_validas[0]
        df_filtrado = df[[col]].rename(columns={col: nombre_activo})
        dataframes[nombre_activo] = df_filtrado

    except Exception as e:
        print(f"⚠️ Error procesando {archivo}: {e}")

if not dataframes:
    print("❌ No se encontraron archivos válidos.")
    exit()

# ------------------------ TRANSFORMACIÓN ------------------------
df_precios = pd.concat(dataframes.values(), axis=1).dropna()
retornos = np.log(df_precios / df_precios.shift(1)).dropna()
cor_matrix = retornos.corr()

# ------------------------ EXPORTACIÓN EXCEL ------------------------
excel_path = os.path.join(CARPETA_SALIDA, 'matriz_correlacion.xlsx')
wb = Workbook()
ws = wb.active
ws.title = "Correlaciones"
for fila in dataframe_to_rows(cor_matrix.round(4), index=True, header=True):
    ws.append(fila)
wb.save(excel_path)

# ------------------------ HEATMAP ------------------------
plt.figure(figsize=(12, 10))
sns.set(style="white", font_scale=1.1)
mask = np.triu(np.ones_like(cor_matrix, dtype=bool))
sns.heatmap(cor_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=.5, cbar_kws={"shrink": .8, 'label': 'Correlación'})
plt.title("Matriz de Correlación entre Activos", fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
img_path = os.path.join(CARPETA_SALIDA, 'heatmap_correlacion.png')
plt.savefig(img_path)
plt.close()

# ------------------------ OUTPUT ------------------------
print(f"\n✅ Matriz de correlación generada correctamente:")
print(f"- Excel: {excel_path}")
print(f"- Imagen: {img_path}")
print("\n🧠 Vista previa de la matriz:")
print(cor_matrix.round(2))
