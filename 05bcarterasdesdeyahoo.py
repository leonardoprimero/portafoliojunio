#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fpdf import FPDF
from pypfopt import expected_returns, risk_models, plotting
from pypfopt.efficient_frontier import EfficientFrontier
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy import stats
import warnings
import textwrap

# Configurar matplotlib para evitar problemas de renderizado
plt.switch_backend("Agg")
plt.style.use("seaborn-v0_8")

# Configurar fuentes para gráficos
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN GLOBAL ---
CARPETA_DATOS_LOCALES = "./datospython1"  # carpeta con CSVs o Excels
CARPETA_SALIDA = "DatosCartera"
CARPETA_GRAFICOS_TEMP = os.path.join(CARPETA_SALIDA, "temp_graficos")
BENCHMARK_DEFAULT = "SPY"

# Asegurar que las carpetas existan
os.makedirs(CARPETA_SALIDA, exist_ok=True)
os.makedirs(CARPETA_GRAFICOS_TEMP, exist_ok=True)

# --- FUNCIONES DE UTILIDAD ---
def cargar_datos_locales():
    """
    Carga datos de precios desde archivos CSV locales en CARPETA_DATOS_LOCALES.
    """
    dataframes = {}
    for archivo in os.listdir(CARPETA_DATOS_LOCALES):
        nombre_sin_ext = os.path.splitext(archivo)[0]

        # Filtrar por nombre de archivo (solo mayúsculas) y extensión
        if not nombre_sin_ext.isupper() or not archivo.lower().endswith(".csv"):
            continue

        ruta = os.path.join(CARPETA_DATOS_LOCALES, archivo)
        try:
            # Leer el CSV, saltando las filas 1 y 2 (índice 0 y 1) y usando la fila 0 (la que contiene 'Price', 'Close', etc.) como encabezado
            # La columna de fecha es la primera columna (índice 0) y se parsea como fecha
            df = pd.read_csv(ruta, skiprows=[1, 2], header=0, index_col=0, parse_dates=True)
            
            # Se busca la columna 'Close' o 'Price' (en ese orden de preferencia)
            columnas = [col for col in df.columns if col.lower() == 'close']
            if not columnas:
                columnas = [col for col in df.columns if col.lower() == 'price']
            if not columnas:
                columnas = [col for col in df.columns if col.lower() in ['adj close', 'precio_cierre']]

            if columnas:
                df_filtrado = df[[columnas[0]]].rename(columns={columnas[0]: nombre_sin_ext})
                df_filtrado[nombre_sin_ext] = pd.to_numeric(df_filtrado[nombre_sin_ext], errors='coerce') # Convertir a numérico, forzando NaN en errores
                df_filtrado = df_filtrado.dropna() # Eliminar filas con NaN después de la conversión
                if not df_filtrado.empty:
                    dataframes[nombre_sin_ext] = df_filtrado
                    print(f"✅ Cargado CSV: {archivo} usando columna '{columnas[0]}'")
            else:
                print(f"⚠️ Archivo CSV {archivo} no tiene columna de precio reconocida. Saltando.")
        except Exception as e:
            print(f"❌ Error leyendo {archivo}: {e}")

    if not dataframes:
        print("❌ No se encontraron datos válidos en la carpeta local.")
        return pd.DataFrame()
    
    return pd.concat(dataframes.values(), axis=1).dropna()

def calcular_beta(activo, benchmark, retornos_combinados):
    """
    Calcula la beta de un activo con respecto a un benchmark.
    """
    if activo not in retornos_combinados.columns or benchmark not in retornos_combinados.columns:
        return np.nan
    
    cov_ab = retornos_combinados[activo].cov(retornos_combinados[benchmark])
    var_b = retornos_combinados[benchmark].var()
    if var_b == 0:
        return np.nan
    return cov_ab / var_b

def calcular_estadisticas_descriptivas(retornos_diarios):
    """
    Calcula estadísticas descriptivas completas para cada activo.
    """
    stats_dict = {}
    for activo in retornos_diarios.columns:
        serie = retornos_diarios[activo].dropna()
        stats_dict[activo] = {
            'Media': serie.mean(),
            'Mediana': serie.median(),
            'Desv_Std': serie.std(),
            'Skewness': stats.skew(serie),
            'Kurtosis': stats.kurtosis(serie),
            'Min': serie.min(),
            'Max': serie.max(),
            'Var': serie.var()
        }
    
    return pd.DataFrame(stats_dict).T

def calcular_volatilidad_movil(df_precios, ventana=60):
    """
    Calcula la volatilidad móvil para cada activo.
    """
    retornos = np.log(df_precios / df_precios.shift(1)).dropna()
    volatilidad_movil = retornos.rolling(window=ventana).std() * np.sqrt(252)
    return volatilidad_movil

def simulacion_monte_carlo(retornos_diarios, num_simulaciones=15000):
    """
    Realiza simulación Monte Carlo para optimización de portafolios.
    """
    print(f"🎲 Iniciando simulación Monte Carlo con {num_simulaciones} iteraciones...")
    
    carteras = []
    datos_activos = []
    
    # Calcular estadísticas de activos individuales
    for ticker in retornos_diarios.columns:
        d = {}
        d['ticker'] = ticker
        d['retorno'] = retornos_diarios[ticker].mean() * 252
        d['volatilidad'] = retornos_diarios[ticker].std() * np.sqrt(252)
        d['sharpe'] = d['retorno'] / d['volatilidad'] if d['volatilidad'] != 0 else 0
        datos_activos.append(d)
    
    datos_activos = pd.DataFrame(datos_activos, index=retornos_diarios.columns)
    
    # Simulación Monte Carlo
    for i in range(num_simulaciones):
        # Generar pesos aleatorios
        pesos = np.array(np.random.random(len(retornos_diarios.columns)))
        pesos = pesos / np.sum(pesos)  # Normalizar para que sumen 1
        
        # Calcular métricas del portafolio
        r = {}
        r['retorno'] = np.sum(retornos_diarios.mean() * pesos * 252)
        r['volatilidad'] = np.sqrt(np.dot(pesos, np.dot(retornos_diarios.cov() * 252, pesos)))
        r['sharpe'] = r['retorno'] / r['volatilidad'] if r['volatilidad'] != 0 else 0
        r['pesos'] = pesos.round(5)
        carteras.append(r)
    
    carteras_df = pd.DataFrame(carteras)
    
    # Encontrar portafolio óptimo
    optimo = carteras_df.loc[carteras_df.sharpe.idxmax()]
    mejor_portafolio = optimo['pesos']
    datos_activos['ponderacion_optima'] = mejor_portafolio
    
    print(f"✅ Simulación Monte Carlo completada. Mejor Sharpe Ratio: {optimo.sharpe:.4f}")
    
    return carteras_df, datos_activos, optimo

def limpiar_texto(texto):
    """
    Limpia el texto para evitar problemas de codificación en el PDF.
    """
    if isinstance(texto, str):
        # Reemplazar caracteres problemáticos
        replacements = {
            'ñ': 'n', 'Ñ': 'N',
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ü': 'u', 'Ü': 'U',
            '°': ' grados', '€': 'EUR', '£': 'GBP', '¥': 'JPY',
            '–': '-', '—': '-', '‘': "'", '’': "'", '“': '"', '”': '"',
            '•': '-', '●': '-', '◦': '-', '★': '*', '☆': '*',
            '→': '->', '←': '<-', '↑': '^', '↓': 'v',
            '≥': '>=', '≤': '<=', '≠': '!=', '±': '+/-',
            '…': '...', '‰': 'por mil', '∞': 'infinito'
        }
        for old, new in replacements.items():
            texto = texto.replace(old, new)
        
        # Remover cualquier carácter que no sea ASCII básico
        texto = ''.join(char for char in texto if ord(char) < 256)
        
        # Codificar y decodificar para asegurar compatibilidad
        try:
            texto = texto.encode('latin-1', 'replace').decode('latin-1')
        except:
            # Si aún hay problemas, usar solo caracteres básicos
            texto = ''.join(char for char in texto if ord(char) < 128)
    
    return texto

def dividir_texto_en_lineas(texto, max_chars=80):
    """
    Divide el texto en líneas que no excedan el ancho máximo.
    """
    texto = limpiar_texto(texto)
    if len(texto) <= max_chars:
        return texto
    
    # Usar textwrap para dividir el texto correctamente
    lineas = textwrap.wrap(texto, width=max_chars, break_long_words=True)
    return '\n'.join(lineas)

# --- CLASE PDF MEJORADA ---
class PDFMejorado(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(20, 20, 20)  # Márgenes más amplios
        
    def header(self):
        # Solo mostrar header en páginas que no sean la portada
        if self.page_no() > 1:
            # Línea superior decorativa
            self.set_line_width(0.3)
            self.line(20, 15, self.w - 20, 15)
            
            # Header con información profesional
            self.set_font('Arial', 'B', 10)
            texto_header = limpiar_texto('Informe Cuantitativo de Carteras')
            self.cell(0, 8, texto_header, 0, 0, 'L')
            
            # Fecha en el header derecho
            fecha_header = datetime.now().strftime("%d-%m-%Y")
            self.cell(0, 8, f"Leonardo Caliva | {fecha_header}", 0, 1, 'R')
            
            # Línea inferior del header
            self.ln(2)
            self.set_line_width(0.2)
            self.line(20, self.get_y(), self.w - 20, self.get_y())
            self.ln(8)

    def footer(self):
        # Solo mostrar footer en páginas que no sean la portada
        if self.page_no() > 1:
            self.set_y(-20)
            
            # Línea superior del footer
            self.set_line_width(0.2)
            self.line(20, self.get_y(), self.w - 20, self.get_y())
            
            self.ln(3)
            
            # Footer profesional con información dividida
            self.set_font('Arial', 'I', 8)
            
            # Lado izquierdo: información del autor
            self.cell(0, 6, 'www.leocaliva.com', 0, 0, 'L')
            
            # Centro: número de página
            self.cell(0, 6, f'Pagina {self.page_no()}', 0, 0, 'C')
            
            # Lado derecho: disclaimer
            self.cell(0, 6, '@leonardoprimero        ', 0, 1, 'R')

    def chapter_title(self, title):
        title = limpiar_texto(title)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(3)

    def chapter_body(self, body):
        body = limpiar_texto(body)
        self.set_font('Arial', '', 10)
        # Dividir en párrafos y procesar cada uno
        paragrafos = body.split('\n')
        for paragrafo in paragrafos:
            if paragrafo.strip():
                # Usar multi_cell para ajuste automático de texto
                self.multi_cell(0, 6, paragrafo, 0, 'L')
                self.ln(2)

    def add_cover_page_profesional(self):
        """Crea una portada profesional con el diseño específico solicitado"""
        self.add_page()
        
        # Marco decorativo superior
        self.set_line_width(1.0)
        self.line(20, 25, self.w - 20, 25)
        
        # TÍTULO PRINCIPAL - Centrado, grande y negrita
        self.ln(35)  # Espacio desde arriba
        self.set_font('Arial', 'B', 24)
        titulo_principal = limpiar_texto("Informe Completo de Analisis Cuantitativo de Carteras")
        
        # Dividir el título en líneas para mejor formato
        lineas_titulo = [
            "Informe Completo de",
            "Analisis Cuantitativo",
            "de Carteras"
        ]
        
        for linea in lineas_titulo:
            self.cell(0, 12, limpiar_texto(linea), 0, 1, 'C')
        
        # SUBTÍTULO - Centrado, tamaño más pequeño
        self.ln(12)
        self.set_font('Arial', 'I', 16)
        subtitulo = limpiar_texto("Analisis Cuantitativo Avanzado")
        self.cell(0, 10, subtitulo, 0, 1, 'C')
        
        # Línea decorativa debajo del título
        self.ln(15)
        self.set_line_width(0.5)
        self.line(60, self.get_y(), self.w - 60, self.get_y())
        
        # Espacio considerable hacia abajo
        self.ln(40)
        
        # INFORMACIÓN DEL AUTOR - Alineado a la izquierda con formato profesional
        # Crear un bloque de información del autor
        self.set_font('Arial', 'B', 16)
        self.cell(0, 12, "AUTOR", 0, 1, 'L')
        
        self.ln(5)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, "Leonardo Caliva", 0, 1, 'L')
        
        self.ln(3)
        self.set_font('Arial', '', 12)
        self.cell(0, 8, "Quant y Desarrollador de Sistemas Algoritmicos", 0, 1, 'L')
        
        self.ln(12)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, "CONTACTO", 0, 1, 'L')
        
        self.ln(3)
        self.set_font('Arial', '', 11)
        self.cell(0, 7, "Sitio Web: www.leocaliva.com", 0, 1, 'L')
        
        self.ln(3)
        self.cell(0, 7, "GitHub: Leonardo I (a.k.a. leonardoprimero)", 0, 1, 'L')
        
        # FECHA - Destacada en un recuadro
        self.ln(45)
        fecha_actual = datetime.now().strftime("%d-%m-%Y")
        
        # Crear un recuadro para la fecha
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, "FECHA DE GENERACION", 0, 1, 'L')
        
        self.ln(3)
        self.set_font('Arial', '', 14)
        self.cell(0, 10, fecha_actual, 0, 1, 'L')
        
        # Línea decorativa al final
        self.ln(20)
        self.set_line_width(0.5)
        self.line(20, self.get_y(), self.w - 20, self.get_y())
        
        # Información adicional al pie - más profesional
        self.ln(8)
        self.set_font('Arial', 'I', 9)
        texto_pie = limpiar_texto("Analisis basado en datos historicos de mercado")
        self.cell(0, 6, texto_pie, 0, 1, 'C')
        
        self.ln(2)
        self.set_font('Arial', 'I', 9)
        texto = "Este informe fue generado mediante modelos cuantitativos avanzados y refleja análisis reproducibles con fundamentos estadísticos y financieros."
        self.multi_cell(0, 5, limpiar_texto(texto), 0, 'C')
        
        # Marco decorativo inferior
        self.ln(8)
        self.set_line_width(1.0)
        self.line(20, self.h - 25, self.w - 20, self.h - 25)

    def add_image_with_caption(self, image_path, caption):
        if os.path.exists(image_path):
            # Calcular tamaño de imagen para que quepa en la página
            page_width = self.w - 2 * self.l_margin
            max_width = min(page_width, 160)  # Máximo 160mm de ancho
            
            self.image(image_path, x=self.l_margin, w=max_width)
            self.ln(5)
            
            caption = limpiar_texto(caption)
            self.set_font('Arial', 'I', 9)
            self.multi_cell(0, 5, caption, 0, 'C')
            self.ln(8)
        else:
            caption = limpiar_texto(caption)
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f"[Imagen no encontrada: {caption}]", 0, 1, 'C')
            self.ln(5)

    def add_table_mejorada(self, dataframe, title="", max_rows=20):
        """Añade una tabla mejorada al PDF desde un DataFrame"""
        if title:
            self.chapter_title(title)
        
        # Verificar si tenemos datos
        if dataframe.empty:
            self.chapter_body("No hay datos disponibles para mostrar.")
            return
        
        # Limitar número de filas si es muy grande
        if len(dataframe) > max_rows:
            df_display = dataframe.head(max_rows)
            self.chapter_body(f"Mostrando las primeras {max_rows} filas.")
        else:
            df_display = dataframe.copy()

        # Preparar datos para la tabla
        # Asegurarse de que el índice sea una columna si no lo es ya
        df_temp = df_display.reset_index() if df_display.index.name else df_display.copy()
        
        # Convertir todas las columnas a string para evitar problemas de formato
        df_temp = df_temp.astype(str)

        # Calcular ancho de columnas dinámicamente
        col_widths = []
        for col in df_temp.columns:
            # Ancho mínimo, o el máximo entre el largo del encabezado y el largo de los datos
            max_data_len = df_temp[col].apply(len).max()
            col_widths.append(max(self.get_string_width(col) + 6, max_data_len * 1.5))
        
        # Normalizar anchos para que sumen el ancho de la página
        total_width = sum(col_widths)
        page_width = self.w - 2 * self.l_margin
        
        if total_width > page_width:
            scale_factor = page_width / total_width
            col_widths = [w * scale_factor for w in col_widths]
        
        # Añadir encabezados
        self.set_font('Arial', 'B', 8)
        for i, col in enumerate(df_temp.columns):
            self.cell(col_widths[i], 8, limpiar_texto(col), 1, 0, 'C')
        self.ln()
        
        # Añadir filas de datos
        self.set_font('Arial', '', 8)
        for index, row in df_temp.iterrows():
            for i, cell_value in enumerate(row):
                self.cell(col_widths[i], 8, limpiar_texto(cell_value), 1, 0, 'C')
            self.ln()
        self.ln(5)

# --- CÓDIGO PRINCIPAL ---
df_precios = cargar_datos_locales()

if df_precios.empty:
    print("❌ No se encontraron datos válidos para procesar. Saliendo.")
    exit()

# Eliminar el benchmark si está presente en los datos
if BENCHMARK_DEFAULT in df_precios.columns:
    df_precios = df_precios.drop(columns=[BENCHMARK_DEFAULT])

if df_precios.empty:
    print("❌ No hay datos de precios después de la limpieza. Saliendo.")
    exit()

# Calcular retornos diarios
retornos_diarios = df_precios.pct_change().dropna()

if retornos_diarios.empty:
    print("❌ No hay retornos diarios válidos después de la limpieza. Saliendo.")
    exit()

# --- 1. Estadísticas Descriptivas ---
print("📊 Calculando estadísticas descriptivas...")
estadisticas_df = calcular_estadisticas_descriptivas(retornos_diarios)
print("✅ Estadísticas descriptivas calculadas.")

# --- 2. Matriz de Correlación ---
print("📈 Calculando matriz de correlación...")
matriz_correlacion = retornos_diarios.corr()
print("✅ Matriz de correlación calculada.")

# --- 3. Volatilidad Móvil ---
print("📉 Calculando volatilidad móvil...")
volatilidad_movil_df = calcular_volatilidad_movil(df_precios)
print("✅ Volatilidad móvil calculada.")

# --- 4. Simulación Monte Carlo ---
carteras_df, datos_activos, optimo = simulacion_monte_carlo(retornos_diarios)

# --- 5. Frontera Eficiente (PyPortfolioOpt) ---
print("✨ Calculando Frontera Eficiente...")
mu = expected_returns.mean_historical_return(df_precios)
S = risk_models.sample_cov(df_precios)

# Crear una nueva instancia de EfficientFrontier para el plotting
ef_plot = EfficientFrontier(mu, S)

ef = EfficientFrontier(mu, S)
weights_max_sharpe = ef.max_sharpe()
cleaned_weights_sharpe = ef.clean_weights()

ef_vol = EfficientFrontier(mu, S)
weights_min_volatility = ef_vol.min_volatility()
cleaned_weights_min_volatility = ef_vol.clean_weights()

print("✅ Frontera Eficiente calculada.")

# --- 6. Análisis de Componentes Principales (PCA) ---
print("🔍 Realizando Análisis de Componentes Principales (PCA)...")
pca = PCA().fit(retornos_diarios)
explained_variance_ratio = pca.explained_variance_ratio_
print("✅ PCA completado.")

# --- 7. Clustering Jerárquico ---
print("🌳 Realizando Clustering Jerárquico...")
# Calcular la matriz de distancia (1 - correlación absoluta)
distance_matrix = 1 - matriz_correlacion.abs()
# Convertir la matriz de distancia a formato condensado
condensed_distance = distance_matrix.values[np.triu_indices_from(distance_matrix, k=1)]
# Realizar el clustering jerárquico
Z = linkage(condensed_distance, method='ward')

# Asignar clusters (por ejemplo, 3 clusters)
num_clusters = min(3, len(matriz_correlacion.columns))
clusters = fcluster(Z, num_clusters, criterion='maxclust')
cluster_map = pd.DataFrame({'Activo': matriz_correlacion.columns, 'Cluster': clusters})
print("✅ Clustering Jerárquico completado.")

# --- GENERACIÓN DE GRÁFICOS ---
print("🎨 Generando gráficos...")

# Gráfico 1: Heatmap de Correlación
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación de Retornos')
plt.tight_layout()
heatmap_path = os.path.join(CARPETA_GRAFICOS_TEMP, 'heatmap_correlacion.png')
plt.savefig(heatmap_path)
plt.close()

# Gráfico 2: Frontera Eficiente
plt.figure(figsize=(10, 6))
plotting.plot_efficient_frontier(ef_plot, show_assets=True)
plt.title('Frontera Eficiente')
plt.tight_layout()
efficient_frontier_path = os.path.join(CARPETA_GRAFICOS_TEMP, 'frontera_eficiente.png')
plt.savefig(efficient_frontier_path)
plt.close()

# Gráfico 3: Pesos del Portafolio de Máximo Sharpe
plt.figure(figsize=(8, 6))
plotting.plot_weights(cleaned_weights_sharpe)
plt.title('Pesos del Portafolio de Máximo Sharpe')
plt.tight_layout()
weights_sharpe_path = os.path.join(CARPETA_GRAFICOS_TEMP, 'pesos_max_sharpe.png')
plt.savefig(weights_sharpe_path)
plt.close()

# Gráfico 4: Pesos del Portafolio de Mínima Volatilidad
plt.figure(figsize=(8, 6))
plotting.plot_weights(cleaned_weights_min_volatility)
plt.title('Pesos del Portafolio de Mínima Volatilidad')
plt.tight_layout()
weights_min_vol_path = os.path.join(CARPETA_GRAFICOS_TEMP, 'pesos_min_volatilidad.png')
plt.savefig(weights_min_vol_path)
plt.close()

# Gráfico 5: Varianza Explicada por PCA
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100)
plt.xlabel('Componente Principal')
plt.ylabel('% de Varianza Explicada')
plt.title('Varianza Explicada por PCA')
plt.tight_layout()
pca_variance_path = os.path.join(CARPETA_GRAFICOS_TEMP, 'pca_varianza_explicada.png')
plt.savefig(pca_variance_path)
plt.close()

# Gráfico 6: Dendrograma
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=matriz_correlacion.columns, leaf_rotation=90)
plt.title('Dendrograma de Clustering Jerárquico')
plt.ylabel('Distancia')
plt.tight_layout()
dendrogram_path = os.path.join(CARPETA_GRAFICOS_TEMP, 'dendrograma.png')
plt.savefig(dendrogram_path)
plt.close()

# Gráfico 7: Simulación Monte Carlo
plt.figure(figsize=(10, 6))
plt.scatter(carteras_df.volatilidad, carteras_df.retorno, c=carteras_df.sharpe, cmap='viridis')
plt.title('Simulación Monte Carlo de Portafolios')
plt.xlabel('Volatilidad')
plt.ylabel('Retorno Anualizado')
plt.scatter(optimo.volatilidad, optimo.retorno, color='red', marker='*', s=200, label='Portafolio Óptimo')
plt.legend()
plt.tight_layout()
monte_carlo_path = os.path.join(CARPETA_GRAFICOS_TEMP, 'monte_carlo_simulacion.png')
plt.savefig(monte_carlo_path)
plt.close()

print("✅ Gráficos generados.")

# --- GENERACIÓN DE PDF ---
print("📄 Generando informe PDF...")
pdf = PDFMejorado()
pdf.add_cover_page_profesional()

# Resumen Ejecutivo
pdf.add_page()
pdf.chapter_title("Resumen Ejecutivo")
resumen_ejecutivo_texto = (
    "Este informe presenta un análisis cuantitativo exhaustivo de la cartera de inversión, "
    "abarcando desde estadísticas descriptivas de los activos hasta la optimización de portafolios "
    "mediante simulación Monte Carlo y la Frontera Eficiente. Se incluyen análisis avanzados "
    "como PCA y clustering jerárquico para una comprensión profunda de la estructura de la cartera."
)
pdf.chapter_body(resumen_ejecutivo_texto)

# Estadísticas Descriptivas
pdf.add_page()
pdf.add_table_mejorada(estadisticas_df.round(4), title="Estadísticas Descriptivas de Retornos Diarios")

# Matriz de Correlación
pdf.add_page()
pdf.chapter_title("Matriz de Correlación de Retornos")
pdf.add_image_with_caption(heatmap_path, "Heatmap de la Matriz de Correlación de Retornos Diarios")

# Volatilidad Móvil
pdf.add_page()
pdf.chapter_title("Volatilidad Móvil Anualizada (Ventana de 60 días)")
# Generar gráficos de volatilidad móvil para cada activo
for col in volatilidad_movil_df.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(volatilidad_movil_df.index, volatilidad_movil_df[col])
    plt.title(f'Volatilidad Móvil de {col}')
    plt.xlabel('Fecha')
    plt.ylabel('Volatilidad Anualizada')
    plt.tight_layout()
    vol_path = os.path.join(CARPETA_GRAFICOS_TEMP, f'volatilidad_movil_{col}.png')
    plt.savefig(vol_path)
    plt.close()
    pdf.add_image_with_caption(vol_path, f"Volatilidad Móvil Anualizada de {col}")

# Simulación Monte Carlo
pdf.add_page()
pdf.chapter_title("Simulación Monte Carlo para Optimización de Portafolios")
pdf.add_image_with_caption(monte_carlo_path, "Gráfico de Simulación Monte Carlo de Portafolios (puntos coloreados por Sharpe Ratio)")
pdf.add_table_mejorada(datos_activos.round(4), title="Resultados de Simulación Monte Carlo por Activo")

# Frontera Eficiente
pdf.add_page()
pdf.chapter_title("Frontera Eficiente y Optimización de Portafolios")
pdf.add_image_with_caption(efficient_frontier_path, "Frontera Eficiente (curva de riesgo-retorno óptima)")
pdf.add_image_with_caption(weights_sharpe_path, "Pesos del Portafolio de Máximo Sharpe Ratio")
pdf.add_image_with_caption(weights_min_vol_path, "Pesos del Portafolio de Mínima Volatilidad")

# Análisis de Componentes Principales (PCA)
pdf.add_page()
pdf.chapter_title("Análisis de Componentes Principales (PCA)")
pdf.add_image_with_caption(pca_variance_path, "Varianza Explicada por cada Componente Principal")
pdf.chapter_body(
    "El PCA ayuda a reducir la dimensionalidad de los datos, identificando los factores "
    "subyacentes que explican la mayor parte de la varianza en los retornos de los activos. "
    "Los primeros componentes principales suelen representar los movimientos más amplios del mercado."
)

# Clustering Jerárquico
pdf.add_page()
pdf.chapter_title("Clustering Jerárquico de Activos")
pdf.add_image_with_caption(dendrogram_path, "Dendrograma de Clustering Jerárquico de Activos")
pdf.add_table_mejorada(cluster_map, title="Asignación de Activos a Clusters")
pdf.chapter_body(
    "El clustering jerárquico agrupa activos con comportamientos de retorno similares, "
    "revelando relaciones y dependencias naturales dentro de la cartera. "
    "Esto puede ser útil para la diversificación y la gestión de riesgos sectoriales."
)

# Guardar PDF
pdf_final_path = os.path.join(CARPETA_SALIDA, "informe_cartera_cuantitativo.pdf")
try:
    pdf.output(pdf_final_path)
    print(f"✅ Informe PDF generado en: {pdf_final_path}")
except Exception as e:
    print(f"❌ Error al generar el PDF: {e}")

# Limpiar gráficos temporales
for f in os.listdir(CARPETA_GRAFICOS_TEMP):
    os.remove(os.path.join(CARPETA_GRAFICOS_TEMP, f))
os.rmdir(CARPETA_GRAFICOS_TEMP)
print("✅ Archivos temporales de gráficos eliminados.")


