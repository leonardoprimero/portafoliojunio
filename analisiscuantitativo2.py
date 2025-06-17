#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yfinance as yf
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

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE FUENTE DE DATOS ---
FUENTE_DATOS = "web"  # "web" o "local"

if FUENTE_DATOS == "web":
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
    FECHA_INICIO = "2020-01-01"
    FECHA_FIN = "2024-12-31"
else:
    CARPETA_LOCALES = "./carpeta_datos_locales"  # carpeta con CSVs o Excels

# --- CONFIGURACIÓN GLOBAL ---
CARPETA_SALIDA = "DatosCartera"
CARPETA_GRAFICOS_TEMP = os.path.join(CARPETA_SALIDA, "temp_graficos")
CARPETA_DATOS_CACHE = os.path.join(CARPETA_SALIDA, "data_cache")
BENCHMARK_DEFAULT = "SPY"

# Asegurar que las carpetas existan
os.makedirs(CARPETA_SALIDA, exist_ok=True)
os.makedirs(CARPETA_GRAFICOS_TEMP, exist_ok=True)
os.makedirs(CARPETA_DATOS_CACHE, exist_ok=True)

# --- FUNCIONES DE UTILIDAD ---
dataframes = {}

if FUENTE_DATOS == "web":
    import yfinance as yf
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=FECHA_INICIO, end=FECHA_FIN, progress=False)
            if 'Adj Close' in df.columns:
                df = df[['Adj Close']].rename(columns={'Adj Close': ticker})
            else:
                df = df[['Close']].rename(columns={'Close': ticker})
            dataframes[ticker] = df
        except Exception as e:
            print(f"❌ Error al descargar {ticker}: {e}")

elif FUENTE_DATOS == "local":
    import os
    import pandas as pd

    for archivo in os.listdir(CARPETA_LOCALES):
        if archivo.endswith((".csv", ".xlsx", ".xls")):
            ruta = os.path.join(CARPETA_LOCALES, archivo)
            nombre = os.path.splitext(archivo)[0].upper()
            try:
                if archivo.endswith(".csv"):
                    df = pd.read_csv(ruta, index_col=0, parse_dates=True)
                else:
                    df = pd.read_excel(ruta, index_col=0, parse_dates=True)

                if 'Adj Close' in df.columns:
                    df = df[['Adj Close']].rename(columns={'Adj Close': nombre})
                else:
                    df = df[['Close']].rename(columns={'Close': nombre})

                dataframes[nombre] = df
            except Exception as e:
                print(f"❌ Error leyendo {archivo}: {e}")

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
    Basado en el código de montecarlo.py proporcionado.
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
    
    datos_activos = pd.DataFrame(datos_activos).set_index('ticker')
    
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
    mejor_portafolio = carteras_df.iloc[carteras_df.sharpe.idxmax()]['pesos']
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
            '–': '-', '—': '-', ''': "'", ''': "'", '"': '"', '"': '"',
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
            self.chapter_body(f"Mostrando las primeras {max_rows} filas de {len(dataframe)} total.")
        else:
            df_display = dataframe
        
        # Calcular ancho de columnas
        available_width = self.w - 2 * self.l_margin - 10
        num_cols = len(df_display.columns) + 1  # +1 para el índice
        col_width = available_width / num_cols
        
        # Asegurar ancho mínimo y máximo
        col_width = max(15, min(col_width, 35))
        
        # Encabezados
        self.set_font('Arial', 'B', 8)
        
        # Índice
        self.cell(col_width, 7, 'Activo', 1, 0, 'C')
        
        # Columnas
        for col in df_display.columns:
            col_text = limpiar_texto(str(col))
            if len(col_text) > 12:
                col_text = col_text[:12] + '...'
            self.cell(col_width, 7, col_text, 1, 0, 'C')
        self.ln()
        
        # Datos
        self.set_font('Arial', '', 7)
        for index, row in df_display.iterrows():
            # Verificar si necesitamos nueva página
            if self.get_y() > self.h - 30:
                self.add_page()
                # Repetir encabezados
                self.set_font('Arial', 'B', 8)
                self.cell(col_width, 7, 'Activo', 1, 0, 'C')
                for col in df_display.columns:
                    col_text = limpiar_texto(str(col))
                    if len(col_text) > 12:
                        col_text = col_text[:12] + '...'
                    self.cell(col_width, 7, col_text, 1, 0, 'C')
                self.ln()
                self.set_font('Arial', '', 7)
            
            # Índice
            index_text = limpiar_texto(str(index))
            if len(index_text) > 10:
                index_text = index_text[:10] + '...'
            self.cell(col_width, 6, index_text, 1, 0, 'C')
            
            # Valores
            for col in df_display.columns:
                value = row[col]
                if isinstance(value, (int, float)):
                    if abs(value) < 0.0001 and value != 0:
                        value_text = f'{value:.2e}'
                    else:
                        value_text = f'{value:.4f}'
                else:
                    value_text = limpiar_texto(str(value))
                    if len(value_text) > 10:
                        value_text = value_text[:10] + '...'
                
                self.cell(col_width, 6, value_text, 1, 0, 'C')
            self.ln()
        
        self.ln(5)

    def add_metrics_section(self, title, metrics_dict):
        """Añade una sección de métricas formateada"""
        self.chapter_title(title)
        
        for key, value in metrics_dict.items():
            key_clean = limpiar_texto(str(key))
            if isinstance(value, (int, float)):
                if abs(value) < 1:
                    value_text = f"{value:.2%}" if abs(value) < 1 else f"{value:.4f}"
                else:
                    value_text = f"{value:.4f}"
            else:
                value_text = limpiar_texto(str(value))
            
            self.chapter_body(f"- {key_clean}: {value_text}")
        
        self.ln(3)


if __name__ == "__main__":
    # Configuración de activos y fechas
    activos_cartera = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    fecha_inicio_analisis = "2020-01-01"
    fecha_fin_analisis = "2024-12-31"
    benchmark_elegido = BENCHMARK_DEFAULT

    print("🚀 Iniciando análisis cuantitativo completo de carteras...")
    
    # Cargar o descargar datos
    df_precios = cargar_datos_locales(activos_cartera, benchmark_elegido)
    if df_precios is None or df_precios.empty:
        df_precios = descargar_datos(activos_cartera, fecha_inicio_analisis, fecha_fin_analisis, benchmark_elegido)
    
    if df_precios.empty:
        print("❌ No se pudieron obtener datos para el análisis. Saliendo.")
        exit()

    print(f"\n📊 Datos de precios cargados correctamente")
    print(f"Período: {df_precios.index[0]} a {df_precios.index[-1]}")
    print(f"Activos: {list(df_precios.columns)}")
    
    # Separar benchmark
    if benchmark_elegido in df_precios.columns:
        df_benchmark = df_precios[[benchmark_elegido]].copy()
        df_activos = df_precios.drop(columns=[benchmark_elegido]).copy()
    else:
        print(f"⚠️ El benchmark {benchmark_elegido} no se encontró en los datos.")
        df_benchmark = None
        df_activos = df_precios.copy()

    # --- CÁLCULOS PRINCIPALES ---
    
    # 1. Retornos diarios
    print("\n📈 Calculando retornos y estadísticas...")
    retornos_diarios = np.log(df_activos / df_activos.shift(1)).dropna()
    
    # 2. Estadísticas descriptivas COMPLETAS
    estadisticas_descriptivas = calcular_estadisticas_descriptivas(retornos_diarios)
    print("✅ Estadísticas descriptivas calculadas")
    
    # 3. Volatilidad móvil (60 días)
    volatilidad_movil = calcular_volatilidad_movil(df_activos, ventana=60)
    print("✅ Volatilidad móvil calculada")
    
    # 4. Matrices de correlación
    cor_matrix_pearson = retornos_diarios.corr()
    cor_matrix_spearman = retornos_diarios.corr(method="spearman")
    
    # 5. PCA
    print("\n🔍 Realizando análisis PCA...")
    retornos_for_pca = retornos_diarios.dropna(axis=1, how="any")
    if retornos_for_pca.shape[0] > 0 and retornos_for_pca.shape[1] > 0:
        pca = PCA()
        pca.fit(retornos_for_pca)
        explained_var = pca.explained_variance_ratio_
        components = pd.DataFrame(
            pca.components_,
            columns=retornos_for_pca.columns,
            index=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
        )
        print(f"✅ PCA completado. Varianza explicada por PC1: {explained_var[0]:.2%}")
    else:
        explained_var = np.array([])
        components = pd.DataFrame()
        print("❗️ No hay datos suficientes para PCA")

    # 6. Cálculo de Betas
    print("\n📊 Calculando coeficientes Beta...")
    betas = {}
    if df_benchmark is not None and not df_benchmark.empty:
        retornos_benchmark_diarios = np.log(df_benchmark / df_benchmark.shift(1)).dropna()
        retornos_combinados = pd.concat([retornos_diarios, retornos_benchmark_diarios], axis=1).dropna()
        for activo in retornos_diarios.columns:
            betas[activo] = calcular_beta(activo, benchmark_elegido, retornos_combinados)
        betas_df = pd.DataFrame(betas.items(), columns=["Activo", f"Beta_{benchmark_elegido}"])
        betas_df = betas_df.set_index("Activo")
        print(f"✅ Betas calculadas respecto a {benchmark_elegido}")
    else:
        betas_df = pd.DataFrame()
        print("⚠️ No se pueden calcular Betas: Benchmark no disponible")

    # 7. Clustering
    print("\n🎯 Realizando clustering de activos...")
    if not cor_matrix_pearson.empty and cor_matrix_pearson.shape[0] > 1:
        linkage_matrix = linkage(1 - cor_matrix_pearson.abs(), method="ward")
        cluster_labels = fcluster(linkage_matrix, t=0.5, criterion="distance")
        cluster_df = pd.DataFrame({"Activo": cor_matrix_pearson.columns, "Cluster": cluster_labels})
        cluster_df = cluster_df.set_index("Activo")
        print(f"✅ Clustering completado. {len(set(cluster_labels))} clusters identificados")
    else:
        cluster_df = pd.DataFrame()
        print("❗️ No hay datos suficientes para clustering")

    # 8. Simulación Monte Carlo
    print("\n🎲 Ejecutando simulación Monte Carlo...")
    carteras_mc, datos_activos_mc, optimo_mc = simulacion_monte_carlo(retornos_diarios, num_simulaciones=15000)

    # 9. Optimización de Markowitz - TRES PORTAFOLIOS
    print("\n📊 Calculando portafolios óptimos...")
    
    mu = expected_returns.mean_historical_return(df_activos)
    S = risk_models.sample_cov(df_activos)
    
    # Portafolio 1: Máxima Sharpe
    ef1 = EfficientFrontier(mu, S)
    weights_sharpe = ef1.max_sharpe()
    cleaned_weights_sharpe = ef1.clean_weights()
    perf_sharpe = ef1.portfolio_performance(verbose=False)
    
    # Portafolio 2: Mínima Volatilidad
    ef2 = EfficientFrontier(mu, S)
    weights_min_vol = ef2.min_volatility()
    cleaned_weights_min_vol = ef2.clean_weights()
    perf_min_vol = ef2.portfolio_performance(verbose=False)
    
    # Portafolio 3: Equal Weight
    num_assets = len(df_activos.columns)
    weights_equal = {asset: 1/num_assets for asset in df_activos.columns}
    
    # Calcular performance del Equal Weight
    equal_weights_array = np.array([1/num_assets] * num_assets)
    equal_return = np.sum(mu * equal_weights_array)
    equal_volatility = np.sqrt(np.dot(equal_weights_array, np.dot(S, equal_weights_array)))
    equal_sharpe = equal_return / equal_volatility
    perf_equal = (equal_return, equal_volatility, equal_sharpe)
    
    print(f"✅ Portafolio Máxima Sharpe - Retorno: {perf_sharpe[0]:.2%}, Volatilidad: {perf_sharpe[1]:.2%}, Sharpe: {perf_sharpe[2]:.4f}")
    print(f"✅ Portafolio Mínima Volatilidad - Retorno: {perf_min_vol[0]:.2%}, Volatilidad: {perf_min_vol[1]:.2%}, Sharpe: {perf_min_vol[2]:.4f}")
    print(f"✅ Portafolio Equal Weight - Retorno: {perf_equal[0]:.2%}, Volatilidad: {perf_equal[1]:.2%}, Sharpe: {perf_equal[2]:.4f}")

    # --- EVOLUCIÓN DE LOS TRES PORTAFOLIOS ---
    print("\n📈 Calculando evolución de los tres portafolios...")
    
    # Calcular retornos de cada portafolio
    portfolio_returns_sharpe = (retornos_diarios * pd.Series(cleaned_weights_sharpe)).sum(axis=1)
    portfolio_returns_min_vol = (retornos_diarios * pd.Series(cleaned_weights_min_vol)).sum(axis=1)
    portfolio_returns_equal = (retornos_diarios * pd.Series(weights_equal)).sum(axis=1)
    
    # Calcular evolución acumulada
    cumulative_sharpe = (1 + portfolio_returns_sharpe).cumprod()
    cumulative_min_vol = (1 + portfolio_returns_min_vol).cumprod()
    cumulative_equal = (1 + portfolio_returns_equal).cumprod()
    
    # Agregar benchmark si está disponible
    if df_benchmark is not None and not df_benchmark.empty:
        benchmark_daily_returns = np.log(df_benchmark / df_benchmark.shift(1)).dropna()
        cumulative_benchmark = (1 + benchmark_daily_returns).cumprod()
        
        # Combinar todos los retornos
        combined_returns = pd.DataFrame({
            "Maxima Sharpe": cumulative_sharpe,
            "Minima Volatilidad": cumulative_min_vol,
            "Equal Weight": cumulative_equal,
            f"Benchmark ({benchmark_elegido})": cumulative_benchmark.iloc[:, 0]
        }).dropna()
    else:
        combined_returns = pd.DataFrame({
            "Maxima Sharpe": cumulative_sharpe,
            "Minima Volatilidad": cumulative_min_vol,
            "Equal Weight": cumulative_equal
        }).dropna()

    # --- GENERACIÓN DE GRÁFICOS ---
    print("\n🎨 Generando gráficos...")
    
    # 1. Heatmap de correlación
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(cor_matrix_pearson, dtype=bool))
    sns.heatmap(cor_matrix_pearson, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=.5, cbar_kws={"shrink": .8, "label": "Correlacion"})
    plt.title("Matriz de Correlacion entre Activos", fontsize=16, weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    img_corr_path = os.path.join(CARPETA_GRAFICOS_TEMP, "heatmap_correlacion.png")
    plt.savefig(img_corr_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Frontera eficiente CON LOS TRES PORTAFOLIOS
    ef_plot = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(12, 8))
    plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
    
    # Plotear los tres portafolios
    ax.scatter(perf_sharpe[1], perf_sharpe[0], marker="*", s=300, c="red", label="Maxima Sharpe", zorder=3)
    ax.scatter(perf_min_vol[1], perf_min_vol[0], marker="*", s=300, c="green", label="Minima Volatilidad", zorder=3)
    ax.scatter(perf_equal[1], perf_equal[0], marker="*", s=300, c="blue", label="Equal Weight", zorder=3)
    
    # Plotear resultado Monte Carlo
    ax.scatter(optimo_mc.volatilidad, optimo_mc.retorno, marker="D", s=200, c="orange", label="Monte Carlo Optimo", zorder=3)
    
    ax.set_title("Frontera Eficiente y Portafolios Optimos", fontsize=16, weight="bold")
    ax.set_xlabel("Volatilidad Anualizada")
    ax.set_ylabel("Retorno Anualizado")
    ax.legend()
    plt.tight_layout()
    img_ef_path = os.path.join(CARPETA_GRAFICOS_TEMP, "frontera_eficiente_completa.png")
    plt.savefig(img_ef_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Evolución de los tres portafolios
    plt.figure(figsize=(14, 8))
    combined_returns.plot(linewidth=2)
    plt.title("Evolucion Acumulada de los Portafolios vs. Benchmark", fontsize=16, weight="bold")
    plt.xlabel("Fecha")
    plt.ylabel("Retorno Acumulado")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    img_evolution_path = os.path.join(CARPETA_GRAFICOS_TEMP, "evolucion_tres_portafolios.png")
    plt.savefig(img_evolution_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Simulación Monte Carlo
    plt.figure(figsize=(12, 8))
    plt.scatter(carteras_mc.volatilidad, carteras_mc.retorno, c=carteras_mc.sharpe, s=1, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatilidad Anualizada')
    plt.ylabel('Retorno Anualizado')
    plt.title('Simulacion Monte Carlo - Optimizacion de Portafolios')
    
    # Plotear portafolio óptimo de Monte Carlo
    plt.scatter(optimo_mc.volatilidad, optimo_mc.retorno, c='red', s=500, marker='*', label='Optimo Monte Carlo')
    
    # Plotear activos individuales
    for ticker in df_activos.columns:
        vol = datos_activos_mc.loc[ticker, 'volatilidad']
        ret = datos_activos_mc.loc[ticker, 'retorno']
        plt.scatter(vol, ret, c='white', s=200, edgecolors='black', linewidth=2)
        plt.text(vol, ret, ticker, ha='center', va='center', fontweight='bold')
    
    plt.legend()
    plt.tight_layout()
    img_montecarlo_path = os.path.join(CARPETA_GRAFICOS_TEMP, "simulacion_monte_carlo.png")
    plt.savefig(img_montecarlo_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Volatilidad móvil
    plt.figure(figsize=(14, 8))
    for activo in volatilidad_movil.columns:
        plt.plot(volatilidad_movil.index, volatilidad_movil[activo], label=activo, linewidth=1.5)
    plt.title("Volatilidad Movil de 60 dias (Anualizada)", fontsize=16, weight="bold")
    plt.xlabel("Fecha")
    plt.ylabel("Volatilidad")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    img_vol_movil_path = os.path.join(CARPETA_GRAFICOS_TEMP, "volatilidad_movil.png")
    plt.savefig(img_vol_movil_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Dendrograma de clustering
    if not cluster_df.empty:
        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, labels=cor_matrix_pearson.columns, leaf_rotation=45)
        plt.title("Dendrograma de Clustering de Activos", fontsize=16, weight="bold")
        plt.tight_layout()
        dendro_path = os.path.join(CARPETA_GRAFICOS_TEMP, "dendrograma_clustering.png")
        plt.savefig(dendro_path, dpi=300, bbox_inches='tight')
        plt.close()

    # 7. Varianza explicada PCA
    if explained_var.size > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_var)+1), explained_var*100, alpha=0.7, color='steelblue')
        plt.xlabel("Componente Principal")
        plt.ylabel("% de Varianza Explicada")
        plt.title("Varianza Explicada por Analisis de Componentes Principales (PCA)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pca_path = os.path.join(CARPETA_GRAFICOS_TEMP, "pca_varianza.png")
        plt.savefig(pca_path, dpi=300, bbox_inches='tight')
        plt.close()

    # --- GUARDAR RESULTADOS EN EXCEL ---
    print("\n💾 Guardando resultados en Excel...")
    excel_results_path = os.path.join(CARPETA_SALIDA, "analisisCuantitativoDeActivos.xlsx")
    with pd.ExcelWriter(excel_results_path, engine="openpyxl") as writer:
        # Hoja 1: Estadísticas descriptivas
        estadisticas_descriptivas.to_excel(writer, sheet_name="Estadisticas_Descriptivas")
        
        # Hoja 2: Correlaciones
        cor_matrix_pearson.round(4).to_excel(writer, sheet_name="Correlacion_Pearson")
        cor_matrix_spearman.round(4).to_excel(writer, sheet_name="Correlacion_Spearman")
        
        # Hoja 3: PCA
        if explained_var.size > 0:
            pd.DataFrame(explained_var, columns=["Varianza_Explicada"]).to_excel(writer, sheet_name="PCA_Varianza")
            components.T.round(4).to_excel(writer, sheet_name="PCA_Loadings")
        
        # Hoja 4: Betas
        if not betas_df.empty:
            betas_df.to_excel(writer, sheet_name="Betas")
        
        # Hoja 5: Clustering
        if not cluster_df.empty:
            cluster_df.to_excel(writer, sheet_name="Clusters")
        
        # Hoja 6: Portafolios
        portafolios_df = pd.DataFrame({
            'Maxima Sharpe': pd.Series(cleaned_weights_sharpe),
            'Minima Volatilidad': pd.Series(cleaned_weights_min_vol),
            'Equal Weight': pd.Series(weights_equal)
        }).fillna(0)
        portafolios_df.to_excel(writer, sheet_name="Portafolios")
        
        # Hoja 7: Monte Carlo
        datos_activos_mc.to_excel(writer, sheet_name="Monte_Carlo_Activos")
        
        # Hoja 8: Evolución de portafolios
        combined_returns.to_excel(writer, sheet_name="Evolucion_Portafolios")

    print(f"✅ Resultados guardados en {excel_results_path}")

    # --- GENERACIÓN DE INFORME PDF MEJORADO ---
    print("\n📄 Generando informe PDF mejorado...")
    pdf = PDFMejorado()
    
    # Carátula Profesional
    pdf.add_cover_page_profesional()

    # SECCIÓN 1: Resumen Ejecutivo
    pdf.add_page()
    pdf.chapter_title("1. Resumen Ejecutivo")
    resumen_text = f"""Este informe presenta un analisis cuantitativo completo de una cartera compuesta por {len(activos_cartera)} activos: {', '.join(activos_cartera)}. 

El periodo de analisis abarca desde {fecha_inicio_analisis} hasta {fecha_fin_analisis}.

Se han evaluado tres estrategias de portafolio:
1. Maxima Sharpe Ratio
2. Minima Volatilidad  
3. Equal Weight

Ademas de una optimizacion por simulacion Monte Carlo con {len(carteras_mc)} iteraciones."""
    
    pdf.chapter_body(resumen_text)
    
    # SECCIÓN 2: Estadísticas Descriptivas
    pdf.add_page()
    pdf.add_table_mejorada(estadisticas_descriptivas.round(6), "2. Estadisticas Descriptivas por Activo")
    pdf.chapter_body("Las estadisticas muestran el comportamiento historico de los retornos diarios de cada activo. La media indica la tendencia central, la desviacion estandar mide la volatilidad, el skewness la asimetria y la kurtosis la forma de la distribucion.")
    
    # SECCIÓN 3: Coeficientes Beta
    if not betas_df.empty:
        pdf.add_page()
        pdf.add_table_mejorada(betas_df.round(4), f"3. Coeficientes Beta respecto a {benchmark_elegido}")
        pdf.chapter_body(f"Los coeficientes Beta miden la sensibilidad de cada activo respecto al benchmark {benchmark_elegido}. Un Beta mayor a 1 indica mayor volatilidad que el mercado, mientras que menor a 1 indica menor volatilidad.")

    # SECCIÓN 4: Análisis de Correlación
    pdf.add_page()
    pdf.chapter_title("4. Analisis de Correlacion")
    pdf.chapter_body("La matriz de correlacion muestra las relaciones lineales entre los retornos de los activos. Correlaciones altas pueden indicar falta de diversificacion.")
    pdf.add_image_with_caption(img_corr_path, "Figura 4.1: Matriz de Correlacion entre Activos")

    # SECCIÓN 5: Análisis de Componentes Principales (PCA)
    if explained_var.size > 0:
        pdf.add_page()
        pdf.chapter_title("5. Analisis de Componentes Principales (PCA)")
        pdf.chapter_body("El PCA identifica las direcciones de mayor varianza en los datos, ayudando a entender que factores comunes afectan a los activos.")
        pdf.add_image_with_caption(pca_path, "Figura 5.1: Varianza Explicada por PCA")
        pdf.add_table_mejorada(components.T.round(4), "Loadings de PCA por Activo (Primeros 5 componentes)", max_rows=10)

    # SECCIÓN 6: Clustering de Activos
    if not cluster_df.empty:
        pdf.add_page()
        pdf.chapter_title("6. Clustering de Activos")
        pdf.chapter_body("El analisis de clustering agrupa activos con comportamientos similares, util para estrategias de diversificacion.")
        pdf.add_image_with_caption(dendro_path, "Figura 6.1: Dendrograma de Clustering")
        pdf.add_table_mejorada(cluster_df, "Asignacion de Activos a Clusters")

    # SECCIÓN 7: Volatilidad Móvil
    pdf.add_page()
    pdf.chapter_title("7. Analisis de Volatilidad Movil")
    pdf.chapter_body("La volatilidad movil de 60 dias muestra como ha evolucionado el riesgo de cada activo a lo largo del tiempo.")
    pdf.add_image_with_caption(img_vol_movil_path, "Figura 7.1: Volatilidad Movil de 60 dias")

    # SECCIÓN 8: Frontera Eficiente y Optimización
    pdf.add_page()
    pdf.chapter_title("8. Frontera Eficiente y Optimizacion de Portafolios")
    pdf.chapter_body("Se han optimizado tres tipos de portafolios utilizando la Teoria Moderna de Portafolios de Markowitz, ademas de una optimizacion por Monte Carlo.")
    pdf.add_image_with_caption(img_ef_path, "Figura 8.1: Frontera Eficiente con Portafolios Optimos")
    
    # Métricas de optimización
    metricas_opt = {
        "Portafolio Maxima Sharpe": f"Retorno {perf_sharpe[0]:.2%}, Volatilidad {perf_sharpe[1]:.2%}, Sharpe {perf_sharpe[2]:.4f}",
        "Portafolio Minima Volatilidad": f"Retorno {perf_min_vol[0]:.2%}, Volatilidad {perf_min_vol[1]:.2%}, Sharpe {perf_min_vol[2]:.4f}",
        "Portafolio Equal Weight": f"Retorno {perf_equal[0]:.2%}, Volatilidad {perf_equal[1]:.2%}, Sharpe {perf_equal[2]:.4f}",
        "Portafolio Monte Carlo": f"Retorno {optimo_mc.retorno:.2%}, Volatilidad {optimo_mc.volatilidad:.2%}, Sharpe {optimo_mc.sharpe:.4f}"
    }
    pdf.add_metrics_section("Resultados de Optimizacion", metricas_opt)

    # SECCIÓN 9: Simulación Monte Carlo
    pdf.add_page()
    pdf.chapter_title("9. Simulacion Monte Carlo")
    pdf.chapter_body(f"Se realizaron {len(carteras_mc)} simulaciones aleatorias para encontrar portafolios optimos mediante exploracion estocastica del espacio de soluciones.")
    pdf.add_image_with_caption(img_montecarlo_path, "Figura 9.1: Simulacion Monte Carlo - Optimizacion de Portafolios")

    # SECCIÓN 10: Evolución y Comparación de Portafolios
    pdf.add_page()
    pdf.chapter_title("10. Evolucion y Comparacion de Portafolios")
    pdf.chapter_body("Comparacion del rendimiento historico de las tres estrategias de portafolio y el benchmark durante el periodo de analisis.")
    pdf.add_image_with_caption(img_evolution_path, "Figura 10.1: Evolucion Acumulada de los Portafolios")
    
    # Métricas de rendimiento comparativas
    metricas_rendimiento = {}
    for nombre, returns in [("Maxima Sharpe", cumulative_sharpe), ("Minima Volatilidad", cumulative_min_vol), ("Equal Weight", cumulative_equal)]:
        if not returns.empty:
            total_return = (returns.iloc[-1] / returns.iloc[0]) - 1
            annualized_return = (1 + total_return)**(252 / len(returns)) - 1
            daily_ret = returns.pct_change().dropna()
            ann_vol = daily_ret.std() * np.sqrt(252)
            sharpe = annualized_return / ann_vol if ann_vol != 0 else 0
            metricas_rendimiento[nombre] = f"Retorno Total {total_return:.2%}, Retorno Anualizado {annualized_return:.2%}, Volatilidad {ann_vol:.2%}, Sharpe {sharpe:.4f}"
    
    pdf.add_metrics_section("Metricas de Rendimiento Comparativas", metricas_rendimiento)

    # SECCIÓN 11: Composición de Portafolios
    pdf.add_page()
    pdf.chapter_title("11. Composicion Detallada de Portafolios")
    pdf.chapter_body("Pesos especificos de cada activo en los diferentes portafolios optimizados:")
    pdf.add_table_mejorada(portafolios_df.round(4), "Composicion de Portafolios")

    # SECCIÓN 12: Conclusiones y Recomendaciones
    pdf.add_page()
    pdf.chapter_title("12. Conclusiones y Recomendaciones")
    
    # Determinar el mejor portafolio
    mejor_sharpe = max(perf_sharpe[2], perf_min_vol[2], perf_equal[2], optimo_mc.sharpe)
    if mejor_sharpe == perf_sharpe[2]:
        mejor_portafolio = "Maxima Sharpe"
    elif mejor_sharpe == perf_min_vol[2]:
        mejor_portafolio = "Minima Volatilidad"
    elif mejor_sharpe == perf_equal[2]:
        mejor_portafolio = "Equal Weight"
    else:
        mejor_portafolio = "Monte Carlo"
    
    conclusiones_text = f"""Basandose en el analisis cuantitativo realizado, se destacan las siguientes conclusiones:

- El portafolio con mejor ratio Sharpe es: {mejor_portafolio} ({mejor_sharpe:.4f})

- La correlacion promedio entre activos es: {cor_matrix_pearson.mean().mean():.4f}

- El primer componente principal explica {explained_var[0]*100:.1f}% de la varianza

- Se recomienda monitorear la volatilidad movil para ajustes dinamicos de posiciones

- La diversificacion puede mejorarse considerando los resultados del clustering

- La simulacion Monte Carlo proporciona validacion adicional de los resultados de optimizacion

- Se sugiere rebalanceo periodico basado en las metricas de riesgo calculadas"""
    
    pdf.chapter_body(conclusiones_text)

    # Guardar PDF
    pdf_output_path = os.path.join(CARPETA_SALIDA, f"InformeDeActivos{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    pdf.output(pdf_output_path)
    
    print(f"✅ Informe PDF mejorado generado: {pdf_output_path}")
    print(f"\n🎉 ANALISIS COMPLETADO EXITOSAMENTE")
    print(f"📁 Resultados disponibles en: {CARPETA_SALIDA}")
    print(f"📊 Excel: {excel_results_path}")
    print(f"📄 PDF Mejorado: {pdf_output_path}")
    print(f"🖼️ Graficos: {CARPETA_GRAFICOS_TEMP}")
