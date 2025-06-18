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

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN GLOBAL ---
CARPETA_SALIDA = "DatosCartera"
CARPETA_GRAFICOS_TEMP = os.path.join(CARPETA_SALIDA, "temp_graficos")
CARPETA_DATOS_LOCAL = "pythondatos"  # Carpeta donde están los datos locales
BENCHMARK_DEFAULT = "SPY"

# Asegurar que las carpetas existan
os.makedirs(CARPETA_SALIDA, exist_ok=True)
os.makedirs(CARPETA_GRAFICOS_TEMP, exist_ok=True)

# --- FUNCIONES DE UTILIDAD ---
def cargar_datos_locales(tickers, benchmark=None):
    """
    Carga datos de precios desde archivos CSV locales en la carpeta pythondatos.
    """
    todos_tickers = list(tickers)
    if benchmark and benchmark not in todos_tickers:
        todos_tickers.append(benchmark)

    df_cargado = pd.DataFrame()
    datos_disponibles = True

    print(f"Cargando datos desde la carpeta: {CARPETA_DATOS_LOCAL}")
    
    for ticker in todos_tickers:
        filepath = os.path.join(CARPETA_DATOS_LOCAL, f"{ticker}.csv")
        if os.path.exists(filepath):
            try:
                df_temp = pd.read_csv(filepath, index_col=0, parse_dates=True)
                # Asegurar que tenemos una columna de precios válida
                if df_temp.shape[1] == 1:
                    df_temp.columns = [ticker]
                elif 'Close' in df_temp.columns:
                    df_temp = df_temp[['Close']].rename(columns={'Close': ticker})
                elif 'Adj Close' in df_temp.columns:
                    df_temp = df_temp[['Adj Close']].rename(columns={'Adj Close': ticker})
                else:
                    # Tomar la primera columna numérica
                    numeric_cols = df_temp.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        df_temp = df_temp[[numeric_cols[0]]].rename(columns={numeric_cols[0]: ticker})
                    else:
                        print(f"⚠️ No se encontraron columnas numéricas en {filepath}")
                        datos_disponibles = False
                        break
                
                df_cargado = pd.concat([df_cargado, df_temp], axis=1)
                print(f"✅ {ticker} cargado correctamente desde {filepath}")
            except Exception as e:
                print(f"⚠️ Error al cargar {filepath}: {e}")
                datos_disponibles = False
                break
        else:
            print(f"❌ Archivo no encontrado: {filepath}")
            datos_disponibles = False
            break

    if datos_disponibles and not df_cargado.empty:
        # Limpiar datos
        df_cargado = df_cargado.dropna()
        print(f"✅ Datos cargados correctamente. Shape: {df_cargado.shape}")
        print(f"Período: {df_cargado.index[0]} a {df_cargado.index[-1]}")
        return df_cargado
    else:
        print("❌ No se pudieron cargar todos los datos desde la carpeta local.")
        return pd.DataFrame()

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
        reemplazos = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ñ': 'n', 'ü': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'Ñ': 'N', 'Ü': 'U',
            '°': 'grados', '²': '2', '³': '3',
            'β': 'beta', 'α': 'alfa', 'σ': 'sigma'
        }
        for caracter, reemplazo in reemplazos.items():
            texto = texto.replace(caracter, reemplazo)
        
        # Eliminar caracteres no ASCII
        texto = ''.join(char if ord(char) < 128 else '?' for char in texto)
    
    return str(texto)

# --- CLASE PDF PERSONALIZADA ---
class PDFInformeActivos(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        self.set_font('Arial', 'B', 15)
        titulo = limpiar_texto('Informe de Analisis Cuantitativo de Cartera')
        self.cell(0, 10, titulo, 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        footer_text = limpiar_texto(f'Pagina {self.page_no()} - Generado por MiniMax Agent - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        self.cell(0, 10, footer_text, 0, 0, 'C')
        
        # Línea decorativa
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

    def chapter_title(self, title):
        title = limpiar_texto(title)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        body = limpiar_texto(body)
        self.set_font('Arial', '', 11)
        
        # Envolver el texto
        wrapped_text = textwrap.fill(body, width=90)
        for line in wrapped_text.split('\n'):
            self.cell(0, 6, line, 0, 1)
        self.ln()

    def add_metrics_section(self, title, metrics_dict):
        self.chapter_title(title)
        for key, value in metrics_dict.items():
            key = limpiar_texto(key)
            value = limpiar_texto(value)
            self.set_font('Arial', 'B', 10)
            self.cell(60, 8, f"{key}:", 0, 0, 'L')
            self.set_font('Arial', '', 10)
            self.cell(0, 8, value, 0, 1, 'L')
        self.ln(5)

if __name__ == "__main__":
    # Configuración de activos
    activos_cartera = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    benchmark_elegido = BENCHMARK_DEFAULT

    print("🚀 Iniciando análisis cuantitativo completo de carteras...")
    print(f"📁 Leyendo datos desde la carpeta: {CARPETA_DATOS_LOCAL}")
    
    # Cargar datos desde carpeta local
    df_precios = cargar_datos_locales(activos_cartera, benchmark_elegido)
    
    if df_precios.empty:
        print("❌ No se pudieron obtener datos para el análisis.")
        print("Asegúrate de que la carpeta 'pythondatos' existe y contiene los archivos CSV:")
        for ticker in activos_cartera + [benchmark_elegido]:
            print(f"  - {ticker}.csv")
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

    # 9. Optimización de Markowitz
    print("\n📊 Calculando portafolios óptimos...")
    try:
        # Calcular retornos y covarianza anualizados
        mu = expected_returns.mean_historical_return(df_activos)
        S = risk_models.sample_cov(df_activos)
        
        # Máxima Sharpe
        ef_sharpe = EfficientFrontier(mu, S)
        weights_sharpe = ef_sharpe.max_sharpe()
        ef_sharpe.portfolio_performance()
        perf_sharpe = ef_sharpe.portfolio_performance()
        
        # Mínima volatilidad
        ef_min_vol = EfficientFrontier(mu, S)
        weights_min_vol = ef_min_vol.min_volatility()
        perf_min_vol = ef_min_vol.portfolio_performance()
        
        # Equal weight
        n_assets = len(mu)
        weights_equal = dict(zip(mu.index, [1/n_assets] * n_assets))
        ef_equal = EfficientFrontier(mu, S)
        ef_equal.set_weights(weights_equal)
        perf_equal = ef_equal.portfolio_performance()
        
        print("✅ Optimización de portafolios completada")
        
    except Exception as e:
        print(f"⚠️ Error en optimización: {e}")
        weights_sharpe = weights_min_vol = weights_equal = {}
        perf_sharpe = perf_min_vol = perf_equal = (0, 0, 0)

    # --- GENERAR GRÁFICOS ---
    print("\n📊 Generando gráficos...")
    
    # 1. Matriz de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(cor_matrix_pearson, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Matriz de Correlación de Retornos')
    plt.tight_layout()
    img_corr_path = os.path.join(CARPETA_GRAFICOS_TEMP, "correlacion_matrix.png")
    plt.savefig(img_corr_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Volatilidad móvil
    plt.figure(figsize=(12, 8))
    for col in volatilidad_movil.columns:
        plt.plot(volatilidad_movil.index, volatilidad_movil[col], label=col, linewidth=1.5)
    plt.title('Volatilidad Móvil (60 días) - Anualizada')
    plt.xlabel('Fecha')
    plt.ylabel('Volatilidad')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    img_vol_movil_path = os.path.join(CARPETA_GRAFICOS_TEMP, "volatilidad_movil.png")
    plt.savefig(img_vol_movil_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. PCA
    if explained_var.size > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, min(11, len(explained_var)+1)), explained_var[:10], alpha=0.8)
        plt.xlabel('Componente Principal')
        plt.ylabel('Varianza Explicada')
        plt.title('Varianza Explicada por Componentes Principales')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pca_path = os.path.join(CARPETA_GRAFICOS_TEMP, "pca_variance.png")
        plt.savefig(pca_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        pca_path = ""
    
    # 4. Dendrograma de clustering
    if not cluster_df.empty:
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, labels=cor_matrix_pearson.columns, leaf_rotation=90)
        plt.title('Dendrograma de Clustering de Activos')
        plt.ylabel('Distancia')
        plt.tight_layout()
        dendro_path = os.path.join(CARPETA_GRAFICOS_TEMP, "dendrogram.png")
        plt.savefig(dendro_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        dendro_path = ""
    
    # 5. Simulación Monte Carlo
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(carteras_mc['volatilidad'], carteras_mc['retorno'], 
                         c=carteras_mc['sharpe'], cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    # Marcar el mejor portafolio
    plt.scatter(optimo_mc.volatilidad, optimo_mc.retorno, 
               color='red', s=100, marker='*', label=f'Óptimo MC (Sharpe: {optimo_mc.sharpe:.3f})')
    
    plt.xlabel('Volatilidad')
    plt.ylabel('Retorno')
    plt.title('Simulación Monte Carlo - Optimización de Portafolios')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    img_montecarlo_path = os.path.join(CARPETA_GRAFICOS_TEMP, "monte_carlo.png")
    plt.savefig(img_montecarlo_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráficos generados")

    # --- GUARDAR RESULTADOS EN EXCEL ---
    print("\n💾 Guardando resultados en Excel...")
    excel_results_path = os.path.join(CARPETA_SALIDA, f"ResultadosAnalisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    
    with pd.ExcelWriter(excel_results_path) as writer:
        estadisticas_descriptivas.to_excel(writer, sheet_name='Estadisticas')
        cor_matrix_pearson.to_excel(writer, sheet_name='Correlacion_Pearson')
        if not betas_df.empty:
            betas_df.to_excel(writer, sheet_name='Betas')
        if not cluster_df.empty:
            cluster_df.to_excel(writer, sheet_name='Clusters')
        if explained_var.size > 0:
            components.head().to_excel(writer, sheet_name='PCA_Components')
        datos_activos_mc.to_excel(writer, sheet_name='Monte_Carlo_Activos')
        
        # Composición de portafolios
        portafolios_data = {}
        if weights_sharpe:
            portafolios_data['Maxima_Sharpe'] = weights_sharpe
        if weights_min_vol:
            portafolios_data['Min_Volatilidad'] = weights_min_vol
        if weights_equal:
            portafolios_data['Equal_Weight'] = weights_equal
            
        if portafolios_data:
            portafolios_df = pd.DataFrame(portafolios_data).fillna(0)
            portafolios_df.to_excel(writer, sheet_name='Portafolios')
    
    print(f"✅ Resultados guardados en: {excel_results_path}")

    # --- GENERAR INFORME PDF ---
    print("\n📄 Generando informe PDF...")
    
    pdf = PDFInformeActivos()
    pdf.add_page()
    
    # Resumen ejecutivo
    resumen_text = f"""Este informe presenta un analisis cuantitativo completo de la cartera de activos.
Los activos analizados son: {', '.join(df_activos.columns)}
Periodo de analisis: {df_precios.index[0].strftime('%Y-%m-%d')} a {df_precios.index[-1].strftime('%Y-%m-%d')}

Se han evaluado tres estrategias de portafolio:
1. Maxima Sharpe Ratio
2. Minima Volatilidad  
3. Equal Weight

Ademas de una optimizacion por simulacion Monte Carlo con {len(carteras_mc)} iteraciones."""
    
    pdf.chapter_body(resumen_text)
    
    # Estadísticas descriptivas
    pdf.add_page()
    pdf.add_table_mejorada(estadisticas_descriptivas.round(6), "Estadisticas Descriptivas por Activo")
    
    # Coeficientes Beta
    if not betas_df.empty:
        pdf.add_page()
        pdf.add_table_mejorada(betas_df.round(4), f"Coeficientes Beta respecto a {benchmark_elegido}")
    
    # Análisis de correlación
    pdf.add_page()
    pdf.chapter_title("Analisis de Correlacion")
    pdf.add_image_with_caption(img_corr_path, "Matriz de Correlacion entre Activos")
    
    # PCA
    if explained_var.size > 0:
        pdf.add_page()
        pdf.chapter_title("Analisis de Componentes Principales (PCA)")
        pdf.add_image_with_caption(pca_path, "Varianza Explicada por PCA")
    
    # Clustering
    if not cluster_df.empty:
        pdf.add_page()
        pdf.chapter_title("Clustering de Activos")
        pdf.add_image_with_caption(dendro_path, "Dendrograma de Clustering")
        pdf.add_table_mejorada(cluster_df, "Asignacion de Activos a Clusters")
    
    # Volatilidad móvil
    pdf.add_page()
    pdf.chapter_title("Analisis de Volatilidad Movil")
    pdf.add_image_with_caption(img_vol_movil_path, "Volatilidad Movil de 60 dias")
    
    # Monte Carlo
    pdf.add_page()
    pdf.chapter_title("Simulacion Monte Carlo")
    pdf.add_image_with_caption(img_montecarlo_path, "Simulacion Monte Carlo - Optimizacion de Portafolios")
    
    # Conclusiones
    pdf.add_page()
    pdf.chapter_title("Conclusiones")
    
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
    
    conclusiones_text = f"""Basandose en el analisis cuantitativo realizado:

- El portafolio con mejor ratio Sharpe es: {mejor_portafolio} ({mejor_sharpe:.4f})
- La correlacion promedio entre activos es: {cor_matrix_pearson.mean().mean():.4f}
- El primer componente principal explica {explained_var[0]*100:.1f}% de la varianza
- Se recomienda monitorear la volatilidad movil para ajustes dinamicos
- La diversificacion puede mejorarse considerando los resultados del clustering
- La simulacion Monte Carlo proporciona validacion adicional de los resultados"""
    
    pdf.chapter_body(conclusiones_text)

    # Guardar PDF
    pdf_output_path = os.path.join(CARPETA_SALIDA, f"InformeDeActivos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    pdf.output(pdf_output_path)
    
    print(f"✅ Informe PDF generado: {pdf_output_path}")
    print(f"\n🎉 ANALISIS COMPLETADO EXITOSAMENTE")
    print(f"📁 Resultados disponibles en: {CARPETA_SALIDA}")
    print(f"📊 Excel: {excel_results_path}")
    print(f"📄 PDF: {pdf_output_path}")
    print(f"🖼️ Graficos: {CARPETA_GRAFICOS_TEMP}")
