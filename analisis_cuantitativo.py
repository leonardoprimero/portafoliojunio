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
def descargar_datos(tickers, start_date, end_date, benchmark=None):
    """
    Descarga datos históricos de precios de Yahoo Finance.
    """
    todos_tickers = list(tickers)
    if benchmark and benchmark not in todos_tickers:
        todos_tickers.append(benchmark)

    print(f"Descargando datos para: {todos_tickers} desde {start_date} hasta {end_date}")
    data = yf.download(todos_tickers, start=start_date, end=end_date)
    
    if data.empty:
        print("❌ No se pudieron descargar datos. Verifique los tickers y el rango de fechas.")
        return pd.DataFrame()

    print(f"Columnas de datos descargados: {data.columns}")
    print(f"Primeras 5 filas de datos descargados:\n{data.head()}")
    if "Adj Close" in data.columns:
        df_precios = data["Adj Close"].dropna(how="all")
    elif "Close" in data.columns:
        df_precios = data["Close"].dropna(how="all")
    else:
        print("❌ No se encontró la columna 'Adj Close' ni 'Close' en los datos descargados.")
        return pd.DataFrame()
    
    # Guardar datos descargados en caché
    for ticker in df_precios.columns:
        filepath = os.path.join(CARPETA_DATOS_CACHE, f"{ticker}.csv")
        df_precios[[ticker]].to_csv(filepath)
        print(f"✅ Datos de {ticker} guardados en caché: {filepath}")

    return df_precios

def cargar_datos_locales(tickers, benchmark=None):
    """
    Intenta cargar datos de precios desde archivos CSV locales.
    """
    todos_tickers = list(tickers)
    if benchmark and benchmark not in todos_tickers:
        todos_tickers.append(benchmark)

    df_cargado = pd.DataFrame()
    datos_disponibles = True

    for ticker in todos_tickers:
        filepath = os.path.join(CARPETA_DATOS_CACHE, f"{ticker}.csv")
        if os.path.exists(filepath):
            try:
                df_temp = pd.read_csv(filepath, index_col=0, parse_dates=True)
                df_cargado = pd.concat([df_cargado, df_temp], axis=1)
            except Exception as e:
                print(f"⚠️ Error al cargar {filepath}: {e}")
                datos_disponibles = False
                break
        else:
            print(f"Archivo no encontrado: {filepath}")
            datos_disponibles = False
            break

    if datos_disponibles and not df_cargado.empty:
        print("✅ Datos cargados desde caché local.")
        return df_cargado
    else:
        print("No se pudieron cargar todos los datos desde caché local. Se procederá a descargar.")
        return None

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

# --- CLASE PDF PERSONALIZADA ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Informe de Análisis Cuantitativo de Carteras', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 8, body)
        self.ln()

    def add_cover_page(self, title, author, date):
        self.add_page()
        self.image('/home/ubuntu/upload/image.png', x=self.w/4, y=self.h/4, w=self.w/2)
        self.set_font('Arial', 'B', 24)
        self.ln(self.h/2)
        self.cell(0, 10, title, 0, 1, 'C')
        self.set_font('Arial', '', 16)
        self.cell(0, 10, author, 0, 1, 'C')
        self.cell(0, 10, date, 0, 1, 'C')

    def add_image_with_caption(self, image_path, caption):
        if os.path.exists(image_path):
            self.image(image_path, x=self.l_margin, w=self.w - 2 * self.l_margin)
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, caption, 0, 1, 'C')
            self.ln(5)
        else:
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f"[Imagen no encontrada: {caption}]", 0, 1, 'C')
            self.ln(5)


if __name__ == "__main__":
    # Configuración de activos y fechas (ejemplo, el usuario podrá modificar esto)
    activos_cartera = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    fecha_inicio_analisis = "2020-01-01"
    fecha_fin_analisis = "2024-12-31"
    benchmark_elegido = BENCHMARK_DEFAULT # Se puede cambiar a "GLD", "QQQ", etc.

    # Intentar cargar datos locales, si no existen, descargar
    df_precios = cargar_datos_locales(activos_cartera, benchmark_elegido)
    if df_precios is None or df_precios.empty:
        df_precios = descargar_datos(activos_cartera, fecha_inicio_analisis, fecha_fin_analisis, benchmark_elegido)
    
    if df_precios.empty:
        print("❌ No se pudieron obtener datos para el análisis. Saliendo.")
        exit()

    print(f"\nDatos de precios cargados (primeras 5 filas):\n{df_precios.head()}")
    print(f"Datos de precios cargados (últimas 5 filas):\n{df_precios.tail()}")
    print(f"Columnas disponibles: {df_precios.columns.tolist()}")
    
    # Separar el benchmark si está presente en las columnas de activos
    if benchmark_elegido in df_precios.columns:
        df_benchmark = df_precios[[benchmark_elegido]].copy()
        df_activos = df_precios.drop(columns=[benchmark_elegido]).copy()
    else:
        print(f"⚠️ El benchmark {benchmark_elegido} no se encontró en los datos de precios. No se realizará comparación con benchmark.")
        df_benchmark = None
        df_activos = df_precios.copy()

    if df_activos.empty:
        print("❌ No hay activos para analizar después de separar el benchmark. Saliendo.")
        exit()

    # --- CÁLCULO DE RETORNOS ---
    retornos_diarios = np.log(df_activos / df_activos.shift(1)).dropna()
    print(f"\nRetornos diarios calculados (primeras 5 filas):\n{retornos_diarios.head()}")

    # --- ANÁLISIS DE CORRELACIÓN, PCA, BETAS Y CLUSTERING (adaptado de informe_quant_correlation.py) ---
    print("\n--- Realizando análisis de correlación, PCA, Betas y Clustering ---")
    cor_matrix_pearson = retornos_diarios.corr()
    cor_matrix_spearman = retornos_diarios.corr(method="spearman")
    
    # PCA
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
    else:
        explained_var = np.array([])
        components = pd.DataFrame()
        print("❗️ATENCIÓN: No hay datos suficientes para hacer PCA.")

    # Betas
    betas = {}
    if df_benchmark is not None and not df_benchmark.empty:
        retornos_benchmark_diarios = np.log(df_benchmark / df_benchmark.shift(1)).dropna()
        retornos_combinados = pd.concat([retornos_diarios, retornos_benchmark_diarios], axis=1).dropna()
        for activo in retornos_diarios.columns:
            betas[activo] = calcular_beta(activo, benchmark_elegido, retornos_combinados)
    else:
        print("⚠️ No se puede calcular Betas: Benchmark no disponible.")

    # Clustering
    if not cor_matrix_pearson.empty and cor_matrix_pearson.shape[0] > 1:
        linkage_matrix = linkage(1 - cor_matrix_pearson.abs(), method="ward")
        cluster_labels = fcluster(linkage_matrix, t=0.5, criterion="distance") # Usar "distance" y un umbral
        cluster_df = pd.DataFrame({"Activo": cor_matrix_pearson.columns, "Cluster": cluster_labels})
    else:
        cluster_df = pd.DataFrame()
        print("❗️ATENCIÓN: No hay datos suficientes para hacer Clustering.")

    # Guardar matrices de correlación y otros resultados en Excel
    excel_results_path = os.path.join(CARPETA_SALIDA, "analisis_cuantitativo_resultados.xlsx")
    with pd.ExcelWriter(excel_results_path, engine="openpyxl") as writer:
        cor_matrix_pearson.round(4).to_excel(writer, sheet_name="Correlacion_Pearson")
        cor_matrix_spearman.round(4).to_excel(writer, sheet_name="Correlacion_Spearman")
        if explained_var.size > 0:
            pd.DataFrame(explained_var, columns=["Varianza Explicada"]).to_excel(writer, sheet_name="PCA_Varianza")
            components.T.round(4).to_excel(writer, sheet_name="PCA_Loadings")
        if betas:
            pd.DataFrame(betas.items(), columns=["Activo", f"Beta_{benchmark_elegido}"]).to_excel(writer, sheet_name="Betas")
        if not cluster_df.empty:
            cluster_df.to_excel(writer, sheet_name="Clusters", index=False)
    print(f"Resultados de correlación, PCA, Betas y Clustering guardados en {excel_results_path}")
    
    # Generar Heatmap de correlación
    plt.figure(figsize=(12, 10))
    sns.set(style="white", font_scale=1.1)
    mask = np.triu(np.ones_like(cor_matrix_pearson, dtype=bool))
    sns.heatmap(cor_matrix_pearson, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=.5, cbar_kws={"shrink": .8, "label": "Correlación"})
    plt.title("Matriz de Correlación entre Activos", fontsize=16, weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    img_corr_path = os.path.join(CARPETA_GRAFICOS_TEMP, "heatmap_correlacion.png")
    plt.savefig(img_corr_path)
    plt.close()
    print(f"Heatmap de correlación guardado en {img_corr_path}")

    # Dendrograma de clustering
    if not cluster_df.empty:
        plt.figure(figsize=(12, 5))
        dendrogram(linkage_matrix, labels=cor_matrix_pearson.columns, leaf_rotation=45)
        plt.title("Dendrograma de Clustering de Activos")
        plt.tight_layout()
        dendro_path = os.path.join(CARPETA_GRAFICOS_TEMP, "dendrograma_clustering.png")
        plt.savefig(dendro_path)
        plt.close()
        print(f"Dendrograma de clustering guardado en {dendro_path}")

    # Varianza explicada (PCA)
    if explained_var.size > 0:
        plt.figure(figsize=(8, 4))
        plt.bar(range(1, len(explained_var)+1), explained_var*100)
        plt.xlabel("Componente Principal")
        plt.ylabel("% de Varianza Explicada")
        plt.title("Varianza Explicada por PCA")
        plt.tight_layout()
        pca_path = os.path.join(CARPETA_GRAFICOS_TEMP, "pca_varianza.png")
        plt.savefig(pca_path)
        plt.close()
        print(f"Gráfico de varianza explicada por PCA guardado en {pca_path}")

    # --- FRONTERA EFICIENTE Y PORTAFOLIO DE MARKOWITS ---
    print("\n--- Calculando Frontera Eficiente y Portafolio de Markowitz ---")
    
    # Calcular retornos esperados y covarianza de la muestra
    mu = expected_returns.mean_historical_return(df_activos)
    S = risk_models.sample_cov(df_activos)
    
    # Optimizar para el portafolio de máxima Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights_sharpe = ef.max_sharpe()
    cleaned_weights_sharpe = ef.clean_weights()
    print("Portafolio de Máxima Sharpe Ratio:", cleaned_weights_sharpe)
    perf_sharpe = ef.portfolio_performance(verbose=True)
    
    # Optimizar para el portafolio de mínima volatilidad
    ef = EfficientFrontier(mu, S)
    weights_min_vol = ef.min_volatility()
    cleaned_weights_min_vol = ef.clean_weights()
    print("Portafolio de Mínima Volatilidad:", cleaned_weights_min_vol)
    perf_min_vol = ef.portfolio_performance(verbose=True)
    
    # Calcular la frontera eficiente
    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(10, 6))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
    
    # Plotear los portafolios óptimos
    ax.scatter(perf_sharpe[1], perf_sharpe[0], marker="*", s=200, c="r", label="Máxima Sharpe")
    ax.scatter(perf_min_vol[1], perf_min_vol[0], marker="*", s=200, c="g", label="Mínima Volatilidad")
    
    ax.set_title("Frontera Eficiente y Portafolios Óptimos")
    ax.set_xlabel("Volatilidad Anualizada")
    ax.set_ylabel("Retorno Anualizado")
    ax.legend()
    plt.tight_layout()
    img_ef_path = os.path.join(CARPETA_GRAFICOS_TEMP, "frontera_eficiente.png")
    plt.savefig(img_ef_path)
    plt.close()
    print(f"Frontera eficiente guardada en {img_ef_path}")

    # --- EVOLUCIÓN DEL PORTAFOLIO Y COMPARACIÓN CON BENCHMARK ---
    print("\n--- Calculando Evolución del Portafolio y Comparación con Benchmark ---")
    
    # Calcular retornos diarios del portafolio de máxima Sharpe
    portfolio_returns = (retornos_diarios * pd.Series(cleaned_weights_sharpe)).sum(axis=1)
    cumulative_portfolio_returns = (1 + portfolio_returns).cumprod()
    
    if df_benchmark is not None and not df_benchmark.empty:
        # Calcular retornos diarios del benchmark
        benchmark_daily_returns = np.log(df_benchmark / df_benchmark.shift(1)).dropna()
        cumulative_benchmark_returns = (1 + benchmark_daily_returns).cumprod()
        
        # Asegurar que los índices coincidan para la graficación
        combined_returns = pd.DataFrame({
            "Portafolio (Máxima Sharpe)": cumulative_portfolio_returns,
            f"Benchmark ({benchmark_elegido})": cumulative_benchmark_returns.iloc[:, 0] # Tomar la primera columna del benchmark
        }).dropna()
        
        plt.figure(figsize=(12, 6))
        combined_returns.plot(ax=plt.gca())
        plt.title("Evolución Acumulada del Portafolio vs. Benchmark")
        plt.xlabel("Fecha")
        plt.ylabel("Retorno Acumulado")
        plt.grid(True)
        plt.tight_layout()
        img_evolution_path = os.path.join(CARPETA_GRAFICOS_TEMP, "evolucion_portafolio_benchmark.png")
        plt.savefig(img_evolution_path)
        print(f"Gráfico de evolución del portafolio vs. benchmark guardado en {img_evolution_path}")
    else:
        print("No se puede graficar la evolución del portafolio vs. benchmark: Benchmark no disponible.")
        
    # --- BACKTESTING ---
    print("\n--- Realizando Backtesting ---")
    # Para un backtesting más completo, se necesitaría una estrategia de rebalanceo y un motor de backtesting.
    # Aquí se presenta un ejemplo simplificado de cálculo de métricas de rendimiento.
    
    # Asumiendo que "cumulative_portfolio_returns" es el valor de la cartera a lo largo del tiempo
    # y "cumulative_benchmark_returns" es el valor del benchmark.
    
    if not cumulative_portfolio_returns.empty:
        # Calcular métricas básicas de rendimiento
        total_return_portfolio = (cumulative_portfolio_returns.iloc[-1] / cumulative_portfolio_returns.iloc[0]) - 1
        annualized_return_portfolio = (1 + total_return_portfolio)**(252 / len(cumulative_portfolio_returns)) - 1 # Asumiendo 252 días de trading
        
        daily_returns_portfolio = cumulative_portfolio_returns.pct_change().dropna()
        annualized_volatility_portfolio = daily_returns_portfolio.std() * np.sqrt(252)
        
        print(f"\nMétricas de Rendimiento del Portafolio (Máxima Sharpe):\n  Retorno Total: {total_return_portfolio:.2%}\n  Retorno Anualizado: {annualized_return_portfolio:.2%}\n  Volatilidad Anualizada: {annualized_volatility_portfolio:.2%}")
        
        if df_benchmark is not None and not df_benchmark.empty:
            total_return_benchmark = (cumulative_benchmark_returns.iloc[-1].item() / cumulative_benchmark_returns.iloc[0].item()) - 1
            annualized_return_benchmark = (1 + total_return_benchmark)**(252 / len(cumulative_benchmark_returns)) - 1
            
            daily_returns_benchmark = cumulative_benchmark_returns.pct_change().dropna()
            annualized_volatility_benchmark = daily_returns_benchmark.std().item() * np.sqrt(252)
            
            print(f"\nMétricas de Rendimiento del Benchmark ({benchmark_elegido}):\n  Retorno Total: {total_return_benchmark:.2%}\n  Retorno Anualizado: {annualized_return_benchmark:.2%}\n  Volatilidad Anualizada: {annualized_volatility_benchmark:.2%}")
            
            # Calcular Sharpe Ratio (asumiendo tasa libre de riesgo = 0 por simplicidad)
            sharpe_ratio_portfolio = annualized_return_portfolio / annualized_volatility_portfolio
            sharpe_ratio_benchmark = annualized_return_benchmark / annualized_volatility_benchmark
            
            print(f"\nSharpe Ratio del Portafolio: {sharpe_ratio_portfolio:.2f}")
            print(f"Sharpe Ratio del Benchmark: {sharpe_ratio_benchmark:.2f}")
            
            # Gráfico de Drawdowns (simplificado)
            # Calcular drawdowns para el portafolio
            roll_max = cumulative_portfolio_returns.expanding(min_periods=1).max()
            daily_drawdown = cumulative_portfolio_returns / roll_max - 1.0
            max_daily_drawdown = daily_drawdown.min()
            
            plt.figure(figsize=(12, 6))
            daily_drawdown.plot(color="red", label="Portafolio Drawdown")
            plt.title("Drawdown Diario del Portafolio")
            plt.xlabel("Fecha")
            plt.ylabel("Drawdown")
            plt.grid(True)
            plt.tight_layout()
            img_drawdown_path = os.path.join(CARPETA_GRAFICOS_TEMP, "drawdown_portafolio.png")
            plt.savefig(img_drawdown_path)
            plt.close()
            print(f"Gráfico de Drawdown guardado en {img_drawdown_path}")
            print(f"Máximo Drawdown del Portafolio: {max_daily_drawdown:.2%}")

    # --- ANÁLISIS INDIVIDUAL DE ACTIVOS (adaptado de analisiscompleto.py) ---
    print("\n--- Realizando análisis individual de activos ---")
    for ticker in df_activos.columns:
        df_activo = df_precios[[ticker]].copy()
        df_activo.columns = ["Precio"] # Renombrar para consistencia con analisiscompleto.py
        
        df_activo["Retorno"] = df_activo["Precio"].pct_change()
        df_activo["SMA_21"] = df_activo["Precio"].rolling(window=21).mean()
        df_activo["SMA_63"] = df_activo["Precio"].rolling(window=63).mean()
        df_activo["SMA_252"] = df_activo["Precio"].rolling(window=252).mean()

        # Gráfico de precios
        fig, ax = plt.subplots(figsize=(10, 5))
        df_activo["Precio"].plot(ax=ax, label="Precio", linewidth=1.2)
        df_activo["SMA_21"].plot(ax=ax, label="SMA 21d")
        df_activo["SMA_63"].plot(ax=ax, label="SMA 63d")
        df_activo["SMA_252"].plot(ax=ax, label="SMA 252d")
        ax.set_title(f"{ticker} - Precio con Medias Móviles")
        ax.set_ylabel("Precio de cierre (USD)")
        ax.set_xlabel("Fecha")
        ax.legend()
        fig.tight_layout()
        img_precio_path = os.path.join(CARPETA_GRAFICOS_TEMP, f"{ticker}_precio.png")
        fig.savefig(img_precio_path)
        plt.close(fig)
        print(f"Gráfico de precios para {ticker} guardado en {img_precio_path}")

        # Retornos diarios
        fig, ax = plt.subplots(figsize=(10, 4))
        df_activo["Retorno"].plot(ax=ax, color="orange")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{ticker} - Retornos Diarios")
        ax.set_ylabel("Retorno")
        ax.set_xlabel("Fecha")
        fig.tight_layout()
        img_retorno_path = os.path.join(CARPETA_GRAFICOS_TEMP, f"{ticker}_retorno.png")
        fig.savefig(img_retorno_path)
        plt.close(fig)
        print(f"Gráfico de retornos para {ticker} guardado en {img_retorno_path}")

        # Histograma
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_activo["Retorno"].dropna(), bins=50, kde=True, ax=ax, color="teal")
        mu = df_activo["Retorno"].mean()
        sigma = df_activo["Retorno"].std()
        ax.axvline(mu, color="red", linestyle="--", label=f"Media: {mu:.4f}")
        ax.axvline(mu + sigma, color="green", linestyle="--", label=f"+1σ: {mu + sigma:.4f}")
        ax.axvline(mu - sigma, color="green", linestyle="--", label=f"-1σ: {mu - sigma:.4f}")
        ax.set_title(f"{ticker} - Histograma de Retornos")
        ax.set_xlabel("Retorno diario")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        fig.tight_layout()
        img_hist_path = os.path.join(CARPETA_GRAFICOS_TEMP, f"{ticker}_histograma.png")
        fig.savefig(img_hist_path)
        plt.close(fig)
        print(f"Histograma de retornos para {ticker} guardado en {img_hist_path}")

    # --- CARTERAS ESPECÍFICAS ---
    print("\n--- Calculando Carteras Específicas ---")
    
    # Cartera 1: Máxima Sharpe (ya calculada)
    # Cartera 2: Mínima Volatilidad (ya calculada)
    
    # Cartera 3: Equal-Weighted (Xweight simplificado)
    num_assets = len(df_activos.columns)
    weights_equal = {asset: 1/num_assets for asset in df_activos.columns}
    print("Portafolio Equal-Weighted (Xweight simplificado):", weights_equal)
    
    # Suma producto de las carteras (ejemplo: combinación de Máxima Sharpe y Mínima Volatilidad)
    # Esto es un ejemplo, la lógica real dependerá de cómo se quiera combinar
    combined_weights = {
        asset: (cleaned_weights_sharpe.get(asset, 0) * 0.5) + (cleaned_weights_min_vol.get(asset, 0) * 0.5)
        for asset in df_activos.columns
    }
    print("Suma Producto de Carteras (50% Sharpe, 50% Min Vol):", combined_weights)

    # --- GENERACIÓN DE INFORME PDF ---
    print("\n--- Generando Informe PDF ---")
    pdf = PDF()
    pdf.alias_nb_pages()
    
    # Carátula
    pdf.add_cover_page("Informe de Análisis Cuantitativo de Carteras", "Quant Analyst", datetime.now().strftime("%Y-%m-%d"))

    # Sección de Correlación
    pdf.add_page()
    pdf.chapter_title("1. Análisis de Correlación")
    pdf.chapter_body("Esta sección presenta la matriz de correlación entre los activos de la cartera, mostrando la relación lineal entre sus retornos diarios. Una correlación cercana a 1 indica que los activos se mueven en la misma dirección, mientras que una cercana a -1 indica que se mueven en direcciones opuestas. Una correlación cercana a 0 sugiere poca o ninguna relación lineal.")
    pdf.add_image_with_caption(img_corr_path, "Figura 1: Heatmap de Correlación entre Activos")
    pdf.chapter_body(f"Las matrices de correlación de Pearson y Spearman se han guardado en el archivo Excel: {excel_results_path}")

    # Sección de Frontera Eficiente y Markowitz
    pdf.add_page()
    pdf.chapter_title("2. Frontera Eficiente y Portafolio de Markowitz")
    pdf.chapter_body("La frontera eficiente representa el conjunto de portafolios que ofrecen el mayor retorno esperado para un nivel dado de riesgo, o el menor riesgo para un nivel dado de retorno esperado. Se han identificado dos portafolios clave: el de Máxima Sharpe Ratio (que maximiza el retorno por unidad de riesgo) y el de Mínima Volatilidad (que minimiza el riesgo absoluto).")
    pdf.add_image_with_caption(img_ef_path, "Figura 2: Frontera Eficiente y Portafolios Óptimos")
    pdf.chapter_body(f"Portafolio de Máxima Sharpe Ratio: {cleaned_weights_sharpe}")
    pdf.chapter_body(f"Rendimiento (Anualizado): {perf_sharpe[0]:.2%}, Volatilidad (Anualizada): {perf_sharpe[1]:.2%}, Sharpe Ratio: {perf_sharpe[2]:.2f}")
    pdf.chapter_body(f"Portafolio de Mínima Volatilidad: {cleaned_weights_min_vol}")
    pdf.chapter_body(f"Rendimiento (Anualizado): {perf_min_vol[0]:.2%}, Volatilidad (Anualizada): {perf_min_vol[1]:.2%}, Sharpe Ratio: {perf_min_vol[2]:.2f}")

    # Sección de Evolución del Portafolio vs. Benchmark
    if df_benchmark is not None and not df_benchmark.empty:
        pdf.add_page()
        pdf.chapter_title("3. Evolución del Portafolio vs. Benchmark")
        pdf.chapter_body(f"Este gráfico muestra la evolución acumulada del portafolio de Máxima Sharpe Ratio en comparación con el benchmark ({benchmark_elegido}) durante el período de análisis. Permite visualizar el rendimiento relativo de la cartera.")
        pdf.add_image_with_caption(img_evolution_path, f"Figura 3: Evolución Acumulada del Portafolio vs. Benchmark ({benchmark_elegido})")

    # Sección de Backtesting
    if not cumulative_portfolio_returns.empty:
        pdf.add_page()
        pdf.chapter_title("4. Métricas de Rendimiento y Backtesting")
        pdf.chapter_body("Se presentan las métricas clave de rendimiento para el portafolio y el benchmark, incluyendo el retorno total, retorno anualizado, volatilidad anualizada y el Sharpe Ratio. Además, se incluye un gráfico de drawdown para evaluar la máxima caída desde un pico.")
        pdf.chapter_body(f"\nMétricas de Rendimiento del Portafolio (Máxima Sharpe):\n  Retorno Total: {total_return_portfolio:.2%}\n  Retorno Anualizado: {annualized_return_portfolio:.2%}\n  Volatilidad Anualizada: {annualized_volatility_portfolio:.2%}\n  Sharpe Ratio: {sharpe_ratio_portfolio:.2f}")
        if df_benchmark is not None and not df_benchmark.empty:
            pdf.chapter_body(f"\nMétricas de Rendimiento del Benchmark ({benchmark_elegido}):\n  Retorno Total: {total_return_benchmark:.2%}\n  Retorno Anualizado: {annualized_return_benchmark:.2%}\n  Volatilidad Anualizada: {annualized_volatility_benchmark:.2%}")
            
            # Calcular Sharpe Ratio (asumiendo tasa libre de riesgo = 0 por simplicidad)
            sharpe_ratio_portfolio = annualized_return_portfolio / annualized_volatility_portfolio
            sharpe_ratio_benchmark = annualized_return_benchmark / annualized_volatility_benchmark
            
            pdf.chapter_body(f"\nSharpe Ratio del Portafolio: {sharpe_ratio_portfolio:.2f}")
            pdf.chapter_body(f"Sharpe Ratio del Benchmark: {sharpe_ratio_benchmark:.2f}")
            
        pdf.add_image_with_caption(img_drawdown_path, "Figura 4: Drawdown Diario del Portafolio")
        pdf.chapter_body(f"Máximo Drawdown del Portafolio: {max_daily_drawdown:.2%}")

    # Sección de Análisis Individual de Activos
    pdf.add_page()
    pdf.chapter_title("5. Análisis Individual de Activos")
    pdf.chapter_body("Esta sección proporciona un análisis detallado para cada activo de la cartera, incluyendo gráficos de precios con medias móviles, retornos diarios e histogramas de retornos para comprender su comportamiento individual.")
    for ticker in df_activos.columns:
        img_precio_path = os.path.join(CARPETA_GRAFICOS_TEMP, f"{ticker}_precio.png")
        img_retorno_path = os.path.join(CARPETA_GRAFICOS_TEMP, f"{ticker}_retorno.png")
        img_hist_path = os.path.join(CARPETA_GRAFICOS_TEMP, f"{ticker}_histograma.png")
        
        pdf.add_page()
        pdf.chapter_title(f"5.{df_activos.columns.get_loc(ticker)+1}. Análisis de {ticker}")
        pdf.add_image_with_caption(img_precio_path, f"Figura 5.{df_activos.columns.get_loc(ticker)+1}.1: {ticker} - Precio con Medias Móviles")
        pdf.add_image_with_caption(img_retorno_path, f"Figura 5.{df_activos.columns.get_loc(ticker)+1}.2: {ticker} - Retornos Diarios")
        pdf.add_image_with_caption(img_hist_path, f"Figura 5.{df_activos.columns.get_loc(ticker)+1}.3: {ticker} - Histograma de Retornos")

    # Sección de PCA y Clustering
    if explained_var.size > 0 or not cluster_df.empty:
        pdf.add_page()
        pdf.chapter_title("6. Análisis de Componentes Principales (PCA) y Clustering")
        pdf.chapter_body("El Análisis de Componentes Principales (PCA) ayuda a reducir la dimensionalidad de los datos, identificando las direcciones de mayor varianza. El clustering agrupa activos similares en función de sus retornos, lo que puede ser útil para la diversificación.")
        if explained_var.size > 0:
            pdf.add_image_with_caption(pca_path, "Figura 6.1: Varianza Explicada por PCA")
            pdf.chapter_body(f"La varianza explicada por cada componente principal se ha guardado en el archivo Excel: {excel_results_path}")
        if not cluster_df.empty:
            pdf.add_image_with_caption(dendro_path, "Figura 6.2: Dendrograma de Clustering de Activos")
            pdf.chapter_body(f"Los resultados del clustering se han guardado en el archivo Excel: {excel_results_path}")

    # Sección de Carteras Específicas
    pdf.add_page()
    pdf.chapter_title("7. Carteras Específicas")
    pdf.chapter_body("Se han calculado y analizado diferentes tipos de carteras, incluyendo la de Máxima Sharpe, Mínima Volatilidad, y una cartera Equal-Weighted. También se muestra un ejemplo de combinación de carteras.")
    pdf.chapter_body(f"Portafolio Equal-Weighted (Xweight simplificado): {weights_equal}")
    pdf.chapter_body(f"Suma Producto de Carteras (50% Sharpe, 50% Min Vol): {combined_weights}")

    pdf_output_path = os.path.join(CARPETA_SALIDA, f"Informe_Analisis_Carteras_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    pdf.output(pdf_output_path)
    print(f"Informe PDF generado y guardado en {pdf_output_path}")

    # Limpiar archivos temporales (opcional, se puede comentar para depuración)
    # import shutil
    # shutil.rmtree(CARPETA_DATOS_TEMP)
    # shutil.rmtree(CARPETA_GRAFICOS_TEMP)
    # print("Archivos temporales limpiados.")

