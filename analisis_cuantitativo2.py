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
from pypfopt import expected_returns, risk_models, EfficientFrontier

# --- CONFIGURACIÓN GLOBAL ---
CARPETA_SALIDA = "DatosCartera"
CARPETA_GRAFICOS_TEMP = os.path.join(CARPETA_SALIDA, "temp_graficos")
CARPETA_DATOS_CACHE = os.path.join(CARPETA_SALIDA, "data_cache")
BENCHMARK_DEFAULT = "SPY"

os.makedirs(CARPETA_SALIDA, exist_ok=True)
os.makedirs(CARPETA_GRAFICOS_TEMP, exist_ok=True)
os.makedirs(CARPETA_DATOS_CACHE, exist_ok=True)

# --- FUNCIONES DE UTILIDAD ---
def descargar_datos(tickers, start_date, end_date, benchmark=None):
    todos_tickers = list(tickers)
    if benchmark and benchmark not in todos_tickers:
        todos_tickers.append(benchmark)
    data = yf.download(todos_tickers, start=start_date, end=end_date)
    if data.empty:
        return pd.DataFrame()
    if "Adj Close" in data.columns:
        df_precios = data["Adj Close"].dropna(how="all")
    elif "Close" in data.columns:
        df_precios = data["Close"].dropna(how="all")
    else:
        return pd.DataFrame()
    for ticker in df_precios.columns:
        filepath = os.path.join(CARPETA_DATOS_CACHE, f"{ticker}.csv")
        df_precios[[ticker]].to_csv(filepath)
    return df_precios

def cargar_datos_locales(tickers, benchmark=None):
    todos_tickers = list(tickers)
    if benchmark and benchmark not in todos_tickers:
        todos_tickers.append(benchmark)
    df_cargado = pd.DataFrame()
    for ticker in todos_tickers:
        filepath = os.path.join(CARPETA_DATOS_CACHE, f"{ticker}.csv")
        if not os.path.exists(filepath):
            return None
        df_temp = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df_cargado = pd.concat([df_cargado, df_temp], axis=1)
    return df_cargado

def calcular_beta(activo, benchmark, retornos):
    cov_ab = retornos[activo].cov(retornos[benchmark])
    var_b = retornos[benchmark].var()
    return cov_ab / var_b if var_b != 0 else np.nan

def calcular_rendimiento_cartera(df_activos, pesos):
    retornos = np.log(df_activos / df_activos.shift(1)).dropna()
    retorno_cartera = (retornos * pd.Series(pesos)).sum(axis=1)
    return (1 + retorno_cartera).cumprod()

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

    def add_image(self, path, caption):
        if os.path.exists(path):
            self.image(path, x=self.l_margin, w=self.w - 2 * self.l_margin)
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, caption, 0, 1, 'C')
            self.ln(5)

if __name__ == "__main__":
    activos = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    fecha_inicio = "2020-01-01"
    fecha_fin = "2024-12-31"
    benchmark = BENCHMARK_DEFAULT

    df = cargar_datos_locales(activos, benchmark)
    if df is None:
        df = descargar_datos(activos, fecha_inicio, fecha_fin, benchmark)
    if df.empty:
        exit()

    if benchmark in df.columns:
        df_benchmark = df[[benchmark]]
        df_activos = df.drop(columns=[benchmark])
    else:
        df_benchmark = None
        df_activos = df.copy()

    retornos = np.log(df_activos / df_activos.shift(1)).dropna()
    mu = expected_returns.mean_historical_return(df_activos)
    S = risk_models.sample_cov(df_activos)

    ef = EfficientFrontier(mu, S)
    w_sharpe = ef.max_sharpe()
    cleaned_w_sharpe = ef.clean_weights()
    perf_sharpe = ef.portfolio_performance(verbose=False)

    ef = EfficientFrontier(mu, S)
    w_minvol = ef.min_volatility()
    cleaned_w_minvol = ef.clean_weights()
    perf_minvol = ef.portfolio_performance(verbose=False)

    w_equal = {a: 1 / len(df_activos.columns) for a in df_activos.columns}
    w_combi = {a: 0.5 * cleaned_w_sharpe.get(a, 0) + 0.5 * cleaned_w_minvol.get(a, 0) for a in df_activos.columns}

    ret_sharpe = calcular_rendimiento_cartera(df_activos, cleaned_w_sharpe)
    ret_minvol = calcular_rendimiento_cartera(df_activos, cleaned_w_minvol)
    ret_equal = calcular_rendimiento_cartera(df_activos, w_equal)
    ret_combi = calcular_rendimiento_cartera(df_activos, w_combi)

    if df_benchmark is not None:
        ret_benchmark = calcular_rendimiento_cartera(df_benchmark, {benchmark: 1})
    else:
        ret_benchmark = pd.Series(dtype=float)

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.chapter_title("Resumen de Rendimientos Comparativos")
    pdf.chapter_body("Se comparan los retornos totales, anualizados y volatilidades para cuatro carteras: Máxima Sharpe, Mínima Volatilidad, Equal-Weighted y Combinada, junto al benchmark SPY.")

    def metricas(ret):
        total = (ret.iloc[-1] / ret.iloc[0]) - 1
        annual = (1 + total)**(252 / len(ret)) - 1
        vol = ret.pct_change().std() * np.sqrt(252)
        sharpe = annual / vol if vol != 0 else np.nan
        return total, annual, vol, sharpe

    data_metrics = {
        "Máxima Sharpe": metricas(ret_sharpe),
        "Mínima Volatilidad": metricas(ret_minvol),
        "Equal Weighted": metricas(ret_equal),
        "Combinada": metricas(ret_combi),
        "Benchmark (SPY)": metricas(ret_benchmark) if not ret_benchmark.empty else (np.nan, np.nan, np.nan, np.nan)
    }

    pdf.set_font("Arial", size=10)
    col1, col2, col3, col4, col5 = "Cartera", "Total Return", "Anual", "Volat.", "Sharpe"
    pdf.cell(40, 10, col1)
    pdf.cell(40, 10, col2)
    pdf.cell(40, 10, col3)
    pdf.cell(40, 10, col4)
    pdf.cell(30, 10, col5)
    pdf.ln()
    for name, (t, a, v, s) in data_metrics.items():
        pdf.cell(40, 10, name)
        pdf.cell(40, 10, f"{t:.2%}" if pd.notna(t) else "N/A")
        pdf.cell(40, 10, f"{a:.2%}" if pd.notna(a) else "N/A")
        pdf.cell(40, 10, f"{v:.2%}" if pd.notna(v) else "N/A")
        pdf.cell(30, 10, f"{s:.2f}" if pd.notna(s) else "N/A")
        pdf.ln()

    output_path = os.path.join(CARPETA_SALIDA, f"Informe_Analisis_Carteras_Mejorado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    pdf.output(output_path)
    print(f"✅ PDF generado en: {output_path}")
