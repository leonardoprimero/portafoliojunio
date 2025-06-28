# =============== CONFIGURACIÓN EDITABLE POR EL USUARIO ===============

# Acciones principales
descargar = False
limpiar = False
analizar = False
GENERAR_PDF = False
generar_comparativo = False
generar_correlaciones = False
generar_clustermap = False
mostrar_dendrograma = False

# Correlaciones rolling
generar_correlaciones_rolling = False
ventana_rolling = 60
solo_graficar_pares = False
pares_especificos = []
# pares_especificos = ["AAPL-MSFT", "GOOGL-NVDA"]
generar_pca = False

generar_pdf_correlaciones = False

# ---- Análisis de portafolio (Monte Carlo) ----
simular_cartera = False
n_iteraciones = 10000
capital_usd = 70000
peso_min = 0.05
peso_max = 0.25
USAR_BENCHMARK = True
BENCHMARK_TICKER = "SPY"
BENCHMARK_COMO_ACTIVO = False


hacer_backtest = False
GENERAR_PDF_BACKTEST = False
hacer_backtest_iguales = False
GENERAR_PDF_BACKTEST = False
hacer_backtest_real = True
carpeta_clientes = "datosgenerales/Clientes"
dni_filtrar = 32630214
GENERAR_PDF_BACKTEST = True

# ---- Parámetros generales ----
tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "UNH", "WMT", "NVDA", "KO", "PFE", "SPY"]
start_date = "2000-01-01"
end_date = "2025-06-26"
proveedor = "yahoo"  # 'alphavantage', 'tiingo', 'yahoo'

tema_grafico = "bloomberg_dark"
retornos_a_mostrar = ["log", "lineal"]

activar_retornos_ventana = False
ventanas_móviles = [5, 22, 252]

calcular_retornos_por_periodo = False
frecuencias_temporales = ["W-FRI", "ME", "YE"]

referencias_histograma = {
    "media": True,
    "sigma": True,
    "mediana": True,
    "p1": False,
    "p10": False,
    "p25": False,
    "p75": False,
    "p90": False,
    "p99": False
}

# Calcula tickers_portafolio automáticamente
if USAR_BENCHMARK and not BENCHMARK_COMO_ACTIVO:
    tickers_portafolio = [t for t in tickers if t != BENCHMARK_TICKER]
else:
    tickers_portafolio = tickers

# =============== FIN DE LA CONFIGURACIÓN ===============
