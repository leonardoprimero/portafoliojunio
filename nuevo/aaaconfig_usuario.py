# =============== CONFIGURACIÓN EDITABLE POR EL USUARIO ===============

# Acciones principales

# Primera etapa

descargar = False 
limpiar = False   
analizar = False 
generar_comparativo = False
GENERAR_PDF = False
analizar_tasas_libres = False 

# ^IRX: Treasury 13 Week Bill Yield      (≈ 3 meses, T-bill corto plazo)
# ^FVX: Treasury 5 Year Note Yield       (5 años, T-note mediano plazo)
# ^TNX: Treasury 10 Year Note Yield      (10 años, T-note clásico, referencia global)
# ^TYX: Treasury 30 Year Bond Yield      (30 años, bono largo plazo)

tasa_libre_riesgo_ticker = "^TNX"

hacer_seleccion_activos = False  # Activa la lógica para seleccionar cartera desde todos los activos
nivel_volatilidad_cliente = 0.25  # Máxima volatilidad tolerada
sectores_cliente = []  # O dejarlo en [] para todos    "Technology", "Healthcare", "Energy"
max_activos_por_sector = 3

# Segunda etapa





cartera_para_correlacion = "Moderado"  #"todos", "Conservador", "Mixta", "Agresivo", 

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
USAR_BENCHMARK = False
BENCHMARK_TICKER = "SPY"
BENCHMARK_COMO_ACTIVO = False

simular_cartera_de_los_activos_seleccionados = False


hacer_backtest = False
GENERAR_PDF_BACKTEST = False
hacer_backtest_iguales = False
GENERAR_PDF_BACKTEST = False
hacer_backtest_real = False
carpeta_clientes = "datosgenerales/Clientes"
dni_filtrar = None 
GENERAR_PDF_BACKTEST = False

# ---- Parámetros generales ----

tickers = ['MMM', 'ABT', 'ABBV', 'ANF', 'ACN',
           'ADGO', 'ADBE', 'JMIA', 'AAP', 
           'AMD', 'AEG', 'AEM', 'ABNB', 'BABA', 'GOOGL', 
           'MO', 'AMZN', 'ABEV', 
           'AMX', 'AAL', 'AXP', 'AIG', 'AMGN', 'ADI', 'AAPL', 
           'AMAT', 'ARCO', 'ARKK', 'ARM', 'ASML', 'AZN', 'TEAM', 
           'T', 'ADP', 'AVY', 'CAR', 'BIDU', 'BKR', 'BBD', 
           'BSBR', 'SAN', 'BA.C', 'BCS', 
           'B', 'BHP', 'BBV', 'BIOX', 'BIIB', 
           'BB', 'BKNG', 'BP', 'LND', 'BAK', 'BRFS', 'AVGO', 'BNG', 'AI', 
           'CAH', 'CCL', 'CAT', 'CLS', 'CX', 'EBR', 'SCHW', 'CVX', 
           'SNP', 'CSCO', 'C', 'CDE', 'COIN', 
           'ELP', 'SID', 'CSNA3', 'CEG', 'GLW', 'CAAP', 'COST', 'CS', 
           'CVS', 'DHR', 'BSN', 'DECK', 'DE', 'DAL', 'DTEA', 'DOCU', 'DOW', 
           'DD', 'EOAN', 'EBAY', 'EA', 'LLY', 'AKO.B', 'ERJ', 'E', 'EFX', 
           'São', 'EQNR', 'ETSY', 'XOM', 'FNMA', 'FDX', 'RACE', 'FSLR', 
           'FMX', 'F', 'FMCC', 'FCX', 'GRMN', 'GE', 'GM', 'GPRK', 'GGB', 
           'GILD', 'GLOB', 'GFI', 'GT', 'PAC', 'ASR', 'TV', 'GSK', 'HAL', 
           'HAPV3', 'HOG', 'HMY', 'HDB', 'HL', 'HMC', 'HON', 'HWM', 'HPQ', 
           'HSBC', 'HNPIY', 'HUT', 'IBN', 'INFY', 'ING', 'INTC', 'IBM', 
           'IFF', 'IP', 'ITUB', 'JPM', 'JBSS3', 'JD', 'JNJ', 'JCI', 'JOYY', 
           'KB', 'KMB', 'KGC', 'PHG', 'KEP', 'LRCX', 'LVS', 'LAR', 'LAC', 
           'LYG', 'ERIC', 'RENT3', 'LMT', 'LREN3', 'MGLU3', 'MRVL', 'MA', 
           'MCD', 'MUX', 'MDT', 'MELI', 'MBG', 'MRK', 'META', 'MU', 'MSFT', 
           'MSTR', 'MFG', 'MBT', 'MRNA', 'MDLZ', 'MSI', 'NGG', 'NEC1', 
           'NTES', 'NEM', 'NXE', 'NKE', 'NIO', 'NSAN', 'NOKA', 'NG', 'NVS', 
           'UN', 'NUE', 'OXY', 'ORCL', 'ORAN', 'PCAR', 'PAGS', 'PLTR', 
           'PANW', 'PAAS', 'PCRF', 'PYPL', 'PDD', 'PSO', 'PEP', 'PRIO3', 
           'PETR3', 'PBR', 'PTR', 'PFE', 'PM', 'PSX', 'PINS', 'PBI', 'PKS', 
           'PG', 'QCOM', 'RTX', 'RGTI', 'RIO', 'RIOT', 'RBLX', 'ROKU', 
           'ROST', 'SPGI', 'CRM', 'SAP', 'SATL', 'SLB', 'SE', 'NOW', 'SIEGY', 
           'SI', 'SWKS', 'SNAP', 'SNA', 'SNOW', 'SONY', 'SCCO', 'SPOT', 
           'XYZ', 'SBUX', 'STLA', 'STNE', 'SDA', 'SUZB3', 'SUZ', 'SYY', 
           'TSM', 'TGT', 'TTM', 'RCTB4', 'VIV', 'VIVT3', 'TEFO', 'TIMS3', 
           'TEM', 'TEN', 'TXR', 'TSLA', 'TXN', 'BK', 'BA', 'KO', 'GS', 'HD', 
           'MOS', 'TRVV', 'DISN', 'TIMB', 'TJX', 'TMUS', 'TTE', 'TM', 
           'TCOM', 'TRIP', 'TWLO', 'TWTR', 'USB', 'UBER', 'PATH', 'UGP', 
           'UL', 'UNP', 'UAL', 'X', 'UNH', 'UPST', 'URBN', 'VALE', 'VALE3', 
           'VIG', 'VRSN', 'VZ', 'VRTX', 'SPCE', 'V', 'VIST', 'VST', 'VOD', 
           'WBA', 'WMT', 'WEGE3', 'WBO', 'WFC', 'XROX', 'XP', 'New', 'AUY', 
           'YZCA', 'YELP', 'ZM', "^IRX", '^FVX', '^TNX', '^TYX']

# tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
#             "JPM", "BAC", "WFC", "GS", "MS",
#             "XOM", "CVX", "COP", "PSX", "OXY",
#             "UNH", "CI", "HUM", "ELV", "CVS",
#             "WMT", "COST", "TGT", "DG", "KR",
#             "NVDA", "AMD", "INTC", "AVGO", "QCOM",
#             "PEP", "KO", "MDLZ", "CL", "PG",
#             "HD", "LOW", "NKE", "SBUX", "MCD",
#             "BA", "LMT", "RTX", "GE", "NOC",
#             "DUK", "NEE", "SO", "D", "AEP", "SPY"]
start_date = "2000-01-01"
end_date = "2025-07-01"
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
