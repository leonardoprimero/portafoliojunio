from descarga_datos import descargar_datos, limpiar_datos_crudos
from analisis_retornos import calcular_retornos_diarios_acumulados

# ---------------- CONFIGURACIÓN DE ACCIONES ----------------
descargar = False       # Descargar nuevos datos desde el proveedor
limpiar = False         # Limpiar y transformar los datos crudos descargados
analizar = True         # Realizar análisis y gráficos

# ---------------- CONFIGURACIÓN ----------------
tickers = ["AAPL", "MSFT", "GOOGL"]
start_date = "2000-01-01"
end_date = "2024-12-31"
proveedor = "yahoo"   # 'alphavantage', 'tiingo' , 'yahoo'

#  "bloomberg_dark"   "modern_light", "jupyter_quant", "nyu_quant"
tema_grafico = "bloomberg_dark"         # 'dark', 'vintage', 'modern', 'normal'
retorno_logaritmico = True     # True para logarítmico, False para simple


activar_retornos_ventana = False
ventanas_móviles = [5, 22, 252]  # Por defecto: semanal, mensual, anual

# Opcional: calcular retornos por bloque (semana/mes/año)
calcular_retornos_por_periodo = False
frecuencias_temporales = ["W-FRI", "ME", "YE"]  # semanal, mensual, anual

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

# ---------------- EJECUCIÓN DE ACCIONES ----------------
if descargar:
    descargar_datos(tickers, start_date, end_date, proveedor)

if limpiar:
    limpiar_datos_crudos()

if analizar:
    calcular_retornos_diarios_acumulados(
        carpeta_datos_limpios="DatosLimpios",
        carpeta_salida="RetornoDiarioAcumulado",
        tema=tema_grafico,
        logaritmico=retorno_logaritmico,
        calcular_rolling=activar_retornos_ventana,
        ventanas=ventanas_móviles,
        calcular_bloques=calcular_retornos_por_periodo,
        frecuencias=frecuencias_temporales,
        referencias_histograma=referencias_histograma
    )
