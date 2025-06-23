from descarga_datos import descargar_datos, limpiar_datos_crudos
from analisis_retornos import calcular_retornos_diarios_acumulados

# ---------------- CONFIGURACIÓN ----------------
tickers = ["AAPL", "MSFT", "GOOGL"]
start_date = "2020-01-01"
end_date = "2024-12-31"
proveedor = "yahoo"   # 'alphavantage', 'tiingo' , 'yahoo'

tema_grafico = "dark"          # Opciones: 'dark', 'vintage', 'modern', 'normal'
retorno_logaritmico = False     # True para logarítmico, False para simple

activar_retornos_ventana = False
ventanas_móviles = [5, 22, 252]  # Por defecto: semanal, mensual, anual

# Opcional: calcular retornos por bloque (semana/mes/año)
calcular_retornos_por_periodo = True
frecuencias_temporales = ["W-FRI", "M", "Y"]  # semanal, mensual, anual

# ------------------------------------------------

# DESCARGA Y LIMPIEZA
descargar_datos(tickers, start_date, end_date, proveedor)
limpiar_datos_crudos()

# ANÁLISIS
calcular_retornos_diarios_acumulados(
    carpeta_datos_limpios="DatosLimpios",
    carpeta_salida="RetornoDiarioAcumulado",
    tema=tema_grafico,
    logaritmico=retorno_logaritmico,
    calcular_rolling=activar_retornos_ventana,
    ventanas=ventanas_móviles,
    calcular_bloques=calcular_retornos_por_periodo,
    frecuencias=frecuencias_temporales
)


