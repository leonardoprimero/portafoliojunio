from descarga_datos import descargar_datos, limpiar_datos_crudos
from analisis_retornos import calcular_retornos_diarios_acumulados
from generar_pdf import generar_pdf_informe_por_activos
from generar_graficos import graficar_retorno_comparado
from analisis_correlaciones import calcular_matriz_correlacion


# ---------------- CONFIGURACIÓN DE ACCIONES ----------------
descargar = True       # Descargar nuevos datos desde el proveedor
limpiar = True         # Limpiar y transformar los datos crudos descargados
analizar = True         # Realizar análisis y gráficos
GENERAR_PDF = True  # ← Activalo o desactivalo desde acá
generar_comparativo = True  # ← Activa esto para ver todos los retornos en un solo gráfico
generar_correlaciones = False

# ---------------- CONFIGURACIÓN ----------------
tickers = ["AAPL", "MSFT", "GOOGL"]
start_date = "2000-01-01"
end_date = "2024-12-31"
proveedor = "yahoo"   # 'alphavantage', 'tiingo' , 'yahoo'

#  "bloomberg_dark"   "modern_light", "jupyter_quant", "nyu_quant"
tema_grafico = "bloomberg_dark"         # 'dark', 'vintage', 'modern', 'normal'
retornos_a_mostrar = ["log", "lineal"]  # podés usar: ["log"], ["lineal"] o ambos


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
        tipos_retornos=retornos_a_mostrar,
        calcular_rolling=activar_retornos_ventana,
        ventanas=ventanas_móviles,
        calcular_bloques=calcular_retornos_por_periodo,
        frecuencias=frecuencias_temporales,
        referencias_histograma=referencias_histograma
    )

if generar_comparativo:
    graficar_retorno_comparado(
        carpeta_datos_limpios="DatosLimpios", # Asumiendo que los datos limpios están aquí y tienen Daily_Return
        tema=tema_grafico,
        carpeta_salida_retornos="RetornoDiarioAcumulado"
    )

if GENERAR_PDF:
    generar_pdf_informe_por_activos(
        carpeta_imagenes="RetornoDiarioAcumulado",
        nombre_salida="informe_por_activos.pdf"
    )

if generar_correlaciones:
    calcular_matriz_correlacion(
        carpeta_datos_limpios="DatosLimpios",
        carpeta_salida="Correlaciones",
        metodo="pearson",         # o "spearman"
        tema=tema_grafico,
        extension_salida="xlsx"
    )
