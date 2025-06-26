from descarga_datos import descargar_datos, limpiar_datos_crudos
from analisis_retornos import calcular_retornos_diarios_acumulados
from generar_pdf import generar_pdf_informe_por_activos
from generar_graficos import graficar_retorno_comparado
from analisis_correlaciones import (
    calcular_matriz_correlacion,
    calcular_correlaciones_rolling,
    graficar_pares_rolling_especificos,
    calcular_metricas_resumen_correlacion,
    guardar_metricas_resumen,
    ranking_correlaciones_extremas,
    guardar_rankings_extremos,
    calcular_estabilidad_rolling,
    guardar_estabilidad_rolling
)

# ---------------- CONFIGURACIÓN DE ACCIONES ----------------
descargar = False       # Descargar nuevos datos desde el proveedor
limpiar = False         # Limpiar y transformar los datos crudos descargados
analizar = False         # Realizar análisis y gráficos
GENERAR_PDF = False  # ← Activalo o desactivalo desde acá
generar_comparativo = False  # ← Activa esto para ver todos los retornos en un solo gráfico
generar_correlaciones = False
generar_clustermap = False  # En la mtariz de correlación
mostrar_dendrograma = False

# Nueva configuración para correlaciones rolling
generar_correlaciones_rolling = True
ventana_rolling = 60 # Días para la ventana móvil
solo_graficar_pares = True     # Si está en True, solo corre la parte de graficar pares específicos
pares_especificos = []  # Por default vacío. Si querés pares, descomentá la línea de abajo y ponelos.
#pares_especificos = ['''"AAPL-MSFT", "GOOGL-NVDA"''']  


# ---------------- CONFIGURACIÓN ----------------
tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "UNH", "WMT", "NVDA", "KO", "PFE","SPY"]
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
    matriz_cluster = calcular_matriz_correlacion(
        carpeta_datos_limpios="DatosLimpios",
        carpeta_salida="Correlaciones",
        metodo="pearson",         # o "spearman"
        tema=tema_grafico,
        extension_salida="xlsx",
        generar_clustermap=generar_clustermap,
        mostrar_dendrograma=mostrar_dendrograma
    )

    # 👇 NUEVO: análisis extendido si hay matriz
    if matriz_cluster is not None:
        resumen = calcular_metricas_resumen_correlacion(matriz_cluster)
        guardar_metricas_resumen(resumen, carpeta_salida="Correlaciones")
        top_pos, top_neg = ranking_correlaciones_extremas(matriz_cluster)
        guardar_rankings_extremos(top_pos, top_neg, carpeta_salida="Correlaciones")

if generar_correlaciones_rolling:
    df_rolling_correlations = calcular_correlaciones_rolling(
        carpeta_datos_limpios="DatosLimpios",
        carpeta_salida="CorrelacionesRolling",
        metodo="pearson",            # o "spearman"
        ventana=ventana_rolling,
        tema=tema_grafico,
        top_n_pares_mas_volatiles=2  # <--- Cambiá este número según lo que quieras
    )

    # 👇 NUEVO: estabilidad rolling
    if df_rolling_correlations is not None:
        resumen_estabilidad = calcular_estabilidad_rolling(df_rolling_correlations)
        guardar_estabilidad_rolling(
            resumen_estabilidad,
            carpeta_salida="CorrelacionesRolling",
            metodo="pearson",
            ventana=ventana_rolling
        )

graficar_pares_rolling_especificos(
    carpeta_salida="CorrelacionesRolling",
    metodo="pearson",
    ventana=ventana_rolling,
    tema=tema_grafico,
    pares_especificos=pares_especificos
)
