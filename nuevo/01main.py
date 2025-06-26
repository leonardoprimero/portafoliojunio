from descarga_datos import descargar_datos, limpiar_datos_crudos
from analisis_retornos import calcular_retornos_diarios_acumulados
from generar_pdf import generar_pdf_informe_por_activos, generar_pdf_informe_correlaciones
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
    guardar_estabilidad_rolling,
    ejecutar_pca_sobre_retornos
)
from analisis_correlacion_sectores import (
    cargar_matriz_correlacion,
    cargar_sectores,
    matriz_correlacion_sectorial,
    graficar_heatmap_sectorial,
    guardar_excel_sectorial
)
from rich.progress import Progress
from analisis_cartera import markowitz_simulacion

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
generar_correlaciones_rolling = False
ventana_rolling = 60 # Días para la ventana móvil
solo_graficar_pares = False     # Si está en True, solo corre la parte de graficar pares específicos
pares_especificos = []  # Por default vacío. Si querés pares, descomentá la línea de abajo y ponelos.
#pares_especificos = ["AAPL-MSFT", "GOOGL-NVDA"]
generar_pca = False

# Bandera para generar el informe de correlaciones en PDF
generar_pdf_correlaciones = False  

##-----------AHORA SI ANALISIS PORTAFOLIO---------------##

simular_cartera = True  # Activalo o desactivalo
n_iteraciones = 8000

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

acciones = [
    ("Descargando datos", descargar),
    ("Limpiando datos", limpiar),
    ("Análisis de retornos", analizar),
    ("Gráfico comparativo", generar_comparativo),
    ("Generar PDF activos", GENERAR_PDF),
    ("Matriz de correlaciones", generar_correlaciones),
    ("Rolling correlations", generar_correlaciones_rolling),
    ("Análisis PCA", generar_pca),
    ("PDF correlaciones", generar_pdf_correlaciones),
    ("Simulación cartera Monte Carlo", simular_cartera)
]

with Progress() as progress:
    total = sum(1 for _, flag in acciones if flag)
    tarea = progress.add_task("[cyan]Ejecución en progreso...", total=total)

    if descargar:
        descargar_datos(tickers, start_date, end_date, proveedor)
        progress.advance(tarea)

    if limpiar:
        limpiar_datos_crudos()
        progress.advance(tarea)

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
        progress.advance(tarea)

    if generar_comparativo:
        graficar_retorno_comparado(
            carpeta_datos_limpios="DatosLimpios",
            tema=tema_grafico,
            carpeta_salida_retornos="RetornoDiarioAcumulado"
        )
        progress.advance(tarea)

    if GENERAR_PDF:
        generar_pdf_informe_por_activos(
            carpeta_imagenes="RetornoDiarioAcumulado",
            nombre_salida="informe_por_activos.pdf"
        )
        progress.advance(tarea)

    if generar_correlaciones:
        matriz_cluster = calcular_matriz_correlacion(
            carpeta_datos_limpios="DatosLimpios",
            carpeta_salida="Correlaciones",
            metodo="pearson",
            tema=tema_grafico,
            extension_salida="xlsx",
            generar_clustermap=generar_clustermap,
            mostrar_dendrograma=mostrar_dendrograma
        )
        if matriz_cluster is not None:
            resumen = calcular_metricas_resumen_correlacion(matriz_cluster)
            guardar_metricas_resumen(resumen, carpeta_salida="Correlaciones")
            top_pos, top_neg = ranking_correlaciones_extremas(matriz_cluster)
            guardar_rankings_extremos(top_pos, top_neg, carpeta_salida="Correlaciones")
        progress.advance(tarea)

    if generar_correlaciones_rolling:
        df_rolling_correlations = calcular_correlaciones_rolling(
            carpeta_datos_limpios="DatosLimpios",
            carpeta_salida="CorrelacionesRolling",
            metodo="pearson",
            ventana=ventana_rolling,
            tema=tema_grafico,
            top_n_pares_mas_volatiles=2
        )
        if df_rolling_correlations is not None:
            resumen_estabilidad = calcular_estabilidad_rolling(df_rolling_correlations)
            guardar_estabilidad_rolling(
                resumen_estabilidad,
                carpeta_salida="CorrelacionesRolling",
                metodo="pearson",
                ventana=ventana_rolling
            )
        progress.advance(tarea)

    graficar_pares_rolling_especificos(
        carpeta_salida="CorrelacionesRolling",
        metodo="pearson",
        ventana=ventana_rolling,
        tema=tema_grafico,
        pares_especificos=pares_especificos
    )
    if generar_pca:
        ejecutar_pca_sobre_retornos(
        carpeta_datos_limpios="DatosLimpios",
        carpeta_salida="PCA",
        n_componentes=2,
        tema=tema_grafico
    )
    progress.advance(tarea
    )

    if generar_pdf_correlaciones:
        matriz = cargar_matriz_correlacion()
        mapa = cargar_sectores()
        sector_mat = matriz_correlacion_sectorial(matriz, mapa)
        graficar_heatmap_sectorial(sector_mat)
        guardar_excel_sectorial(sector_mat)

        generar_pdf_informe_correlaciones(
            nombre_salida="informe_correlaciones.pdf",
            carpeta_correlaciones="Correlaciones",
            carpeta_rolling="CorrelacionesRolling",
            pares_especificos=pares_especificos
        )
        progress.advance(tarea)
    
    if simular_cartera:
        markowitz_simulacion(
            tickers=tickers,
            carpeta_datos_limpios="DatosLimpios",
            n_iter=n_iteraciones,   # ← Lo seleccionás desde el main
            carpeta_salida="Montecarlo",
            tema="modern_light"  # Cambiá por "bloomberg_dark", "modern_light", "nyu_quant", "classic_white"
        )
        progress.advance(tarea)
