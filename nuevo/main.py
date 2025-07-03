from descarga_datos import descargar_datos, limpiar_datos_crudos
from analisis_retornos import calcular_retornos_diarios_acumulados, analizar_tasas_libres_riesgo
from generar_pdf import generar_pdf_informe_por_activos, generar_pdf_informe_correlaciones, generar_pdf_backtesting
from generar_graficos import graficar_retorno_comparado
import pandas as pd
from datetime import datetime
import glob, os
from aaaconfig_usuario import *
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
from backtest_portafolio import (backtest_profesional, buscar_archivo_portafolio,
                                 backtest_equal_weight, BackTestingReal,
                                 buscar_cliente_por_dni_email)

from selector_activos import ejecutar_selector_activos



# # ---------------- CONFIGURACIÓN DE ACCIONES ----------------
# descargar = True       # Descargar nuevos datos desde el proveedor
# limpiar = True         # Limpiar y transformar los datos crudos descargados
# analizar = True         # Realizar análisis y gráficos
# GENERAR_PDF = True  # ← Activalo o desactivalo desde acá
# generar_comparativo = True  # ← Activa esto para ver todos los retornos en un solo gráfico
# generar_correlaciones = True
# generar_clustermap = True  # En la mtariz de correlación
# mostrar_dendrograma = True

# # Nueva configuración para correlaciones rolling
# generar_correlaciones_rolling = True
# ventana_rolling = 60 # Días para la ventana móvil
# solo_graficar_pares = True     # Si está en True, solo corre la parte de graficar pares específicos
# pares_especificos = []  # Por default vacío. Si querés pares, descomentá la línea de abajo y ponelos.
# #pares_especificos = ["AAPL-MSFT", "GOOGL-NVDA"]
# generar_pca = True

# # Bandera para generar el informe de correlaciones en PDF
# generar_pdf_correlaciones = True  

# ##-----------AHORA SI ANALISIS PORTAFOLIO---------------##

# ##   MONTECARLO 

# simular_cartera = True  # Activalo o desactivalo
# n_iteraciones = 80000
# capital_usd = 100000
# peso_min = 0.05   # 5%
# peso_max = 0.25   # 25%
# USAR_BENCHMARK = True
# BENCHMARK_TICKER = "SPY"
# BENCHMARK_COMO_ACTIVO = False
# hacer_backtest = True 
# GENERAR_PDF_BACKTEST = True
# hacer_backtest_iguales = True
# GENERAR_PDF_BACKTEST = True
# hacer_backtest_real = True
# carpeta_clientes = "datosgenerales/Clientes"
# dni_filtrar = 33428871
# GENERAR_PDF_BACKTEST = True



## ---------------- CONFIGURACIÓN ----------------
# tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "UNH", "WMT", "NVDA", "KO", "PFE","SPY"]
# start_date = "2000-01-01"
# end_date = "2024-12-31"
# proveedor = "yahoo"   # 'alphavantage', 'tiingo' , 'yahoo'

# #  "bloomberg_dark"   "modern_light", "jupyter_quant", "nyu_quant"
# tema_grafico = "bloomberg_dark"         # 'dark', 'vintage', 'modern', 'normal'
# retornos_a_mostrar = ["log", "lineal"]  # podés usar: ["log"], ["lineal"] o ambos

# activar_retornos_ventana = False
# ventanas_móviles = [5, 22, 252]  # Por defecto: semanal, mensual, anual

# # Opcional: calcular retornos por bloque (semana/mes/año)
# calcular_retornos_por_periodo = False
# frecuencias_temporales = ["W-FRI", "ME", "YE"]  # semanal, mensual, anual

# #referencias_histograma = {
#     "media": True,
#     "sigma": True,
#     "mediana": True,
#     "p1": False,
#     "p10": False,
#     "p25": False,
#     "p75": False,
#     "p90": False,
#     "p99": False
# }
# if USAR_BENCHMARK and not BENCHMARK_COMO_ACTIVO:
#     tickers_portafolio = [t for t in tickers if t != BENCHMARK_TICKER]
# else:
#     tickers_portafolio = tickers

# ---------------- EJECUCIÓN DE ACCIONES ----------------

acciones = [
    ("Descargando datos", descargar),
    ("Limpiando datos", limpiar),
    ("Análisis de retornos", analizar),
    ("Análisis tasas libres de riesgo", analizar_tasas_libres),
    ("Selector de activos óptimos", hacer_seleccion_activos),
    ("Gráfico comparativo", generar_comparativo),
    ("Generar PDF activos", GENERAR_PDF),
    ("Matriz de correlaciones", generar_correlaciones),
    ("Rolling correlations", generar_correlaciones_rolling),
    ("Análisis PCA", generar_pca),
    ("PDF correlaciones", generar_pdf_correlaciones),
    ("Simulación cartera Monte Carlo", simular_cartera),
  #  ("simular_cartera_de_los_activos_seleccionados", activos_seleccionados),
    ("Backtesting portafolio óptimo", hacer_backtest),
    ("Backtesting portafolio igual ponderado", hacer_backtest_iguales),
    
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
        
    if analizar_tasas_libres:
        analizar_tasas_libres_riesgo(
            carpeta_datos_limpios="DatosLimpios",
            carpeta_salida="TasasLibresRiesgo"
        )
        progress.advance(tarea)

    if generar_comparativo:
        graficar_retorno_comparado(
            carpeta_datos_limpios="DatosLimpios",
            tema=tema_grafico,
            carpeta_salida_retornos="RetornoDiarioAcumulado"
        )
        progress.advance(tarea)
        
    if hacer_seleccion_activos:
        from selector_activos import ejecutar_selector_activos
        ejecutar_selector_activos()
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
        
    
    if simular_cartera:
        datosTickers, port_opt = markowitz_simulacion(
            tickers=tickers_portafolio,
            carpeta_datos_limpios="DatosLimpios",
            n_iter=n_iteraciones,
            carpeta_salida="Montecarlo",
            tema="modern_light",
            capital_usd=capital_usd,
            peso_min=peso_min,
            peso_max=peso_max,
            usar_benchmark=USAR_BENCHMARK,
            benchmark_ticker=BENCHMARK_TICKER,
            benchmark_como_activo=BENCHMARK_COMO_ACTIVO
        )
        pesos_opt = port_opt["pesos"]   # <--- ¡Siempre definí pesos_opt!

    if not simular_cartera and hacer_backtest:
        archivo_opt = buscar_archivo_portafolio()
        print(f"🟢 Usando archivo de pesos óptimos: {archivo_opt}")
        df_opt = pd.read_excel(archivo_opt)
        fila_opt = df_opt[df_opt["Tipo"].str.contains("Sharpe", case=False)].iloc[0]
        pesos_opt = [fila_opt[f"{t} (%)"] for t in tickers_portafolio]

    if hacer_backtest:
        backtest_profesional(
            pesos=pesos_opt,                # <--- ¡SIEMPRE llamá con pesos_opt!
            tickers=tickers_portafolio,
            benchmark=BENCHMARK_TICKER,
            carpeta="DatosLimpios",
            carpeta_salida="BacktestPortafolioPro",
            capital=capital_usd
        )
        progress.advance(tarea)
    if GENERAR_PDF_BACKTEST:
        from generar_pdf import generar_pdf_backtesting
        generar_pdf_backtesting(
            carpeta_imagenes="BacktestPortafolioPro",
            nombre_salida="BacktestPortafolioPro/informe_backtesting_optimo.pdf",
            titulo="Informe Backtesting Portafolio Óptimo"
        )
        progress.advance(tarea)
        
    if hacer_backtest_iguales:
        backtest_equal_weight(
            tickers=tickers_portafolio,
            benchmark=BENCHMARK_TICKER,
            carpeta="DatosLimpios",
            carpeta_salida="BackTestingPortafolioIguales",
            capital=capital_usd
        )
        progress.advance(tarea)
    if GENERAR_PDF_BACKTEST:
        from generar_pdf import generar_pdf_backtesting
        generar_pdf_backtesting(
            carpeta_imagenes="BackTestingPortafolioIguales",
            nombre_salida="BackTestingPortafolioIguales/informe_backtesting_igual.pdf",
            titulo="Informe Backtesting Portafolio Igual"
        )
        progress.advance(tarea)
        
if hacer_backtest_real:
    if dni_filtrar:
        path_cliente = buscar_cliente_por_dni_email(dni=dni_filtrar)
        if path_cliente:
            print(f"🟢 Procesando solo cliente con DNI {dni_filtrar}: {os.path.basename(path_cliente)}")
            nombre, apellido, nombre_carpeta, fecha_inicio = BackTestingReal(excel_path=path_cliente)
            if GENERAR_PDF:
                generar_pdf_backtesting(
                    carpeta_imagenes=nombre_carpeta,
                    nombre_salida=f"{nombre_carpeta}/informe_backtesting_cliente.pdf",
                    titulo=f"Cartera de {nombre} {apellido}",
                    subtitulo=f"{fecha_inicio.strftime('%d/%m/%Y')} a {datetime.today().strftime('%d/%m/%Y')}",
                    path_fondo="datosgenerales/hojaMembretada.jpg"   # Cambia a None si NO querés membretada
                )
        else:
            print(f"😕 No hemos encontrado ese DNI ({dni_filtrar}). ¿Querés probar con el email? (s/n)")
            opcion = input().strip().lower()
            if opcion == "s":
                email_buscar = input("Escribí el email exacto del cliente: ").strip()
                path_cliente = buscar_cliente_por_dni_email(email=email_buscar)
                if path_cliente:
                    print(f"🟢 Procesando solo cliente con email {email_buscar}: {os.path.basename(path_cliente)}")
                    nombre, apellido, nombre_carpeta, fecha_inicio = BackTestingReal(excel_path=path_cliente)
                    if GENERAR_PDF_BACKTEST:
                        generar_pdf_backtesting(
                            carpeta_imagenes=nombre_carpeta,
                            nombre_salida=f"{nombre_carpeta}/informe_backtesting_cliente.pdf",
                            titulo=f"Cartera de {nombre} {apellido}",
                            subtitulo=f"{fecha_inicio.strftime('%d/%m/%Y')} a {datetime.today().strftime('%d/%m/%Y')}",
                            path_fondo="datosgenerales/hojaMembretada.jpg"
                        )
                else:
                    print(f"❌ Tampoco encontramos ese email ({email_buscar}). Revisa bien los datos.")
            else:
                print("❗ Finalizando el programa. Revisa el DNI o corre todos los clientes (poné dni_filtrar=None).")
    else:
        excels_clientes = glob.glob(os.path.join(carpeta_clientes, "*.xlsx"))
        for excel_path in excels_clientes:
            print(f"Procesando backtest real para: {os.path.basename(excel_path)}")
            nombre, apellido, nombre_carpeta, fecha_inicio = BackTestingReal(excel_path=excel_path)
            if GENERAR_PDF_BACKTEST:
                generar_pdf_backtesting(
                    carpeta_imagenes=nombre_carpeta,
                    nombre_salida=f"{nombre_carpeta}/informe_backtesting_cliente.pdf",
                    titulo=f"Cartera de {nombre} {apellido}",
                    subtitulo=f"{fecha_inicio.strftime('%d/%m/%Y')} a {datetime.today().strftime('%d/%m/%Y')}",
                    path_fondo="datosgenerales/hojaMembretada.jpg"
                )