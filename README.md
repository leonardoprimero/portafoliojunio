# Análisis Cuantitativo Integrado de Carteras de Inversión

Este script de Python realiza un análisis cuantitativo completo de una cartera de inversión, desde la obtención de datos históricos hasta la optimización de la cartera y la generación de reportes detallados en formatos PDF y Excel.

## Características Principales

*   **Obtención de Datos**: Descarga datos históricos de precios y volumen para múltiples tickers desde Yahoo Finance.
    *   Permite cachear los datos localmente en formato CSV para evitar descargas repetidas.
*   **Análisis Individual de Activos**: Para cada activo en la cartera:
    *   Cálculo de retornos diarios.
    *   Cálculo de indicadores técnicos configurables:
        *   Medias Móviles Simples (SMA).
        *   Índice de Fuerza Relativa (RSI).
        *   Convergencia/Divergencia de Medias Móviles (MACD).
        *   Bandas de Bollinger (BBands).
    *   Generación de gráficos: Precios con indicadores, Volumen, RSI, MACD, y Distribución de Retornos.
*   **Análisis y Optimización de Cartera**:
    *   Cálculo de la matriz de correlación de los retornos de los activos.
    *   Optimización de la cartera utilizando `PyPortfolioOpt` con varios métodos disponibles (Maximizar Ratio de Sharpe, Minimizar Volatilidad, etc.).
    *   Cálculo de rendimiento esperado, volatilidad y Ratio de Sharpe para la cartera optimizada.
    *   Simulación de Monte Carlo (opcional) para explorar el espacio de posibles carteras.
    *   Generación de gráficos de cartera: Heatmap de correlación, Frontera Eficiente, Composición de la cartera óptima, Evolución histórica simulada vs. Benchmark.
*   **Generación de Reportes**:
    *   **Reporte PDF**: Un informe completo y estructurado con portada, resumen ejecutivo, análisis detallado por activo (con gráficos), y análisis de la cartera optimizada (con gráficos y tablas).
    *   **Reporte Excel**: Un archivo `.xlsx` con múltiples hojas que incluyen:
        *   Resumen general de estadísticas y pesos.
        *   Datos históricos e indicadores para cada activo.
        *   Datos de precios, retornos y correlación de la cartera.
        *   Resultados de la simulación Monte Carlo (si se activa).
*   **Configurable**: Múltiples parámetros ajustables al inicio del script para personalizar el análisis (lista de tickers, fechas, indicadores, métodos de optimización, etc.).

## Requisitos

*   Python 3.7+
*   Bibliotecas principales (se pueden instalar vía `pip`):
    *   `pandas`
    *   `numpy`
    *   `yfinance`
    *   `matplotlib`
    *   `seaborn`
    *   `fpdf2` (o `fpdf` - el script usa la sintaxis de FPDF, `fpdf2` es un fork mantenido y recomendado)
    *   `openpyxl` (para escribir archivos Excel `.xlsx`)
    *   `PyPortfolioOpt`
    *   `Pillow` (para un mejor manejo de imágenes en el PDF)

Se recomienda crear un entorno virtual e instalar las dependencias. Un ejemplo de `requirements.txt` podría ser:
```
pandas>=1.3
numpy>=1.20
yfinance>=0.1.70
matplotlib>=3.4
seaborn>=0.11
fpdf2>=2.5
openpyxl>=3.0
PyPortfolioOpt>=1.5
Pillow>=9.0
```

## Estructura de Carpetas

Al ejecutar el script, se crearán las siguientes carpetas (si no existen) en el mismo directorio donde se encuentra el script:

*   `datos_entrada/`: Utilizada para guardar los archivos CSV cacheados de los datos de los tickers.
*   `reportes_generados/`: Contendrá los reportes PDF y Excel finales.
    *   `reportes_generados/temp_graficos/`: Contiene los gráficos generados como archivos PNG antes de ser incrustados en los reportes.
        *   `activos_individuales/`: Gráficos del análisis por activo.
        *   `analisis_cartera/`: Gráficos del análisis de cartera.

**Nota sobre `temp_graficos`**: Esta carpeta no se limpia automáticamente. Puede ser útil para inspeccionar los gráficos individualmente. Si desea eliminarla después de generar los reportes, puede hacerlo manualmente.

## Configuración del Script

El comportamiento del script se controla mediante variables definidas en la sección `CONFIGURACIÓN GENERAL DEL SCRIPT` al inicio de `analisis_cartera_integrado.py`. Algunas de las configuraciones clave incluyen:

*   `LISTA_TICKERS`: Lista de los tickers de los activos que compondrán la cartera principal.
*   `BENCHMARK_TICKER`: Ticker del activo a usar como benchmark (ej. 'SPY').
*   `FECHA_INICIO_DATOS`, `FECHA_FIN_DATOS`: Periodo para la obtención de datos.
*   `CAPITAL_TOTAL_CARTERA`: Capital total (informativo, no afecta la optimización de pesos directamente pero puede usarse para calcular montos).
*   `USAR_CACHE_DATOS`: `True` para usar/guardar datos en CSV; `False` para descargar siempre.
*   `TEMA_GRAFICOS`, `ESCALA_PRECIOS_GRAFICOS`, `PALETA_COLORES_GRAFICOS`: Opciones de personalización visual.
*   Variables `ACTIVAR_*` y `*_PERIODO`: Para activar/desactivar y configurar indicadores técnicos (SMA, RSI, MACD, BBands, Volumen).
*   `METODO_OPTIMIZACION`: Define el objetivo de la optimización de cartera (ej. "max_sharpe", "min_volatility").
*   `RETORNO_OBJETIVO_OPTIMIZACION`, `VOLATILIDAD_OBJETIVO_OPTIMIZACION`: Para métodos de optimización específicos.
*   `TASA_LIBRE_RIESGO`: Tasa libre de riesgo para el cálculo del Ratio de Sharpe.
*   `ACTIVAR_MONTE_CARLO`, `NUM_SIMULACIONES_MC`: Para la simulación Monte Carlo.
*   `NOMBRE_BASE_REPORTE`, `CARPETA_REPORTES`, `CARPETA_DATOS_ENTRADA`: Para nombres y rutas de salida.

## Cómo Ejecutar el Script

1.  Asegúrese de tener Python y todas las dependencias listadas instaladas.
2.  Guarde el script `analisis_cartera_integrado.py` en un directorio.
3.  Modifique la sección de `CONFIGURACIÓN GENERAL DEL SCRIPT` dentro del archivo según sus necesidades.
4.  Abra una terminal o línea de comandos, navegue al directorio donde guardó el script.
5.  Ejecute el script usando el comando:
    ```bash
    python analisis_cartera_integrado.py
    ```
6.  Una vez finalizada la ejecución, encontrará los reportes PDF y Excel en la carpeta `reportes_generados/`.

## Salidas del Script

*   **Un archivo PDF**: Nombrado `[NOMBRE_BASE_REPORTE]_[timestamp].pdf` (ej. `Reporte_Cartera_Integrado_20231027_103045.pdf`).
    *   Contiene un análisis detallado y visual de cada activo y de la cartera optimizada.
*   **Un archivo Excel**: Nombrado `[NOMBRE_BASE_REPORTE]_[timestamp].xlsx` (ej. `Reporte_Cartera_Integrado_20231027_103045.xlsx`).
    *   Contiene múltiples hojas con datos tabulares: resumen, datos e indicadores por activo, datos de cartera, y resultados de Monte Carlo.

## Posibles Mejoras Futuras

*   Interfaz gráfica de usuario (GUI) o aplicación web para facilitar la configuración y ejecución.
*   Integración de más fuentes de datos (ej. datos fundamentales).
*   Más opciones de optimización y backtesting de estrategias.
*   Mejoras en el formato y personalización de los reportes.
*   Optimización del rendimiento para grandes cantidades de datos o tickers.
```
