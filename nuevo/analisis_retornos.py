import pandas as pd
import numpy as np
import os
from generar_graficos import (
    generar_grafico_retorno_acumulado,
    generar_histograma_retorno,
    calcular_drawdown,
    generar_grafico_drawdown,
    exportar_y_graficar_ratios,
    generar_qq_plot, generar_grafico_volumen,
    test_jarque_bera,
    generar_tabla_jarque_bera_imagen,
    generar_grafico_autocorrelacion
)

def calcular_retornos_diarios_acumulados(
    carpeta_datos_limpios="DatosLimpios",
    carpeta_salida="RetornoDiarioAcumulado",
    tema="normal",
    tipos_retornos=["lineal"],
    calcular_rolling=False,
    ventanas=[5, 22, 252],
    calcular_bloques=False,
    frecuencias=["W-FRI", "ME", "YE"],
    referencias_histograma=None
):
    os.makedirs(carpeta_salida, exist_ok=True)
    print(f"\n📊 Calculando retornos diarios acumulados... (tipos: {tipos_retornos}, tema: {tema}, rolling: {calcular_rolling}, bloques: {calcular_bloques})")

    ratios_summary = []
    jb_results = []

    for root, dirs, files in os.walk(carpeta_datos_limpios):
        for file in files:
            if file.endswith(".csv"):
                ticker = os.path.splitext(file)[0]
                path_csv = os.path.join(root, file)

                try:
                    df = pd.read_csv(path_csv)
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)

                    df["Daily_Return_lineal"] = df["Close"].pct_change()
                    df["Cumulative_Return_lineal"] = (1 + df["Daily_Return_lineal"]).cumprod() - 1

                    df["Daily_Return_log"] = np.log(df["Close"] / df["Close"].shift(1))
                    df["Cumulative_Return_log"] = df["Daily_Return_log"].cumsum()
                    generar_grafico_volumen(ticker, df, tema=tema, carpeta_salida=carpeta_salida)

                    mean_return = df["Daily_Return_lineal"].mean()
                    std_return = df["Daily_Return_lineal"].std()
                    downside_std = df[df["Daily_Return_lineal"] < 0]["Daily_Return_lineal"].std()

                    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else np.nan
                    sortino = (mean_return / downside_std) * np.sqrt(252) if downside_std != 0 else np.nan

                    ratios_summary.append({
                        "Ticker": ticker,
                        "Sharpe Ratio": sharpe,
                        "Sortino Ratio": sortino
                    })

                    rolling_df = pd.DataFrame(index=df.index)
                    if calcular_rolling:
                        for window in ventanas:
                            col_name = f"Cumulative_{window}d"
                            rolling_df[col_name] = (1 + df["Daily_Return_lineal"]).rolling(window).apply(np.prod, raw=True) - 1

                    retornos_bloque = {}
                    if calcular_bloques:
                        for freq in frecuencias:
                            nombre = {
                                "W-FRI": "Retornos Semanales",
                                "M": "Retornos Mensuales",
                                "Y": "Retornos Anuales"
                            }.get(freq, f"Retornos_{freq}")
                            serie = df["Close"].resample(freq).last().pct_change()
                            retornos_bloque[nombre] = serie

                    output_path = os.path.join(carpeta_salida, f"{ticker}.xlsx")
                    with pd.ExcelWriter(output_path) as writer:
                        for tipo in tipos_retornos:
                            df_output = pd.DataFrame({
                                "Daily_Return": df[f"Daily_Return_{tipo}"],
                                "Cumulative_Return": df[f"Cumulative_Return_{tipo}"]
                            })
                            df_output.to_excel(writer, sheet_name=f"Retorno Diario ({tipo})", index=True)

                        if calcular_rolling:
                            rolling_df.to_excel(writer, sheet_name="Retornos Rolling", index=True)
                        if calcular_bloques:
                            for nombre_hoja, serie in retornos_bloque.items():
                                serie.to_frame(name="Return").to_excel(writer, sheet_name=nombre_hoja)

                    print(f"✅ Retornos procesados para {ticker} guardados en {output_path}")

                    for tipo in tipos_retornos:
                        df_output = pd.DataFrame({
                            "Daily_Return": df[f"Daily_Return_{tipo}"],
                            "Cumulative_Return": df[f"Cumulative_Return_{tipo}"]
                        })

                        generar_grafico_retorno_acumulado(
                            ticker, df_output, tema, carpeta_salida,
                            logaritmico=(tipo == "log"),
                            calcular_rolling=calcular_rolling,
                            ventanas=ventanas
                        )

                        df_dd, max_dd, fecha_dd = calcular_drawdown(df_output)
                        generar_grafico_drawdown(ticker, df_dd, tema, carpeta_salida)
                        print(f"📉 Máx. Drawdown de {ticker}: {max_dd:.2%} en {fecha_dd.date()}")

                        generar_qq_plot(ticker, df_output, tema=tema, carpeta_salida=carpeta_salida)

                        jb_stat, jb_p, conclusion = test_jarque_bera(df_output, ticker, carpeta_salida)
                        jb_results.append({
                            "Ticker": ticker,
                            "JB Stat": f"{jb_stat:.2f}",
                            "p-valor": f"{jb_p:.6f}",
                            "Normalidad": "❌ No" if jb_p < 0.05 else "✅ Sí"
                        })

                        generar_histograma_retorno(ticker, df_output, tema, carpeta_salida, bins=60, referencias=referencias_histograma)

                        generar_grafico_autocorrelacion(
                            ticker,
                            df_output,
                            carpeta_salida=carpeta_salida,
                            lags=20,
                            tema=tema
                        )

                except Exception as e:
                    print(f"❌ Error procesando {file}: {e}")

    if len(ratios_summary) > 0:
        exportar_y_graficar_ratios(
            ratios_summary,
            carpeta_salida=carpeta_salida,
            top_n=10,
            tema=tema
        )

    if len(jb_results) > 0:
        generar_tabla_jarque_bera_imagen(
            jb_results,
            carpeta_salida=carpeta_salida,
            tema=tema
        )

    print("\n🏁 Cálculo de retornos y generación de gráficos completados.")
