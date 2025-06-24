import pandas as pd
import numpy as np
import os
from generar_graficos import (
    generar_grafico_retorno_acumulado,
    generar_histograma_retorno,
    calcular_drawdown,
    generar_grafico_drawdown,
    exportar_y_graficar_ratios,
    generar_qq_plot,
    test_jarque_bera,
    generar_tabla_jarque_bera_imagen,
    generar_grafico_autocorrelacion
)

def calcular_retornos_diarios_acumulados(
    carpeta_datos_limpios="DatosLimpios",
    carpeta_salida="RetornoDiarioAcumulado",
    tema="normal",
    logaritmico=False,
    calcular_rolling=False,
    ventanas=[5, 22, 252],
    calcular_bloques=False,
    frecuencias=["W-FRI", "ME", "YE"],
    referencias_histograma=None
):
    os.makedirs(carpeta_salida, exist_ok=True)
    print(f"\n📊 Calculando retornos diarios acumulados... (log: {logaritmico}, tema: {tema}, rolling: {calcular_rolling}, bloques: {calcular_bloques})")

    ratios_summary = []
    jb_results = []  # <- NUEVO

    for root, dirs, files in os.walk(carpeta_datos_limpios):
        for file in files:
            if file.endswith(".csv"):
                ticker = os.path.splitext(file)[0]
                path_csv = os.path.join(root, file)

                try:
                    df = pd.read_csv(path_csv)
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)

                    if logaritmico:
                        df["Daily_Return"] = np.log(df["Close"] / df["Close"].shift(1))
                        df["Cumulative_Return"] = df["Daily_Return"].cumsum()
                    else:
                        df["Daily_Return"] = df["Close"].pct_change()
                        df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

                    df_output = df[["Daily_Return", "Cumulative_Return"]].copy()

                    # --- Ratios ---
                    mean_return = df["Daily_Return"].mean()
                    std_return = df["Daily_Return"].std()
                    downside_std = df[df["Daily_Return"] < 0]["Daily_Return"].std()

                    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else np.nan
                    sortino = (mean_return / downside_std) * np.sqrt(252) if downside_std != 0 else np.nan

                    ratios_summary.append({
                        "Ticker": ticker,
                        "Sharpe Ratio": sharpe,
                        "Sortino Ratio": sortino
                    })

                    # Rolling
                    rolling_df = pd.DataFrame(index=df.index)
                    if calcular_rolling:
                        for window in ventanas:
                            col_name = f"Cumulative_{window}d"
                            rolling_df[col_name] = (1 + df["Daily_Return"]).rolling(window).apply(np.prod, raw=True) - 1
                            df_output[col_name] = rolling_df[col_name]

                    # Retornos por bloque
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

                    # Guardar Excel
                    output_path = os.path.join(carpeta_salida, f"{ticker}.xlsx")
                    with pd.ExcelWriter(output_path) as writer:
                        df_output[["Daily_Return", "Cumulative_Return"]].to_excel(writer, sheet_name="Retorno Diario", index=True)
                        if calcular_rolling:
                            rolling_df.to_excel(writer, sheet_name="Retornos Rolling", index=True)
                        if calcular_bloques:
                            for nombre_hoja, serie in retornos_bloque.items():
                                serie.to_frame(name="Return").to_excel(writer, sheet_name=nombre_hoja)

                    print(f"✅ Retornos procesados para {ticker} guardados en {output_path}")

                    # Gráficos
                    generar_grafico_retorno_acumulado(ticker, df_output, tema, carpeta_salida, logaritmico, calcular_rolling, ventanas)

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

                    df_hist = df.copy()
                    df_hist["Daily_Return"] = df["Close"].pct_change()
                    generar_histograma_retorno(ticker, df_hist, tema, carpeta_salida, bins=60, referencias=referencias_histograma)

                    generar_grafico_autocorrelacion(
                        ticker,
                        df_output,
                        carpeta_salida=carpeta_salida,
                        lags=20,
                        tema=tema
                    )


                except Exception as e:
                    print(f"❌ Error procesando {file}: {e}")

    # Exportar tabla de ratios
    if len(ratios_summary) > 0:
        exportar_y_graficar_ratios(
            ratios_summary,
            carpeta_salida=carpeta_salida,
            top_n=10,
            tema=tema
        )

    # Exportar imagen resumen de Jarque-Bera
    if len(jb_results) > 0:
        generar_tabla_jarque_bera_imagen(
            jb_results,
            carpeta_salida=carpeta_salida,
            tema=tema
        )

    print("\n🏁 Cálculo de retornos y generación de gráficos completados.")
