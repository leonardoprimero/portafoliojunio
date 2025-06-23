import pandas as pd
import numpy as np
import os
from generar_graficos import generar_grafico_retorno_acumulado

def calcular_retornos_diarios_acumulados(
    carpeta_datos_limpios="DatosLimpios",
    carpeta_salida="RetornoDiarioAcumulado",
    tema="normal",
    logaritmico=False,
    calcular_rolling=False,
    ventanas=[5, 22, 252],
    calcular_bloques=False,
    frecuencias=["W-FRI", "M", "Y"]
):
    os.makedirs(carpeta_salida, exist_ok=True)
    print(f"\n📊 Calculando retornos diarios acumulados... (log: {logaritmico}, tema: {tema}, rolling: {calcular_rolling}, bloques: {calcular_bloques})")

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

                    # Rolling acumulado (si querés seguir teniéndolo activable)
                    rolling_df = pd.DataFrame(index=df.index)
                    if calcular_rolling:
                        for window in ventanas:
                            col_name = f"Cumulative_{window}d"
                            rolling_df[col_name] = (1 + df["Daily_Return"]).rolling(window).apply(np.prod, raw=True) - 1
                            df_output[col_name] = rolling_df[col_name]

                    # Retornos por bloque de tiempo real (lo pro de verdad)
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

                    # Guardado
                    output_path = os.path.join(carpeta_salida, f"{ticker}.xlsx")
                    with pd.ExcelWriter(output_path) as writer:
                        df_output[["Daily_Return", "Cumulative_Return"]].to_excel(writer, sheet_name="Retorno Diario", index=True)
                        if calcular_rolling:
                            rolling_df.to_excel(writer, sheet_name="Retornos Rolling", index=True)
                        if calcular_bloques:
                            for nombre_hoja, serie in retornos_bloque.items():
                                serie.to_frame(name="Return").to_excel(writer, sheet_name=nombre_hoja)

                    print(f"✅ Retornos procesados para {ticker} guardados en {output_path}")

                    generar_grafico_retorno_acumulado(
                        ticker,
                        df_output,
                        tema,
                        carpeta_salida,
                        logaritmico=logaritmico,
                        calcular_rolling=calcular_rolling,
                        ventanas=ventanas
                    )

                except Exception as e:
                    print(f"❌ Error procesando {file}: {e}")

    print("\n🏁 Cálculo de retornos y generación de gráficos completados.")
