import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generar_grafico_retorno_acumulado(ticker, df, tema="normal", carpeta_salida="RetornoDiarioAcumulado", logaritmico=False, calcular_rolling=False, ventanas=None):
    os.makedirs(carpeta_salida, exist_ok=True)
    if ventanas is None:
        ventanas = []

    # 🎨 Configuración visual según tema
    if tema == "dark":
        plt.style.use("dark_background")
        sns.set_palette("viridis")
    elif tema == "vintage":
        plt.style.use("classic")
        sns.set_palette("deep")
    elif tema == "normal":
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("pastel")
    elif tema == "modern":
        plt.style.use("ggplot")
        sns.set_palette("coolwarm")
    else:
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("pastel")

    # 📈 Crear figura
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df.index, y=df["Cumulative_Return"], label="Retorno Acumulado")

    # 📊 Agregar líneas de rolling si corresponde
    if calcular_rolling:
        for window in ventanas:
            col_name = f"Cumulative_{window}d"
            if col_name in df.columns:
                sns.lineplot(x=df.index, y=df[col_name], label=f"{window} ruedas")

    # 🏷️ Título del gráfico
    titulo = f"Retorno Diario Acumulado Logarítmico para {ticker}" if logaritmico else f"Retorno Diario Acumulado para {ticker}"
    plt.title(titulo, fontsize=16)
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Retorno Acumulado", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 💾 Guardar gráfico
    output_path = os.path.join(carpeta_salida, f"{ticker}_retorno_acumulado_{'log' if logaritmico else 'simple'}_{tema}.png")
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Gráfico de retorno acumulado para {ticker} ({'logarítmico' if logaritmico else 'simple'}, tema: {tema}) guardado en {output_path}")
