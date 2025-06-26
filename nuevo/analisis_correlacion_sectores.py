import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- CONFIGURÁ ESTOS PATHS SI HACE FALTA ---
MATRIZ_PATH = "./Correlaciones/matriz_correlacion_pearson_clusterizada.xlsx"
SECTORES_PATH = "./datosgenerales/sectores.csv"
OUT_DIR = "./Correlaciones"
TEMA = "YlGnBu"   # TEMA CLARO por defecto ("Blues", "Spectral", "coolwarm", etc.)

def cargar_matriz_correlacion(path=MATRIZ_PATH):
    return pd.read_excel(path, index_col=0)

def cargar_sectores(path=SECTORES_PATH):
    df = pd.read_csv(path)
    return dict(zip(df['Ticker'], df['Sector']))

def matriz_correlacion_sectorial(matriz, mapa_ticker_sector):
    sectores = sorted(list(set(mapa_ticker_sector.values())))
    sector_mat = pd.DataFrame(index=sectores, columns=sectores, dtype=float)

    for s1 in sectores:
        for s2 in sectores:
            tickers1 = [t for t, s in mapa_ticker_sector.items() if s == s1 and t in matriz.index]
            tickers2 = [t for t, s in mapa_ticker_sector.items() if s == s2 and t in matriz.columns]
            # Evitar pares vacíos
            if not tickers1 or not tickers2:
                sector_mat.loc[s1, s2] = np.nan
                continue
            vals = matriz.loc[tickers1, tickers2].values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                sector_mat.loc[s1, s2] = np.mean(vals)
            else:
                sector_mat.loc[s1, s2] = np.nan
    return sector_mat

def graficar_heatmap_sectorial(sector_mat, out_dir=OUT_DIR, tema=TEMA):
    os.makedirs(out_dir, exist_ok=True)
    plt.close("all")
    # TEMA CLARO profesional
    
    cmap = tema  # ahora el tema es un colormap de matplotlib
    sns.set(font_scale=1.1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        sector_mat,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        linecolor="#ddd",
        cbar_kws={"shrink": .7, "label": "Correlación media"},
        ax=ax
    )
    ax.set_title("Correlación Promedio entre Sectores", fontsize=19, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"heatmap_correlacion_sectores_{tema}.png")
    plt.savefig(out_path, dpi=320)
    plt.close()
    print(f"🖼️ Heatmap sectorial guardado en {out_path}")

def guardar_excel_sectorial(sector_mat, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "correlacionesporSectores.xlsx")
    sector_mat.to_excel(out_path)
    print(f"📁 Excel guardado en {out_path}")

def ranking_pares_sectores(sector_mat, top_n=10):
    df = sector_mat.stack().reset_index()
    df.columns = ["Sector 1", "Sector 2", "Correlacion"]
    df = df[df["Sector 1"] != df["Sector 2"]]  # Excluir diagonal
    df = df.dropna().sort_values(by="Correlacion", ascending=False)
    return df.head(top_n)

def main():
    matriz = cargar_matriz_correlacion()
    mapa = cargar_sectores()
    sector_mat = matriz_correlacion_sectorial(matriz, mapa)
    graficar_heatmap_sectorial(sector_mat)
    guardar_excel_sectorial(sector_mat)
    top_pairs = ranking_pares_sectores(sector_mat)
    print("\n🔝 Top pares de sectores más correlacionados:")
    print(top_pairs.to_string(index=False))

if __name__ == "__main__":
    main()
