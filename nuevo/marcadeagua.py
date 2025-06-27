from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def marca_agua_logo_central(ax, logo_path, alpha_logo=0.13, scale=0.7):
    """
    Inserta el logo como marca de agua GRANDE y centrado en el área de datos del ax.
    """
    img = Image.open(logo_path).convert("RGBA")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]

    cx = (xlim[0] + xlim[1]) / 2
    cy = (ylim[0] + ylim[1]) / 2

    logo_w = x_span * scale
    logo_h = logo_w * (img.height / img.width)

    extent = [
        cx - logo_w/2, cx + logo_w/2,
        cy - logo_h/2, cy + logo_h/2
    ]
    ax.imshow(img, extent=extent, aspect='auto', alpha=alpha_logo, zorder=0)

def watermark_text_fade(ax, texto="leocaliva.com", reps_x=8, reps_y=5, alpha_max=0.17, font_size=19, angle=45):
    """
    Pattern de texto 'leocaliva.com' en ángulo, opacidad máxima en el centro y desvanecido hacia los bordes.
    """
    x_coords = np.linspace(0.05, 0.95, reps_x)
    y_coords = np.linspace(0.05, 0.95, reps_y)

    cx, cy = 0.5, 0.5  # centro (en coordenadas ax.transAxes)
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            dist = np.sqrt((x-cx)**2 + (y-cy)**2) / np.sqrt(cx**2 + cy**2)
            alpha = alpha_max * (1 - dist)
            ax.text(
                x, y, texto,
                fontsize=font_size, color="grey", alpha=alpha,
                ha="center", va="center", rotation=angle, weight="bold", zorder=1,
                transform=ax.transAxes, clip_on=True
            )
def marcas_agua_full(
    ax,
    path_logo="datosgenerales/logo1.png",
    alpha_logo=0.13,
    scale_logo=0.7,
    texto="leocaliva.com",
    reps_x=9,
    reps_y=6,
    alpha_text=0.17,
    font_size=19,
    angle=45
):
    """
    Aplica marca de agua de logo centrado + pattern de texto fade en el axis dado.
    """
    marca_agua_logo_central(ax, path_logo, alpha_logo=alpha_logo, scale=scale_logo)
    watermark_text_fade(ax, texto=texto, reps_x=reps_x, reps_y=reps_y,
                        alpha_max=alpha_text, font_size=font_size, angle=angle)
