from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def agregar_marca_agua(fig, logo_path, texto="leocaliva.com", reps_x=8, reps_y=6,
                           alpha_logo=0.10, alpha_text=0.09, size_factor=0.08, font_size=15):
    """
    Marca de agua pattern checkerboard: logo y texto intercalados.
    """
    from PIL import Image
    import numpy as np

    img = Image.open(logo_path).convert("RGBA")
    w, h = img.size
    new_size = (int(w*size_factor), int(h*size_factor))
    img = img.resize(new_size, Image.LANCZOS)
    arr_img = np.asarray(img)
    fig_w, fig_h = fig.bbox.xmax, fig.bbox.ymax

    x_coords = np.linspace(0.07, 0.93, reps_x)
    y_coords = np.linspace(0.10, 0.93, reps_y)
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            if (i + j) % 2 == 0:
                # Logo
                fig.figimage(arr_img,
                    xo=int(fig_w*x)-new_size[0]//2,
                    yo=int(fig_h*y)-new_size[1]//2,
                    alpha=alpha_logo, zorder=9)
            else:
                # Texto
                fig.text(x, y, texto,
                    fontsize=font_size, color="grey", alpha=alpha_text,
                    ha="center", va="center", rotation=30, weight="bold", zorder=9)