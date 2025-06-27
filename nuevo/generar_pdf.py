from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import black, white
from reportlab.lib.utils import ImageReader
from datetime import datetime
import os
import glob



def agregar_imagen(c, img_path, width, height, numero_pagina, path_fondo="datosgenerales/hojaMembretada.jpg"):
    try:
        from reportlab.lib.utils import ImageReader
        if path_fondo:
            fondo = ImageReader(path_fondo)
            c.drawImage(fondo, 0, 0, width=width, height=height)
        img = ImageReader(img_path)
        iw, ih = img.getSize()
        aspect = iw / ih
        img_width = width - 4 * cm
        img_height = img_width / aspect
        if img_height > height - 4 * cm:
            img_height = height - 4 * cm
            img_width = img_height * aspect
        c.drawImage(img, (width - img_width) / 2, (height - img_height) / 2, width=img_width, height=img_height)
    except Exception as e:
        print(f"⚠️ No se pudo agregar {img_path}: {e}")
    agregar_pie_de_pagina(c, width, height, numero_pagina)
    c.showPage()


#path_fondo="datosgenerales/hojaMembretada.jpg"
#path_fondo=None
def generar_portada(c, titulo, width, height, subtitulo=None, path_fondo="datosgenerales/hojaMembretada.jpg"):
    try:
        if path_fondo:
            from reportlab.lib.utils import ImageReader
            fondo = ImageReader(path_fondo)
            c.drawImage(fondo, 0, 0, width=width, height=height)
        else:
            from reportlab.lib.colors import white
            c.setFillColor(white)
            c.rect(0, 0, width, height, fill=1)
    except Exception as e:
        print(f"⚠️ No se pudo cargar la hoja membretada: {e}")

    from reportlab.lib.colors import black
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width / 2, height - 5 * cm, titulo)
    if subtitulo:
        c.setFont("Helvetica", 14)
        c.drawCentredString(width / 2, height - 6 * cm, subtitulo)
        y = height - 7 * cm
    else:
        y = height - 6 * cm

    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, y, "Estudio Martinez.")

    fecha_actual = datetime.today().strftime("%d/%m/%Y")
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, y - 1 * cm, f"Fecha: {fecha_actual}")

    c.setFont("Helvetica", 10)
    c.drawCentredString(width / 2, 2.7 * cm, "Leonardo Caliva")
    c.drawCentredString(width / 2, 2.2 * cm, "Página: leocaliva.com")
    c.linkURL("https://leocaliva.com", (width / 2 - 3 * cm, 2.1 * cm, width / 2 + 3 * cm, 2.4 * cm))
    c.setFont("Times-BoldItalic", 12)
    texto = "Leonardo I (a.k.a. leonardoprimero)"
    c.drawCentredString(width / 2, 1.6 * cm, texto)
    c.linkURL("https://github.com/leonardoprimero", (width / 2 - 4 * cm, 1.5 * cm, width / 2 + 4 * cm, 1.8 * cm))


def agregar_pie_de_pagina(c, width, height, numero_pagina):
    c.setFont("Helvetica", 9)
    c.drawCentredString(width / 2, 1.2 * cm, "leocaliva.com")
    c.drawRightString(width - 2 * cm, 1.2 * cm, f"Página {numero_pagina}")

# def agregar_imagen(c, img_path, width, height, numero_pagina):
#     try:
#         img = ImageReader(img_path)
#         iw, ih = img.getSize()
#         aspect = iw / ih
#         img_width = width - 4 * cm
#         img_height = img_width / aspect
#         if img_height > height - 4 * cm:
#             img_height = height - 4 * cm
#             img_width = img_height * aspect
#         c.drawImage(img, (width - img_width) / 2, (height - img_height) / 2, width=img_width, height=img_height)
#     except Exception as e:
#         print(f"⚠️ No se pudo agregar {img_path}: {e}")
#     agregar_pie_de_pagina(c, width, height, numero_pagina)
#     c.showPage()

def generar_pdf_informe_por_activos(carpeta_imagenes="RetornoDiarioAcumulado", nombre_salida="informe_por_activos.pdf"):
    c = canvas.Canvas(nombre_salida, pagesize=A4)
    width, height = A4
    tickers = sorted(list(set([os.path.basename(f).split("_")[0] for f in os.listdir(carpeta_imagenes) if f.endswith(".png")])))
    generar_portada(c, "Informe por Activos", width, height)
    c.showPage()
    numero_pagina = 1
    for ticker in tickers:
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(width / 2, height - 2.5 * cm, f"Análisis del activo: {ticker}")
        imagenes = sorted(glob.glob(f"{carpeta_imagenes}/{ticker}_*.png"))
        y = height - 4 * cm
        for img_path in imagenes:
            try:
                img = ImageReader(img_path)
                iw, ih = img.getSize()
                aspect = iw / ih
                img_width = width - 4 * cm
                img_height = img_width / aspect
                if y - img_height < 2.5 * cm:
                    agregar_pie_de_pagina(c, width, height, numero_pagina)
                    c.showPage()
                    numero_pagina += 1
                    y = height - 3 * cm
                    c.setFont("Helvetica-Bold", 20)
                    c.drawCentredString(width / 2, height - 2.5 * cm, f"Análisis del activo: {ticker}")
                c.drawImage(img, 2 * cm, y - img_height, width=img_width, height=img_height)
                y -= img_height + 1.5 * cm
            except Exception as e:
                print(f"⚠️ No se pudo agregar {img_path}: {e}")
        agregar_pie_de_pagina(c, width, height, numero_pagina)
        c.showPage()
        numero_pagina += 1
    comparative_graph_path = os.path.join(carpeta_imagenes, "retorno_comparado_bloomberg_dark.png")
    if os.path.exists(comparative_graph_path):
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(width / 2, height - 2.5 * cm, "Retorno Acumulado Comparado")
        agregar_imagen(c, comparative_graph_path, width, height, numero_pagina)
        numero_pagina += 1
    c.save()
    print(f"📄 PDF generado en: {nombre_salida}")

def generar_pdf_informe_correlaciones(nombre_salida="informe_correlaciones.pdf", carpeta_correlaciones="Correlaciones", carpeta_rolling="CorrelacionesRolling", pares_especificos=None, carpeta_pca="PCA"):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    import os
    import glob

    c = canvas.Canvas(nombre_salida, pagesize=A4)
    width, height = A4
    generar_portada(c, "Informe de Correlaciones", width, height)
    c.showPage()
    numero_pagina = 1

    nombres_agregados = set()

    # 📊 Matriz de correlaciones (evita duplicados)
    graficos_correlacion = sorted(glob.glob(f"{carpeta_correlaciones}/*correlacion*png"))
    for img_path in graficos_correlacion:
        nombre_archivo = os.path.basename(img_path)
        if nombre_archivo not in nombres_agregados:
            agregar_imagen(c, img_path, width, height, numero_pagina)
            numero_pagina += 1
            nombres_agregados.add(nombre_archivo)

    # 📈 Heatmap sectorial
    graficos_sectoriales = sorted(glob.glob(f"{carpeta_correlaciones}/heatmap_correlacion_sectores*.png"))
    for img_path in graficos_sectoriales:
        nombre_archivo = os.path.basename(img_path)
        if nombre_archivo not in nombres_agregados:
            agregar_imagen(c, img_path, width, height, numero_pagina)
            numero_pagina += 1
            nombres_agregados.add(nombre_archivo)

    # 📉 Rolling general
    rolling_generales = sorted(glob.glob(f"{carpeta_rolling}/correlaciones_rolling_lineas*topvar*.png"))
    for img_path in rolling_generales:
        nombre_archivo = os.path.basename(img_path)
        if nombre_archivo not in nombres_agregados:
            agregar_imagen(c, img_path, width, height, numero_pagina)
            numero_pagina += 1
            nombres_agregados.add(nombre_archivo)

    # 🎯 Pares específicos
    if pares_especificos:
        for idx, par in enumerate(pares_especificos):
            img_path = os.path.join(carpeta_rolling, f"correlaciones_rolling_lineas_pearson_60d_bloomberg_dark_parte_{idx}.png")
            if os.path.exists(img_path):
                nombre_archivo = os.path.basename(img_path)
                if nombre_archivo not in nombres_agregados:
                    agregar_imagen(c, img_path, width, height, numero_pagina)
                    numero_pagina += 1
                    nombres_agregados.add(nombre_archivo)

    # 🧬 PCA
    img_pca = os.path.join(carpeta_pca, "pca_retorno_2D.png")
    if os.path.exists(img_pca):
        nombre_archivo = os.path.basename(img_pca)
        if nombre_archivo not in nombres_agregados:
            agregar_imagen(c, img_pca, width, height, numero_pagina)
            numero_pagina += 1

    c.save()
    print(f"📄 PDF de correlaciones generado en: {nombre_salida}")

def generar_pdf_backtesting(carpeta_imagenes, nombre_salida, titulo="Informe Backtesting Portafolio", subtitulo=None, path_fondo="datosgenerales/hojaMembretada.jpg"):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    import os, glob
    c = canvas.Canvas(nombre_salida, pagesize=A4)
    width, height = A4
    # Portada
    generar_portada(c, titulo, width, height, subtitulo=subtitulo, path_fondo=path_fondo)
    c.showPage()
    numero_pagina = 1

    # Agregar todas las imágenes PNG de la carpeta
    imagenes = sorted(glob.glob(os.path.join(carpeta_imagenes, "*.png")))
    for img_path in imagenes:
        agregar_imagen(c, img_path, width, height, numero_pagina, path_fondo=path_fondo)
        numero_pagina += 1
    c.save()
    print(f"📄 PDF de backtesting generado en: {nombre_salida}")