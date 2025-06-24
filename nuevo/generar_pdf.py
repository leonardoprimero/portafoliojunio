from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import black, white
from reportlab.lib.utils import ImageReader
from datetime import datetime
import os
import glob

def generar_portada(c, tickers, width, height):
    c.setFillColor(white)
    c.rect(0, 0, width, height, fill=1)

    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width / 2, height - 5 * cm, "Informe por Activos")

    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - 6 * cm, "Estudio Martinez.")

    fecha_actual = datetime.today().strftime("%d/%m/%Y")
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - 7 * cm, f"Fecha: {fecha_actual}")

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

def generar_pdf_informe_por_activos(carpeta_imagenes="RetornoDiarioAcumulado", nombre_salida="informe_por_activos.pdf"):
    c = canvas.Canvas(nombre_salida, pagesize=A4)
    width, height = A4

    tickers = sorted(list(set([os.path.basename(f).split("_")[0] for f in os.listdir(carpeta_imagenes) if f.endswith(".png")])))

    generar_portada(c, tickers, width, height)
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

    c.save()
    print(f"📄 PDF generado en: {nombre_salida}")
