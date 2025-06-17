from fpdf import FPDF
import os
import requests
import pandas as pd

# Rutas a las fuentes DejaVu
FUENTE_REGULAR = "./fonts/DejaVuSans.ttf"
FUENTE_BOLD = "./fonts/DejaVuSans-Bold.ttf"


# Leer los informes desde CSV
df_info = pd.read_csv('informe_empresas.csv')
informes = df_info.to_dict(orient='records')

# Crear PDF y registrar fuente con soporte Unicode
class MyPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.cell(0, 10, "leocaliva.com", 0, 0, "C")

pdf = MyPDF()
pdf.add_font("DejaVu", "", FUENTE_REGULAR, uni=True)
pdf.add_font("DejaVu", "B", FUENTE_BOLD, uni=True)
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("DejaVu", "", 11)

for empresa in informes:
    pdf.add_page()


    # Fondo encabezado (color celeste claro)
    pdf.set_fill_color(230, 240, 255)
    pdf.rect(0, 5, 210, 30, 'F')
    # Logo si hay URL válida
    if pd.notna(empresa['Logo URL']) and empresa['Logo URL']:
        try:
            logo_path = f"temp_logo_{empresa['Ticker']}.png"
            img_data = requests.get(empresa['Logo URL'], timeout=3).content
            with open(logo_path, 'wb') as f:
                f.write(img_data)
            pdf.image(logo_path, 10, 10, 25)  # Logo alineado a y=10
            os.remove(logo_path)
        except Exception as e:
            print(f"⚠️ Error cargando logo de {empresa['Ticker']}: {e}")

    pdf.set_font("DejaVu", "", 24)
    pdf.set_xy(150, 10)  # Misma altura que el logo
    pdf.cell(50, 10, empresa["Ticker"], ln=False)

    # Línea divisoria
    pdf.set_draw_color(100, 100, 100)
    pdf.set_line_width(0.3)
    pdf.line(10, 40, 200, 40)  # Línea horizontal

    # Nombre de empresa
    pdf.set_xy(10, 50)
    pdf.set_font("DejaVu", "", 16)
    pdf.multi_cell(0, 10, empresa['Nombre'])

    # Sector e industria
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, f"Sector: {empresa['Sector']} | Industria: {empresa['Industria']}", ln=True)
    pdf.cell(0, 10, f"Ubicación: {empresa['Ciudad']}, {empresa['País']}", ln=True)

    # Descripción
    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 7, f"Descripción:\n{empresa['Descripción']}")

    # Datos financieros
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 10, "Análisis fundamental:", ln=True)
    pdf.set_font("DejaVu", "", 11)
    pdf.cell(0, 8, f"Market Cap: {empresa['Market Cap']}", ln=True)
    pdf.cell(0, 8, f"P/E Ratio: {empresa['P/E Ratio']}", ln=True)
    pdf.cell(0, 8, f"Dividend Yield: {empresa['Dividend Yield']}", ln=True)
    pdf.cell(0, 8, f"Ingresos Totales: {empresa['Ingresos']}", ln=True)
    pdf.cell(0, 8, f"Ingresos Netos: {empresa['Ingresos Netos']}", ln=True)
    pdf.cell(0, 8, f"Beta: {empresa['Beta']}", ln=True)

pdf.output("informe_empresas.pdf")
print("✅ PDF generado correctamente con soporte Unicode: informe_empresas.pdf")


def footer(self):
    self.set_y(-15)
    self.set_font('DejaVu', '', 8)
    self.cell(0, 10, 'leocaliva.com', 0, 0, 'C')


