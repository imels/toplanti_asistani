"""Markdown benzeri (# başlık, - madde) transkript/özet metnini PDF'e çevirir."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# macOS'ta hazır gelen, Türkçe karakterleri (ğ ş ı ö ü ç İ) destekleyen
# geniş kapsamlı bir Unicode font.
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT_FAMILY = "ArialUnicode"


def _paragraph(pdf: FPDF, height: float, text: str) -> None:
    # multi_cell()'in varsayılanı imleci hücrenin SAĞINA taşır, sola değil.
    # Bunu açıkça belirtmezsek art arda çağrılarda imleç sayfa sonuna kadar
    # kayıp "yeterli yatay alan yok" hatasına yol açıyor (bilinen fpdf2 tuzağı).
    pdf.multi_cell(0, height, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def markdown_to_pdf_bytes(text: str, meeting_title: str = "", meeting_date: str = "") -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(18, 18, 18)
    pdf.add_font(FONT_FAMILY, "", FONT_PATH)
    pdf.set_font(FONT_FAMILY, size=11)

    if meeting_title or meeting_date:
        pdf.set_font(FONT_FAMILY, size=18)
        _paragraph(pdf, 10, meeting_title or "Toplantı")
        if meeting_date:
            pdf.set_font(FONT_FAMILY, size=10)
            pdf.set_text_color(110, 110, 110)
            _paragraph(pdf, 6, meeting_date)
            pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        pdf.set_font(FONT_FAMILY, size=11)

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            pdf.ln(3)
            continue

        if line.startswith("## "):
            pdf.ln(2)
            pdf.set_font(FONT_FAMILY, size=13)
            _paragraph(pdf, 8, line[3:].strip())
            pdf.set_font(FONT_FAMILY, size=11)
        elif line.startswith("# "):
            pdf.set_font(FONT_FAMILY, size=16)
            _paragraph(pdf, 9, line[2:].strip())
            pdf.set_font(FONT_FAMILY, size=11)
            pdf.ln(2)
        elif line.startswith("- "):
            _paragraph(pdf, 7, f"•  {line[2:].strip()}")
        else:
            _paragraph(pdf, 7, line)

    return bytes(pdf.output())
