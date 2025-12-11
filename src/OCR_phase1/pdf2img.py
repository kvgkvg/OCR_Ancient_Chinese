import os
import sys
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "Data/sample.pdf"
    output_folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join("output", os.path.basename(pdf_path))
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    os.makedirs(output_folder, exist_ok=True)

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    for page_num in range(1, num_pages + 1):
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num
        )

        img_path = os.path.join(output_folder, f"{page_num:03d}.png")
        images[0].save(img_path, "PNG")
