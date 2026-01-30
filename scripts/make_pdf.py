#!/usr/bin/env python3

from pathlib import Path
from weasyprint import HTML

# Change this to your folder
input_dir = Path("../notebooks")
output_dir = Path("../notebooks")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Convert all .html and .htm files
for html_file in input_dir.glob("*.htm*"):
    output_pdf = output_dir / (html_file.stem + ".pdf")
    print(f"Converting {html_file.name} → {output_pdf.name}")
    HTML(filename=str(html_file)).write_pdf(str(output_pdf))

print("Done.")

