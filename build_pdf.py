"""
Build a LinkedIn-ready PDF carousel from slide PNGs.

Usage:
    python3 build_pdf.py

Drop slide1.png ... slide4.png in the same folder as this script.
Output: NyayaGPT_carousel.pdf
"""

from PIL import Image
import os, sys

# ---- Config ----
SLIDE_FILES = ["slide1.png", "slide2.png", "slide3.png", "slide4.png"]
OUTPUT_PDF  = "NyayaGPT_carousel.pdf"
# LinkedIn renders document carousels at the page's native aspect ratio.
# Your slides are 1080x1350 (4:5 portrait) — perfect for LinkedIn.
# We'll keep them at native resolution. DPI is metadata-only here.
DPI = 150

def build_pdf():
    here = os.path.dirname(os.path.abspath(__file__))
    images = []

    for fname in SLIDE_FILES:
        path = os.path.join(here, fname)
        if not os.path.isfile(path):
            print(f"ERROR: missing {fname} in {here}")
            sys.exit(1)
        img = Image.open(path)
        # PDF needs RGB (no alpha channel)  
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)
        print(f"  loaded {fname}  ({img.size[0]}x{img.size[1]})")

    out_path = os.path.join(here, OUTPUT_PDF)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        format="PDF",
        resolution=DPI,
        title="NyayaGPT — Domain-specialized Indian Legal LLM",
        author="Gaurav Garwal",
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nWrote {out_path}  ({size_mb:.2f} MB, {len(images)} pages)")
    if size_mb > 100:
        print("WARNING: LinkedIn rejects PDFs >100 MB. Reduce image resolution.")

if __name__ == "__main__":
    build_pdf()