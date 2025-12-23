from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import pypdf
from pypdf import PdfReader
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io

# Use pytesseract instead of PaddleOCR for lighter memory footprint
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Image OCR will be disabled.")

# Enable HEIC/HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Optional math OCR support
try:
    from pix2text import Pix2Text
    MATH_OCR_AVAILABLE = True
except ImportError:
    MATH_OCR_AVAILABLE = False
    print("Warning: pix2text not available. Math equation conversion will be disabled.") 

ROOT_DIR = Path(__file__).resolve().parents[1]  # scripts/ -> project root
CORPUS_RAW_DIR = ROOT_DIR / "data" / "my_corpus" / "raw"
CORPUS_PROCESSED_DIR = ROOT_DIR / "data" / "my_corpus" / "processed"
CORPUS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# OCR Configuration
ENABLE_PDF_IMAGE_EXTRACTION = True  # Re-enabled with lighter Tesseract
ENABLE_MATH_OCR = False  # Disabled to reduce memory usage
ENABLE_IMAGE_PROCESSING = True  # Re-enabled with lighter Tesseract
IMAGE_FORMATS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".heic", ".heif"]

# Tesseract configuration for better accuracy
# PSM modes: 3=Fully automatic page segmentation (default), 6=Uniform block of text
TESSERACT_CONFIG = '--psm 3'  # PSM 3 = auto page segmentation

_math_ocr_engine = None

def get_ocr_engine():
    """Check if Tesseract OCR is available."""
    if not TESSERACT_AVAILABLE:
        raise ImportError("pytesseract is not installed. Install with: pip install pytesseract")
    return pytesseract

def get_math_ocr_engine():
    """Lazy-load Pix2Text engine for math equations."""
    global _math_ocr_engine
    if _math_ocr_engine is None and MATH_OCR_AVAILABLE and ENABLE_MATH_OCR:
        _math_ocr_engine = Pix2Text(
            analyzer_config=dict(model_name='mfd'),
            text_formula_ocr=True
        )
    return _math_ocr_engine

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_md(path: Path) -> str:
    # for now treat md as plain text
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path):
    if pypdf is None:
        raise ImportError("pypdf is required to read PDF files. Please install it via 'pip install pdfplumber'.")

    reader = PdfReader(str(path))
    texts = []

    for page in reader.pages:
        if page:
            t = page.extract_text()
        else:
            t = ""
        texts.append(t)
    return "\n".join(texts)

def read_image(path: Path) -> str:
    """Extract text from image using Tesseract OCR."""
    ocr = get_ocr_engine()

    try:
        # Load and convert image (handles HEIC and all formats via PIL)
        img = Image.open(str(path))

        # Convert to RGB if necessary (HEIC often needs this)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        # For HEIC files, pytesseract needs a temporary file in a supported format
        if path.suffix.lower() in ['.heic', '.heif']:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = Path(tmp.name)
                img.save(tmp_path, 'PNG')
            try:
                text = ocr.image_to_string(str(tmp_path), lang='eng', config=TESSERACT_CONFIG)
            finally:
                tmp_path.unlink()  # Clean up temp file
        else:
            # Use pytesseract directly for other formats
            text = ocr.image_to_string(img, lang='eng', config=TESSERACT_CONFIG)

        return text.strip()
    except Exception as e:
        print(f"Error performing OCR on {path}: {e}")
        return ""

def extract_images_from_pdf(path: Path) -> List[Tuple[int, Image.Image]]:
    """Extract embedded images from PDF pages."""
    images = []
    try:
        reader = PdfReader(str(path))
        for page_num, page in enumerate(reader.pages):
            if hasattr(page, 'images'):
                for image_file_object in page.images:
                    try:
                        img = image_file_object.image
                        images.append((page_num, img))
                    except Exception as e:
                        print(f"Warning: Could not extract image from page {page_num}: {e}")
    except Exception as e:
        print(f"Error extracting images from PDF {path}: {e}")
    return images

def read_pdf_with_images(path: Path) -> str:
    """Enhanced PDF reader with OCR on embedded images."""
    text_parts = [read_pdf(path)]

    if ENABLE_PDF_IMAGE_EXTRACTION and TESSERACT_AVAILABLE:
        images = extract_images_from_pdf(path)
        ocr = get_ocr_engine()

        for page_num, img in images:
            ocr_text = ""
            try:
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')

                # Use pytesseract with configuration
                ocr_text = ocr.image_to_string(img, lang='eng', config=TESSERACT_CONFIG).strip()
            except Exception as e:
                print(f"Warning: OCR failed for image on page {page_num}: {e}")

            if ocr_text:
                text_parts.append(f"\n[Image from page {page_num}]\n{ocr_text}")

    return "\n\n".join(text_parts)


def ingest_my_corpus() -> pd.DataFrame:

    if not CORPUS_RAW_DIR.exists(): 
        raise FileNotFoundError(f"Raw corpus directory not found: {CORPUS_RAW_DIR}. Create it and add Pdfs/mds/txts.")
    

    records = []

    for path in CORPUS_RAW_DIR.rglob("*"):

        suffix = path.suffix.lower()

        if not path.is_file():
            continue

        if suffix not in [".txt", ".md", ".pdf"] + IMAGE_FORMATS:
            continue

        # Skip image files if image processing is disabled
        if suffix in IMAGE_FORMATS and not ENABLE_IMAGE_PROCESSING:
            print(f"Skipping image file (OCR disabled): {path.name}")
            continue

        try:
            if suffix == ".pdf":
                raw_text = read_pdf_with_images(path)
                doc_type = "pdf"
            elif suffix == ".txt":
                raw_text = read_txt(path)
                doc_type = "text"
            elif suffix == ".md":
                raw_text = read_md(path)
                doc_type = "md"
            elif suffix in IMAGE_FORMATS:
                raw_text = read_image(path)
                doc_type = "image"
            else:
                continue
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        raw_text = raw_text.strip()
        if not raw_text:
            continue
            
        title = path.stem

        records.append(
            {
                "source_path": str(path.relative_to(CORPUS_RAW_DIR)),
                "title": title,
                "raw_text": raw_text,
                "doc_type": doc_type
            }
        )

    df = pd.DataFrame.from_records(records)
    df = df.reset_index(drop=True)
    df["doc_id"] = df.index.astype(int)

    return df


def main() -> None:
    df = ingest_my_corpus()
    out_path = CORPUS_PROCESSED_DIR / "personal_documents.parquet"
    print(f"Saving {len(df):,} documents to {out_path}")
    df.to_parquet(out_path, index=False)

    # print(df.head())

    if len(df) > 0:
        print("\nPreview of first document:")
        print(df.loc[0, "raw_text"][:1000])
    else:
        print("\nNo documents were successfully processed.")

    print("Done.")


if __name__ == "__main__":
    main()


            

  