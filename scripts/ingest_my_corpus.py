from pathlib import Path
from typing import List, Dict

import pandas as pd
import pypdf
from pypdf import PdfReader 

ROOT_DIR = Path(__file__).resolve().parents[1]  # scripts/ -> project root
CORPUS_RAW_DIR = ROOT_DIR / "data" / "my_corpus" / "raw"
CORPUS_PROCESSED_DIR = ROOT_DIR / "data" / "my_corpus" / "processed"
CORPUS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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
        


def ingest_my_corpus() -> pd.DataFrame:

    if not CORPUS_RAW_DIR.exists(): 
        raise FileNotFoundError(f"Raw corpus directory not found: {CORPUS_RAW_DIR}. Create it and add Pdfs/mds/txts.")
    

    records = []

    for path in CORPUS_RAW_DIR.rglob("*"):

        suffix = path.suffix.lower()

        if not path.is_file():
            continue

        if suffix not in [".txt", ".md", ".pdf"]:
            continue

        try: 
            if suffix == ".pdf": 
                raw_text = read_pdf(path)
                doc_type = "pdf"
            elif suffix == ".txt": 
                raw_text = read_txt(path)
                doc_type = "text"
            else: 
                raw_text = read_md(path)
                doc_type = "md"
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
    
    print("\nPreview of first document:")
    print(df.loc[0, "raw_text"][:1000])

    print("Done.")


if __name__ == "__main__":
    main()


            

  