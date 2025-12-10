"""
Build a FAISS index over the E5 embeddings.

Reads:
  - data/embeddings/msmarco_e5_embeddings.npy

Writes:
  - data/index/msmarco_e5_faiss.index
"""

from pathlib import Path

import faiss
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT_DIR / "data" / "embeddings"
INDEX_DIR = ROOT_DIR / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    emb_path = EMB_DIR / "msmarco_e5_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} not found. Run scripts/build_e5_embeddings.py first."
        )

    print(f"Loading embeddings from {emb_path}")
    embeddings = np.load(emb_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    n, d = embeddings.shape
    print(f"Embeddings shape: n={n}, d={d}")

    index = faiss.IndexFlatIP(d)

    print("Adding embeddings to FAISS index...")
    index.add(embeddings)

    print(f"Index ntotal = {index.ntotal}")
    out_path = INDEX_DIR / "msmarco_e5_faiss.index"
    print(f"Saving FAISS index to {out_path}")
    faiss.write_index(index, str(out_path))
    print("Done building FAISS index.")

if __name__ == "__main__":
    main()
