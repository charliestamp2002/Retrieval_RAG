from pathlib import Path

import faiss
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT_DIR / "data" / "my_corpus" / "embeddings"
INDEX_DIR = ROOT_DIR / "data" / "my_corpus" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    emb_path = EMB_DIR / "personal_e5_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} not found. Run scripts/build_e5_embeddings_my_corpus.py first."
        )

    embeddings = np.load(emb_path)
    n, d = embeddings.shape
    print(f"Loaded embeddings: n={n}, d={d}")

    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    print(f"Index ntotal = {index.ntotal}")

    out_path = INDEX_DIR / "personal_e5_faiss.index"
    faiss.write_index(index, str(out_path))
    print(f"Saved FAISS index to {out_path}")


if __name__ == "__main__":
    main()