from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def build_context_from_chunks(
        chunks: List[Dict[str, Any]],
        max_chars: int = 4000,
    ) -> str:
        """
        Build a context string by concatenating chunk texts until reaching max_chars.
        Each chunk is annotated with an index so you can later ref [1], [2] etc...
        """
        parts = []

        for i, ch in enumerate(chunks, start=1):
            doc_id = ch.get("doc_id")
            chunk_id = ch.get("chunk_id")
            title = ch.get("title") or ch.get("set") or ""
            source_path = ch.get("source_path") or ch.get("url") or ""
            header_bits = [f"[{i}] doc_id={doc_id}, chunk_id={chunk_id}"]
            if title:
                header_bits.append(f"title={title}")
            if source_path:
                header_bits.append(f"source={source_path}")
            header = " | ".join(header_bits)

            text = str(ch.get("chunk_text", ""))
            parts.append(header + "\n" + text + "\n")

        context = "\n".join(parts)

        if len(context) > max_chars:
            context = context[:max_chars]

        return context

def generate_answer_hf(query: str, context: str) -> str:

    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outs[0], skip_special_tokens=True)


