from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import ollama
from app.settings import settings
from app.loaders import Chunk

INDEX_DIR = settings.faiss_dir
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "meta.jsonl"

def embed_text(text: str) -> np.ndarray:
    """Return embedding vector as float32 numpy array."""
    resp = ollama.embeddings(model=settings.embed_model, prompt=text)
    vec = np.array(resp["embedding"], dtype=np.float32)
    return vec


def _load_meta() -> list[dict[str, Any]]:
    if not META_PATH.exists():
        return []
    items: list[dict[str, Any]] = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def _save_meta(items: list[dict[str, Any]]) -> None:
    with META_PATH.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def index_chunks(chunks: list[Chunk], reset: bool = False) -> dict[str, Any]:
    """
    Build/overwrite a FAISS index from chunks.
    This is CPU-friendly and avoids DB dependencies.
    """
    t0 = time.time()

    if reset:
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        if META_PATH.exists():
            META_PATH.unlink()

    if not chunks:
        return {"chunks_indexed": 0, "seconds": 0.0, "index_dir": str(INDEX_DIR)}

    # Embed one by one (low RAM)
    vecs: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    for c in chunks:
        v = embed_text(c.text)
        vecs.append(v)
        meta.append(
            {
                "id": c.chunk_id,
                "path": c.path,
                "ext": c.ext,
                "text": c.text,
            }
        )

    mat = np.vstack(vecs)  # shape: (n, d)
    d = mat.shape[1]

    # Cosine similarity via inner product on normalised vectors
    mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

    import faiss  # local import to avoid install errors

    index = faiss.IndexFlatIP(d)
    index.add(mat_norm)

    faiss.write_index(index, str(INDEX_PATH))
    _save_meta(meta)

    return {
        "chunks_indexed": len(chunks),
        "seconds": round(time.time() - t0, 2),
        "index_dir": str(INDEX_DIR),
    }


def retrieve(question: str, top_k: int = 3) -> list[dict[str, Any]]:
    import faiss

    if not INDEX_PATH.exists() or not META_PATH.exists():
        return []

    index = faiss.read_index(str(INDEX_PATH))
    meta = _load_meta()

    actual_k = min(top_k, len(meta))
    if actual_k == 0: return []

    q = embed_text(question)
    q = q / (np.linalg.norm(q) + 1e-12)
    q = q.reshape(1, -1)

    scores, idxs = index.search(q, actual_k)

    out: list[dict[str, Any]] = []
    for rank, i in enumerate(idxs[0]):
        if i < 0 or i >= len(meta):
            continue
        item = meta[i]
        out.append(
            {
                "rank": rank,
                "score": float(scores[0][rank]),
                "meta": {"path": item["path"], "ext": item["ext"], "id": item["id"]},
                "text": item["text"],
            }
        )
    return out

def generate_answer(question: str, top_k: int =3, max_tokens: int = 200) -> dict[str, Any]:
    """
    Retrieve context from teh FAISS index and generate a grounded answer using Ollama.
    Returns answer + citations(file paths) + retrieved snippets.
    """
    hits = retrieve(question, top_k=top_k)

    if not hits:
        return {
            "answer": "I couldn't find relevant context in the indexed repository for that question.",
            "sources": [],
            "retrieved": [],
        }

    context_blocks: list[str] = []
    sources: list[dict[str, Any]] = []

    for h in hits:
        path = h["meta"]["path"]
        snippet = h["text"]
        context_blocks.append(f"[SOURCE: {path}]\n{snippet}\n")
        sources.append({"path": path, "snippet": snippet[:300]})

    system_rules = (
        "You are a code documentation assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the context does not contain the answer, say you cannot find it.\n"
        "Be concise and technical.\n"
        "Cite sources by file path (e.g., app/loaders.py) when relevant.\n"
    )

    context_text = "\n".join(context_blocks)
 
    prompt = (
        system_rules
        + "\n\nCONTEXT:\n"
        + context_text
        + "\n\nQUESTION:\n"
        + question
        + "\n\nANSWER:"
    )

    resp = ollama.generate(model=settings.llm_model, prompt=prompt, options={"num_predict": max_tokens, "temperature": 0.2}, stream=False,)

    answer_text = resp.get("response", "").strip()

    return {
        "answer": answer_text,
        "sources": sources,
        "retrieved": hits,
    }

print("rag.py loaded OK; generate_answer exists:", "generate_answer" in globals())
