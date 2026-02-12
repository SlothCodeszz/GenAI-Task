from __future__ import annotations
import re 
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable
from app.settings import settings

@dataclass(frozen=True)
class Chunk:
    text: str
    path: str
    ext: str
    chunk_id: str

def iter_code_files(root: Path) -> Iterable[Path]:
    """
    Yield allowed code files under root, skipping ignored directories.
    """
    root = root.resolve()
    for p in root.rglob("*"):
        if p.is_dir():
            continue

        parts_lower = {part.lower() for part in p.parts}
        if any(d.lower() in parts_lower for d in settings.ignore_dirs):
            continue

        if p.suffix.lower() not in settings.allowed_exts:
            continue
        
        try:
            if p.stat().st_size > settings.max_file_bytes:
                continue
        except OSError:
            continue

        yield p

def read_text(path: Path) -> str:
    """
    Read the file as text with robust fallback.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
    
def _fixed_chunk(text: str, chunk_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    n = len(text)
    if n == 0: return [] 
    while start < n:
        end = min(start + chunk_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
        return chunks
    
def _python_chunks(text: str) -> list[str]:
    """
    Simple python aware chunking: split by top level class boundries where possible and cap to fix size chunks to avoid huge pieces.
    """

    lines = text.splitlines(keepends=True)
    boundary = re.compile(r"^\s*(def|class)\s+\w+", re.IGNORECASE)

    blocks: list[str] = []
    cur: list[str] = []

    for line in lines:
        if boundary.match(line) and cur:
            blocks.append("".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur: 
        blocks.append("".join(cur))

    # Cap blocks to find chunk size
    out: list[str] = []
    for b in blocks:
        if len(b) <= settings.chunk_chars:
            out.append(b)
        else:
            out.extend(_fixed_chunk(b, settings.chunk_chars, settings.chunk_overlap))
    return out

def chunk_file_text(text: str, suffix: str):
    """
    Split file text into chunks.
    Always returns a list — never None.
    """

    if not text or not text.strip():
        return []

    return _fixed_chunk(
        text,
        settings.chunk_chars,
        settings.chunk_overlap
    ) or []

def load_and_chunk_repo(root: Path) -> list[Chunk]:
    """
    Load files and return chunks with metadata suitable for embedding.
    """
    chunks: list[Chunk] = []
    for fp in iter_code_files(root):
        txt = read_text(fp)
        if not txt.strip():
            continue

        pieces = chunk_file_text(txt, fp.suffix)

        rel_path = str(fp.relative_to(root.resolve())).replace("\\", "/")
        
        if not pieces:
            continue

        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            
            cid = f"{rel_path}::chunk{i}"
            chunks.append(
                Chunk(
                    text=piece,
                    path=rel_path,
                    ext=fp.suffix.lower(),
                    chunk_id=cid,
                )
            )
    return chunks