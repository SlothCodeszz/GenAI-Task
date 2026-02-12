from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Vector index directory
    faiss_dir: Path = Path(__file__).resolve().parent.parent / "artifacts" / "faiss"

    # Ollama models
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2:1b"

    # File filtering
    allowed_exts: tuple[str, ...] = (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml")
    ignore_dirs: tuple[str, ...] = ("venv", ".git", "node_modules", "__pycache__", "dist", "build", "artifacts", ".mypy_cache")

    # Chunking
    chunk_chars: int = 1000
    chunk_overlap: int = 150
    max_file_bytes: int = 300_000


settings = Settings()
settings.faiss_dir.mkdir(parents=True, exist_ok=True)
