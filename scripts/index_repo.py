import argparse
from pathlib import Path
from app.loaders import load_and_chunk_repo
from app.rag import index_chunks

def main():
    parser = argparse.ArgumentParser(description="Index a local code repo into Chroma (embeddings via Ollama).")
    parser.add_argument("--path", type=str, required=True, help="Path to the repo to index")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the collection before indexing")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")
    
    print(f"Indexing repo: {root}")
    chunks = load_and_chunk_repo(root)
    print(f"Chunks prepared: {len(chunks)}")

    info = index_chunks(chunks, reset=args.reset)
    print("index complete:", info)

if __name__ == "__main__":
    main()
