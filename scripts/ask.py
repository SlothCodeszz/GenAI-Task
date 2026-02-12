import sys
from app.rag import generate_answer

def main():
    question = "".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit("Usage: python -m scripts.ask \"your question here\"")
    
    out = generate_answer(question, top_k=5, max_tokens=200)

    print("\nANSWER:\n", out["answer"])
    print("\nSOURCES:")
    for s in out["sources"]:
        print("-", s["path"])

if __name__ == "__main__":
    main()
    