Code Documentation Assistant: RAG Engineering Assignment

---

Summary:-

This project implements a lightweight, local-first approached Retrieval-Augmented Generation (RAG) pipeline designed to ingest, index and query a code repository. The system prioritises retrieval correctness and groundedness over UI polishing, providing a technical assistant that cites its sources directly from the local codebase.

---

Architectural Philosophy:-

I followed a modular "Loader → Chunker → Embedder → Retriever → Generator" design pattern. This separation of concerns ensures that each component can be independently tested, optimised, or replaced as the system scales.

Key Design Principles:

• Deterministic Retrieval: Using explicit metadata like file paths and chunk IDs to ensure 1:1 traceability between answers and code.

• Hardware-Aware Engineering: Adapted the pipeline to run reliably on constrained local hardware by selecting lightweight models and optimised vector search.

• Local-First Resilience: Zero external API dependencies using Ollama to ensure 100% uptime and data privacy during the evaluation phase.

---

Technical Decisions & Trade-offs:-

1. Vector Store: FAISS (Facebook AI Similarity Search)

• Decision: Pivoted from ChromaDB to FAISS for local persistence.

• Reasoning: FAISS is a production-standard library for efficient similarity search. It offers a smaller footprint for local CLI tools while maintaining high performance for in-memory queries.

2. Model Selection: Small & Semantic

• Embeddings: all-minilm by Ollama. Chosen for its stability on low-memory systems and its balance between speed and semantic retrieval accuracy.

• LLM: llama3.2:1b. Small enough to run on CPU/RAM constraints while providing sufficient reasoning capability when grounded by a high-quality retrieval context.

3. Strategy: Quality > Complexity

• Chunking: Implemented a fixed-size overlapping strategy (400 chars / 50 overlap). This ensures context is preserved across chunk boundaries without exceeding the embedding model's context window.

• Grounding: The system prompt explicitly instructs the model to refuse answering if the context is insufficient, effectively mitigating hallucinations.

---

Azure-Native Productionisation Strategy:-

To transition this from a local prototype to an enterprise-grade service, I would implement the following Azure architecture:

• Vector Database: Migrate from local FAISS to Azure AI Search. Provide managed scalability, hybrid search like combining vectors with keywords and enterprise-grade security.

• Orchestration: Wrapping the RAG logic in a FastAPI service deployed on Azure Kubernetes Service (AKS) for horizontal scaling and high availability.

• Inference: Use Azure OpenAI Service for production-grade reasoning and lower latency.

• Observability: Would integrate Azure Monitor and Prometheus to track RAG metrics such as retrieval latency, token usage and groundedness scores.

---

Development Workflow & AI Usage:-

• AI as a Force Multiplier: AI coding assistants were utilised for rapid scaffolding and boilerplate generation.

• The "Lead" Guardrail: Every architectural decision—from vector normalisation to prompt constraints was manually verified. I prioritise AI for speed on "known patterns" while retaining total manual control over the system's logic and reliability.

---

Future Enhancements:-

• AST-Aware Chunking: Moving beyond fixed-size windows to understand Python function/class boundaries for even higher retrieval precision.

• Reranking Layer: Adding a Cross-Encoder to re-rank top results before passing them to the LLM.

• Incremental Indexing: Implementing a delta-logic to only re-index files that have changed.

---

Quick Setup Instructions:-

To run this assistant locally, follow these steps:

Environment Setup:


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Model Installation:
Please ensure Ollama is running on your system and pull the required models:


ollama pull all-minilm
ollama pull llama3.2:1b
Indexing the Code:
Index the repository (or the assistant's own code for testing):


python -m scripts.index_repo --path . --reset
Querying the Assistant:
Ask questions about the implementation:


python -m scripts.ask "How is the code indexed?"
Example Interaction
Below is a real-world trace from the system querying its own source code:

Question: "How is the code indexed?"

Answer: > "Based on the provided context, the code is indexed using the Faiss library for vector storage and the Ollama model for text embeddings. The load_and_chunk_repo function in app/loaders.py prepares the repository data, which is then processed by the index_chunks function in app/rag.py to create the searchable index."

Sources Cited:

scripts/index_repo.py

app/rag.py

app/loaders.py
