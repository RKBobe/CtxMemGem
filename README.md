# CtxMemGem 🧠
CtxMemGem is a personal AI memory and synthesis engine that provides a large language model with contextual understanding of your codebase, enabling more relevant and insightful assistance.

## 
Core Functionality
GitHub Integration: Securely connects to your GitHub account via OAuth to access specified repositories.

Code Processing: Reads and processes text-based files, breaking them down into logical, understandable chunks.

Semantic Embedding: Uses a local AI model to generate vector embeddings, capturing the semantic meaning of your code without sending the code itself to an external service.

Vector Storage: Stores these embeddings in a ChromaDB vector database to create a persistent, searchable "memory" of your projects.

Context-Aware Q&A: Provides a powerful API that retrieves the most relevant code snippets based on your natural language questions and synthesizes them for an AI-powered answer.

How It Works 🚀
This tool implements a Retrieval-Augmented Generation (RAG) pipeline:

Ingest: The code from a repository is chunked and converted into vector embeddings, which are stored in the memory.

Retrieve: When you ask a question, the engine performs a semantic search on the stored memory to find the most relevant code snippets.

Augment & Synthesize: These snippets are combined with your original question into a detailed prompt. This prompt is then sent to the Google Gemini API, allowing it to generate an answer that is deeply grounded in the specific context of your work.

Technology Stack 🛠️
Backend: Node.js, Express

Database: ChromaDB (running in Docker)

Local Embeddings: @xenova/transformers.js

Synthesis LLM: Google Gemini API

GitHub Integration: Axios, OAuth 2.0
