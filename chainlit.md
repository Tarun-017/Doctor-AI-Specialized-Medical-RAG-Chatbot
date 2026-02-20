Doctor AI: Specialized Medical RAG Chatbot
Doctor AI is a high-performance Retrieval-Augmented Generation (RAG) chatbot designed to provide accurate, verifiable medical information. Unlike standard LLMs, it answers questions strictly based on a curated knowledge base (Medical Textbooks/PDFs), eliminating hallucinations and ensuring reliability.

Powered by Groq's LPU inference engine, it delivers near-instant answers with precise source citations.

🚀 Key Features
Zero Hallucination: Answers are grounded only in the provided medical documents.

Verifiable Sources: Every response cites the specific PDF Name and Page Number.

Ultra-Low Latency: Uses Groq (Llama 3.1) for lightning-fast inference (<2s response).

Privacy-First: Document embeddings are processed locally using HuggingFace & FAISS.

🛠️ Tech Stack
LLM: Llama-3.1-8b (via Groq API)

Orchestration: LangChain

Vector Database: FAISS (CPU Optimized)

Embeddings: sentence-transformers/all-MiniLM-L6-v2

UI: Chainlit