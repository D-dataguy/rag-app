# Production RAG App

A production-grade Retrieval-Augmented Generation (RAG) API built with FastAPI, LangChain, and ChromaDB.

## Features
- Hybrid search combining vector similarity (ChromaDB) and keyword matching (BM25)
- Citation-enforced answers to prevent hallucination
- REST API built with FastAPI
- Document ingestion pipeline with intelligent chunking

## Tech Stack
- LangChain + LangChain-OpenAI
- ChromaDB (vector store)
- BM25 (keyword search)
- FastAPI + Uvicorn
- OpenAI GPT-3.5-turbo

## Setup
1. Clone the repo
2. Create a virtual environment and install dependencies:
   pip install -r requirements.txt
3. Add your OpenAI API key to a .env file:
   OPENAI_API_KEY=your-key-here
4. Ingest documents:
   python app/ingest.py
5. Start the API:
   uvicorn app.api:app --reload

## Usage
Send a POST request to /ask:
   curl -X POST http://localhost:8000/ask
   -H "Content-Type: application/json"
   -d '{"text": "Your question here"}'