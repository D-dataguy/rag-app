import time
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from app.retriever import hybrid_search
from app.monitor import log_request, get_metrics

load_dotenv()

app = FastAPI()

class Question(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "RAG API is running"}

@app.get("/metrics")
def metrics():
    return get_metrics()

@app.post("/ask")
def ask(question: Question):
    start_time = time.time()

    # 1. Retrieve relevant chunks
    chunks = hybrid_search(question.text, k=3)

    # 2. Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content if hasattr(chunk, 'page_content') else chunk
        context_parts.append(f"[{i+1}] {text}")
    context = "\n\n".join(context_parts)

    # 3. Build prompt
    prompt = f"""You are a helpful assistant. Answer the question using ONLY 
the context below. After each statement, cite the source number like [1] or [2].
If the answer isn't in the context, say "I don't have enough information."

Context:
{context}

Question: {question.text}

Answer:"""

    # 4. Send to LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.invoke(prompt)

    # 5. Calculate latency and log everything
    latency_ms = (time.time() - start_time) * 1000
    log_record = log_request(
        question=question.text,
        answer=response.content,
        chunks=chunks,
        latency_ms=latency_ms
    )

    return {
        "question": question.text,
        "answer": response.content,
        "sources": context_parts,
        "latency_ms": log_record["latency_ms"],
        "estimated_cost_usd": log_record["estimated_cost_usd"]
    }