import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

load_dotenv()

def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

def hybrid_search(query: str, k: int = 3):
    # 1. Vector search — finds semantically similar chunks
    vectorstore = load_vectorstore()
    vector_results = vectorstore.similarity_search(query, k=k)

    # 2. BM25 search — finds keyword matching chunks
    docs = vectorstore.get()
    corpus = docs["documents"]
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # 3. Get top BM25 results
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]
    bm25_results = [corpus[i] for i in top_bm25_indices]

    # 4. Combine both results (vector results take priority)
    combined = list(vector_results)
    vector_texts = [doc.page_content for doc in vector_results]
    for text in bm25_results:
        if text not in vector_texts:
            combined.append(text)

    return combined[:k]

if __name__ == "__main__":
    results = hybrid_search("What is machine learning?")
    for i, doc in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        if hasattr(doc, 'page_content'):
            print(doc.page_content)
        else:
            print(doc)