"""
Searches for relevant documents by anomaly + generates explanation via Ollama.
"""

import json

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer

QDRANT_PATH = "data/qdrant"
COLLECTION = "infra_docs"
EMBED_MODEL = "models/all-MiniLM-L6-v2"
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

RAG_PROMPT = """\
You are an infrastructure engineer analyzing a log anomaly.

ANOMALY:
{anomaly}

RELEVANT DOCUMENTATION:
{context}

Based on the documentation above, explain in 3-5 sentences:
1. What likely caused this anomaly
2. How to investigate or fix it

Be specific and technical. Reply in English."""


class RAGRetriever:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(EMBED_MODEL)
        self.client = QdrantClient(path=QDRANT_PATH)
        print(f"Qdrant: {self.client.get_collection(COLLECTION).points_count} chunks")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        vector = self.model.encode(query).tolist()
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=top_k,
        )
        return [
            {
                "text": hit.payload["text"],
                "source": hit.payload["source"],
                "platform": hit.payload["platform"],
                "score": round(hit.score, 3),
            }
            for hit in results.points
        ]

    def explain(self, anomaly_template: str, model_name: str = "phi35") -> dict:
        """
        Full RAG pipeline:
        1. Search for relevant documents
        2. Generate explanation via LLM
        """
        # Search
        chunks = self.search(anomaly_template)
        if not chunks:
            return {
                "explanation": "No relevant documentation found.",
                "sources": [],
                "chunks": [],
            }

        context = chunks[0]["text"][:300]
        # Generate via Ollama
        try:
            resp = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": RAG_PROMPT.format(
                            anomaly=anomaly_template,
                            context=context,
                        ),
                    }
                ],
                options={"temperature": 0.2, "num_predict": 300, "keep_alive": "10m"},
            )
            try:
                explanation = resp["message"]["content"]
            except (TypeError, KeyError):
                explanation = resp.message.content

        except Exception as e:
            explanation = f"LLM unavailable: {e}"

        return {
            "anomaly": anomaly_template,
            "explanation": explanation,
            "sources": [c["source"] for c in chunks],
            "chunks": chunks,
        }


# ─── Demo

if __name__ == "__main__":
    rag = RAGRetriever()

    # Test anomalies from real logs
    test_cases = [
        "<HOST> hostd[<NUM>]: failed to connect to vpxa: connection refused",
        "<HOST> vmms[<NUM>]: live migration failed: insufficient resources",
        "<HOST> clusteragent[<NUM>]: heartbeat timeout detected",
    ]

    for anomaly in test_cases:
        print(f"\n{'='*60}")
        print(f"ANOMALY: {anomaly}")
        print(f"{'='*60}")

        # Search only (no LLM — fast)
        chunks = rag.search(anomaly)
        print(f"\nTop-{TOP_K} relevant documents:")
        for i, c in enumerate(chunks, 1):
            print(f"\n[{i}] score={c['score']} | {c['platform']}")
            print(f"    {c['text'][:200]}...")

        print("\n" + "-" * 40)
        print("Generating explanation (LLM)...")
        result = rag.explain(anomaly)
        print(f"\nEXPLANATION:\n{result['explanation']}")

    rag.client.close()
