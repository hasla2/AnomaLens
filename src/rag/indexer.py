"""
Parses HTML documents, splits into chunks, creates embeddings and saves to Qdrant.
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

DOCS_DIR = "data/docs"
QDRANT_PATH = "data/qdrant"  # local storage (no Docker)
COLLECTION = "infra_docs"
# EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"  # 80 MB, fast on CPU
EMBED_MODEL = "models/all-MiniLM-L6-v2"
CHUNK_SIZE = 400  # characters
CHUNK_OVERLAP = 80


def extract_text(html_path: Path) -> str:
    """Extracts plain text from HTML."""
    try:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(
            ["script", "style", "nav", "footer", "header", "aside", "form"]
        ):
            tag.decompose()
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"class": "content"})
            or soup.body
        )
        text = main.get_text(separator="\n", strip=True) if main else ""
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
        return "\n".join(lines)
    except Exception as e:
        print(f"  WARN: {html_path.name}: {e}")
        return ""


def chunk_text(text: str, source: str, platform: str) -> list[dict]:
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(
                {
                    "text": chunk,
                    "source": source,
                    "platform": platform,
                    "chunk_id": idx,
                }
            )
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index():
    print("=== RAG Indexer ===\n")

    # 1. Load embedding model
    print(f"Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}\n")

    # 2. Initialize Qdrant (local file mode, no Docker)
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)

    # Recreate collection
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' deleted (recreating)")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION}' created\n")

    # 3. Walk all HTML files
    all_chunks = []
    docs_path = Path(DOCS_DIR)

    for platform in ["vmware", "hyperv"]:
        platform_dir = docs_path / platform
        if not platform_dir.exists():
            print(f"WARN: directory not found: {platform_dir}")
            continue

        html_files = list(platform_dir.rglob("*.html"))
        print(f"[{platform}] files found: {len(html_files)}")

        for html_file in html_files:
            text = extract_text(html_file)
            if len(text) < 100:
                continue
            chunks = chunk_text(text, str(html_file), platform)
            all_chunks.extend(chunks)
            print(f"  {html_file.name}: {len(text)} chars → {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    if not all_chunks:
        print("ERROR: no chunks to index")
        return

    # 4. Create embeddings in batches
    print("\nCreating embeddings...")
    texts = [c["text"] for c in all_chunks]
    batch = 64
    vectors = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i : i + batch]
        embs = model.encode(batch_texts, show_progress_bar=False)
        vectors.extend(embs.tolist())
        print(f"  {min(i+batch, len(texts))}/{len(texts)}", end="\r")
    print()

    # 5. Upload to Qdrant
    print("Uploading to Qdrant...")
    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "text": all_chunks[i]["text"],
                "source": all_chunks[i]["source"],
                "platform": all_chunks[i]["platform"],
                "chunk_id": all_chunks[i]["chunk_id"],
            },
        )
        for i in range(len(all_chunks))
    ]

    # Upload in batches of 100
    for i in range(0, len(points), 100):
        client.upsert(collection_name=COLLECTION, points=points[i : i + 100])

    info = client.get_collection(COLLECTION)
    print(f"\n Indexed: {info.points_count} chunks")
    print(f"   Storage: {QDRANT_PATH}")


if __name__ == "__main__":
    build_index()
