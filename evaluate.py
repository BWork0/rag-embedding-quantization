"""Measure retrieval quality of the fp32 baseline embedding model.

Usage:
    uv run ingest.py --max-docs 50
    uv run evaluate.py --queries 200

Quantized variants (fp16, int8, binary, ...) will be added as extra entries in
VARIANTS and scored by the same function.
"""

import argparse
import json
import random
import time

import numpy as np
from fastembed import TextEmbedding

from ingest import DATASET_DIR, INDEX_DIR

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5


"""
Embed every chunk (with the paper title prepended) and normalize the vectors to
length 1, so a dot product is cosine similarity. Returns the matrix and seconds taken.
"""
def embed_chunks(model: TextEmbedding, chunks: list[dict]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    vectors = np.array(
        list(model.passage_embed([f"{c['title']}\n{c['text']}" for c in chunks])),
        dtype=np.float32,
    )
    elapsed = time.perf_counter() - start
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True), elapsed


"""
Load benchmark questions whose gold paper is in the index, then sample n of them
with a fixed seed so every variant is scored on the same questions.
"""
def load_benchmark(chunks: list[dict], n: int) -> list[tuple[dict, dict]]:
    queries = json.loads((DATASET_DIR / "queries.json").read_text(encoding="utf-8"))
    qrels = json.loads((DATASET_DIR / "qrels.json").read_text(encoding="utf-8"))
    indexed = {c["doc_id"] for c in chunks}
    items = [(queries[qid], rel) for qid, rel in qrels.items() if rel["doc_id"] in indexed]
    return random.Random(0).sample(items, min(n, len(items)))


"""
Score one embedding matrix: for each question, take the top-k chunks by dot
product and check whether the gold paper / exact gold section is among them.
Returns the two hit rates as fractions.
"""
def score(
    model: TextEmbedding,
    vectors: np.ndarray,
    chunks: list[dict],
    items: list[tuple[dict, dict]],
) -> dict[str, float]:
    doc_hits = section_hits = 0
    for query, gold in items:
        q = next(iter(model.query_embed(query["query"])))
        scores = vectors @ (q / np.linalg.norm(q))
        hits = [chunks[i] for i in np.argsort(-scores)[:TOP_K]]
        doc_hits += any(h["doc_id"] == gold["doc_id"] for h in hits)
        section_hits += any(
            h["doc_id"] == gold["doc_id"] and h["section_id"] == gold["section_id"]
            for h in hits
        )
    return {"doc_hit": doc_hits / len(items), "section_hit": section_hits / len(items)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=200)
    args = parser.parse_args()

    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    items = load_benchmark(chunks, args.queries)
    model = TextEmbedding(EMBED_MODEL)

    vectors, seconds = embed_chunks(model, chunks)
    result = score(model, vectors, chunks, items)
    print(f"fp32 baseline: {len(chunks)} chunks, {len(items)} queries")
    print(f"  index size:  {vectors.nbytes / 1e6:.1f} MB")
    print(f"  embed time:  {seconds:.1f} s")
    print(f"  gold paper in top {TOP_K}:   {result['doc_hit']:.1%}")
    print(f"  gold section in top {TOP_K}: {result['section_hit']:.1%}")


if __name__ == "__main__":
    main()
