"""Compare retrieval quality, size and speed of embedding-storage formats.

Usage:
    uv run ingest.py --max-docs 50
    uv run evaluate.py --queries 200
    uv run evaluate.py --recompute      # ignore the cached fp32 vectors
    uv run evaluate.py --rescore-factors 2,4,8,16,32

The model is the same for every variant; only how the chunk vectors are stored
and searched changes. Query vectors stay fp32 (except for binary search).
"""

import argparse
import json
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from fastembed import TextEmbedding

from ingest import DATASET_DIR, INDEX_DIR

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5
RESCORE_FACTORS = (2, 4, 8, 16)  # binary search fetches TOP_K * factor candidates to rescore
CACHE = INDEX_DIR / "embeddings_fp32.npy"
CACHE_META = INDEX_DIR / "embeddings_fp32.json"


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
Return the fp32 chunk vectors and how long they took to embed, from the cache in
index/ when it matches the current chunk count, otherwise by embedding and caching.
"""
def load_vectors(
    model: TextEmbedding, chunks: list[dict], recompute: bool
) -> tuple[np.ndarray, float]:
    if not recompute and CACHE.exists() and CACHE_META.exists():
        meta = json.loads(CACHE_META.read_text())
        if meta["n_chunks"] == len(chunks) and meta["model"] == EMBED_MODEL:
            return np.load(CACHE), meta["embed_seconds"]
    vectors, seconds = embed_chunks(model, chunks)
    np.save(CACHE, vectors)
    CACHE_META.write_text(
        json.dumps({"n_chunks": len(chunks), "model": EMBED_MODEL, "embed_seconds": seconds})
    )
    return vectors, seconds


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
Scalar int8 quantization. Each dimension gets its own range [lo, hi] measured on
the corpus, and values are mapped linearly onto -128..127. Returns the int8 codes
and a dequantize function that maps codes back to approximate fp32 values.
"""
def quantize_int8(vectors: np.ndarray) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    lo, hi = vectors.min(axis=0), vectors.max(axis=0)
    scale = (hi - lo) / 255.0
    codes = np.round((vectors - lo) / scale - 128).astype(np.int8)

    def dequantize(c: np.ndarray) -> np.ndarray:
        return (c.astype(np.float32) + 128) * scale + lo

    return codes, dequantize


"""
Binary quantization: keep only the sign of each dimension and pack 8 dimensions
per byte (384 dims -> 48 bytes, 32x smaller than fp32).
"""
def quantize_binary(vectors: np.ndarray) -> np.ndarray:
    return np.packbits(vectors > 0, axis=1)


"""
One storage format. `search` takes a fp32 query vector and returns the indices of
the TOP_K best chunks. `stored_bytes` is everything the variant keeps (so
binary + rescoring counts its int8 copy too); `memory_bytes` is what has to sit
in RAM for the first-stage search.
"""
@dataclass
class Variant:
    name: str
    search: Callable[[np.ndarray], np.ndarray]
    stored_bytes: int
    memory_bytes: int


def top_k(scores: np.ndarray, k: int = TOP_K) -> np.ndarray:
    return np.argsort(-scores)[:k]


"""
Build every variant from the fp32 vectors. Note that numpy has no fast int8 or
fp16 matrix multiply, so the compact formats are cast to fp32 inside `search`;
their latency here reflects numpy, not an optimized int8/popcount kernel.
"""
def build_variants(
    fp32: np.ndarray, rescore_factors: tuple[int, ...] = RESCORE_FACTORS
) -> list[Variant]:
    fp16 = fp32.astype(np.float16)
    int8, dequantize = quantize_int8(fp32)
    binary = quantize_binary(fp32)

    def search_binary(query: np.ndarray, k: int) -> np.ndarray:
        distance = np.bitwise_count(binary ^ np.packbits(query > 0)).sum(axis=1)
        return np.argsort(distance)[:k]

    def rescore_variant(factor: int) -> Variant:
        def search(query: np.ndarray) -> np.ndarray:
            candidates = search_binary(query, TOP_K * factor)
            scores = dequantize(int8[candidates]) @ query
            return candidates[top_k(scores)]

        return Variant(
            f"binary + int8 rescore ({factor}x)",
            search,
            binary.nbytes + int8.nbytes,
            binary.nbytes,
        )

    return [
        Variant("fp32 (baseline)", lambda q: top_k(fp32 @ q), fp32.nbytes, fp32.nbytes),
        Variant("fp16", lambda q: top_k(fp16.astype(np.float32) @ q), fp16.nbytes, fp16.nbytes),
        Variant("int8", lambda q: top_k(dequantize(int8) @ q), int8.nbytes, int8.nbytes),
        Variant("binary", lambda q: search_binary(q, TOP_K), binary.nbytes, binary.nbytes),
        *[rescore_variant(f) for f in rescore_factors],
    ]


"""
Score one variant on the sampled questions. Reports gold-paper and gold-section
hit rates in the top results, overlap with the fp32 baseline's top results, and
mean search time per query (query embedding excluded).
"""
def score(
    variant: Variant,
    queries: np.ndarray,
    chunks: list[dict],
    items: list[tuple[dict, dict]],
    baseline_tops: list[set[int]] | None,
) -> dict:
    doc_hits = section_hits = 0
    tops, overlap = [], 0.0
    start = time.perf_counter()
    for q in queries:
        tops.append(variant.search(q))
    latency_ms = (time.perf_counter() - start) / len(queries) * 1000

    for i, ((_, gold), top) in enumerate(zip(items, tops)):
        hits = [chunks[j] for j in top]
        doc_hits += any(h["doc_id"] == gold["doc_id"] for h in hits)
        section_hits += any(
            h["doc_id"] == gold["doc_id"] and h["section_id"] == gold["section_id"]
            for h in hits
        )
        if baseline_tops is not None:
            overlap += len(set(top.tolist()) & baseline_tops[i]) / TOP_K
    return {
        "variant": variant.name,
        "stored_mb": variant.stored_bytes / 1e6,
        "memory_mb": variant.memory_bytes / 1e6,
        "latency_ms": latency_ms,
        "doc_hit": doc_hits / len(items),
        "section_hit": section_hits / len(items),
        "overlap": overlap / len(items) if baseline_tops is not None else 1.0,
        "tops": tops,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--rescore-factors",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=RESCORE_FACTORS,
        help="comma-separated candidate multipliers for binary + int8 rescoring",
    )
    args = parser.parse_args()

    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    items = load_benchmark(chunks, args.queries)
    model = TextEmbedding(EMBED_MODEL)

    fp32, embed_seconds = load_vectors(model, chunks, args.recompute)
    queries = np.array(
        [next(iter(model.query_embed(q["query"]))) for q, _ in items], dtype=np.float32
    )
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    results, baseline_tops = [], None
    for variant in build_variants(fp32, args.rescore_factors):
        result = score(variant, queries, chunks, items, baseline_tops)
        if baseline_tops is None:
            baseline_tops = [set(t.tolist()) for t in result["tops"]]
        results.append(result)

    print(f"{len(chunks)} chunks, {len(items)} queries, fp32 embed time {embed_seconds:.1f} s\n")
    print("| Variant | Stored (MB) | RAM (MB) | Latency (ms/query) | Paper hit@5 | Section hit@5 | Overlap@5 vs fp32 |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['variant']} | {r['stored_mb']:.1f} | {r['memory_mb']:.1f} | {r['latency_ms']:.2f} "
            f"| {r['doc_hit']:.1%} | {r['section_hit']:.1%} | {r['overlap']:.1%} |"
        )
    for r in results:
        del r["tops"]
    (INDEX_DIR / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
