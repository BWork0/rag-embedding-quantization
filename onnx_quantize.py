"""Quantize the embedding model itself (ONNX dynamic int8) and measure what it costs.

Usage:
    uv run --group onnx onnx_quantize.py --check            # parity check on a few chunks
    uv run --group onnx onnx_quantize.py --queries 1000     # full evaluation

Three models are compared, all BAAI/bge-small-en-v1.5, scored with the same flat
fp32 search over their own vectors:
  fp32  the official fp32 ONNX export
  int8  that export with dynamic int8 weight quantization (onnxruntime)
  fp16  the model fastembed ships (its weights are fp16, despite the "-Q" repo name);
        its vectors are the cached ones from evaluate.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastembed import TextEmbedding
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from tokenizers import Tokenizer

from evaluate import CACHE, CACHE_META, EMBED_MODEL, load_benchmark, score, top_k, Variant
from ingest import INDEX_DIR

MODEL_DIR = Path("models")
MAX_LENGTH = 512
BATCH_SIZE = 32


"""
Download the official fp32 ONNX export and tokenizer, and create the int8 model from
it with dynamic quantization (weights stored as int8, activations quantized at run
time). Returns the model paths by name and the tokenizer path.
"""
def prepare_models() -> tuple[dict[str, Path], Path]:
    fp32 = Path(hf_hub_download(EMBED_MODEL, "onnx/model.onnx", local_dir=MODEL_DIR))
    tokenizer = Path(hf_hub_download(EMBED_MODEL, "tokenizer.json", local_dir=MODEL_DIR))
    int8 = MODEL_DIR / "model_int8.onnx"
    if not int8.exists():
        quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    return {"fp32": fp32, "int8": int8}, tokenizer


"""
Runs one ONNX model. Reproduces what fastembed does for this model: tokenize, run,
take the first ([CLS]) token's output, and normalize to length 1. Queries get no
instruction prefix, matching evaluate.py.
"""
class Embedder:
    def __init__(self, model_path: Path, tokenizer_path: Path) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(MAX_LENGTH)
        self.tokenizer.enable_padding()
        self.size_mb = model_path.stat().st_size / 1e6

    def embed(self, texts: list[str]) -> np.ndarray:
        # Sort by length so each batch pads to a similar size, then restore the order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.empty((len(texts), 384), dtype=np.float32)
        for start in range(0, len(texts), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            enc = self.tokenizer.encode_batch([texts[i] for i in idx])
            inputs = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
            }
            cls = self.session.run(None, inputs)[0][:, 0]
            out[idx] = cls / np.linalg.norm(cls, axis=1, keepdims=True)
        return out


"""
Embed every chunk (with the paper title prepended, as in evaluate.py) with one model,
cached in index/ so reruns skip the slow part. Returns the vectors and the seconds
the embedding took.
"""
def embed_corpus(name: str, embedder: Embedder, chunks: list[dict]) -> tuple[np.ndarray, float]:
    vec_path, meta_path = INDEX_DIR / f"embeddings_{name}.npy", INDEX_DIR / f"embeddings_{name}.json"
    if vec_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["n_chunks"] == len(chunks):
            return np.load(vec_path), meta["embed_seconds"]
    start = time.perf_counter()
    vectors = embedder.embed([f"{c['title']}\n{c['text']}" for c in chunks])
    seconds = time.perf_counter() - start
    np.save(vec_path, vectors)
    meta_path.write_text(json.dumps({"n_chunks": len(chunks), "embed_seconds": seconds}))
    return vectors, seconds


"""
Parity check on a few chunks: my fp32 pipeline must reproduce fastembed's vectors
to within the fp16 rounding of its weights, and the int8 model should stay close
to fp32. Prints mean cosine similarity between each pair.
"""
def check(embedders: dict[str, Embedder], chunks: list[dict], n: int) -> None:
    texts = [f"{c['title']}\n{c['text']}" for c in chunks[:n]]
    reference = np.array(list(TextEmbedding(EMBED_MODEL).passage_embed(texts)), dtype=np.float32)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    mine = {name: e.embed(texts) for name, e in embedders.items()}
    print(f"mean cosine over {n} chunks")
    print(f"  fp32 (mine) vs fastembed fp16: {(mine['fp32'] * reference).sum(axis=1).mean():.5f}")
    print(f"  int8 (mine) vs fp32 (mine):    {(mine['int8'] * mine['fp32']).sum(axis=1).mean():.5f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N chunks (smoke test)")
    args = parser.parse_args()

    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    paths, tokenizer = prepare_models()
    embedders = {name: Embedder(p, tokenizer) for name, p in paths.items()}

    if args.check:
        check(embedders, chunks, 64)
        return

    if args.limit:
        chunks = chunks[: args.limit]
    items = load_benchmark(chunks, args.queries)
    query_texts = [q["query"] for q, _ in items]

    results, baseline_tops = [], None
    for name in ("fp32", "int8"):
        vectors, seconds = embed_corpus(f"onnx_{name}" + (f"_{args.limit}" if args.limit else ""), embedders[name], chunks)
        queries = embedders[name].embed(query_texts)
        variant = Variant(f"{name} model", lambda q, v=vectors: top_k(v @ q), vectors.nbytes, vectors.nbytes)
        result = score(variant, queries, chunks, items, baseline_tops)
        if baseline_tops is None:
            baseline_tops = [set(t.tolist()) for t in result["tops"]]
        result.update(model_mb=embedders[name].size_mb, embed_seconds=seconds, chunks_per_s=len(chunks) / seconds)
        results.append(result)

    if not args.limit and CACHE.exists():
        meta = json.loads(CACHE_META.read_text())
        if meta["n_chunks"] == len(chunks):
            vectors = np.load(CACHE)
            fastembed_model = TextEmbedding(EMBED_MODEL)
            queries = np.array(
                [next(iter(fastembed_model.query_embed(t))) for t in query_texts], dtype=np.float32
            )
            queries /= np.linalg.norm(queries, axis=1, keepdims=True)
            variant = Variant("fp16 model (fastembed)", lambda q: top_k(vectors @ q), vectors.nbytes, vectors.nbytes)
            result = score(variant, queries, chunks, items, baseline_tops)
            result.update(model_mb=66.5, embed_seconds=meta["embed_seconds"], chunks_per_s=len(chunks) / meta["embed_seconds"])
            results.append(result)

    print(f"{len(chunks)} chunks, {len(items)} queries\n")
    print("| Model | Model size (MB) | Embed time (s) | Chunks/s | Paper hit@5 | Section hit@5 | Overlap@5 vs fp32 model |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['variant']} | {r['model_mb']:.1f} | {r['embed_seconds']:.0f} | {r['chunks_per_s']:.1f} "
            f"| {r['doc_hit']:.1%} | {r['section_hit']:.1%} | {r['overlap']:.1%} |"
        )
    for r in results:
        del r["tops"]
    (INDEX_DIR / "model_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
