# rag-embedding-quantization

How much retrieval quality do you lose when you quantize the embedding model of a RAG system, and how much memory and latency do you save?

Built on the retrieval pipeline of my [very-simple-rag](https://github.com/BWork0/very-simple-rag) project (self-contained here: nothing needs to be installed from it). Uses the [Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench) (arXiv papers) and its gold labels to measure retrieval.

## Setup

```sh
uv sync
uv run ingest.py                 # all 1000 papers (~740 MB) -> index/chunks.json (74,316 chunks)
uv run evaluate.py --queries 1000
```

Use `ingest.py --max-docs 50` and `--queries 200` for a quick run. Embedding the full corpus is slow on CPU (about 2.3 hours on the machine below); the fp32 vectors are cached in `index/`, so re-runs skip it.

## Method

- **Model:** `BAAI/bge-small-en-v1.5` via fastembed (ONNX).
- **Chunking:** ~1500-character chunks with 200 overlap, paper title prepended before embedding.
- **Retrieval:** cosine similarity, top 5.
- **Metrics:** gold paper in top 5, gold section in top 5, index size, embedding time, query latency.
- **Same questions for every variant** (fixed seed).

## Results

Full corpus: 1000 papers, 74,316 chunks, 1000 benchmark questions. fp32 embedding took 8395 s. Machine: Windows 10 + WSL2, i7-11700K.

| Variant | Stored (MB) | RAM (MB) | Paper hit@5 | Section hit@5 | Top-5 overlap with fp32 |
|---|---|---|---|---|---|
| fp32 (baseline) | 114.1 | 114.1 | 93.5% | 80.1% | 100.0% |
| fp16 | 57.1 | 57.1 | 93.5% | 80.1% | 99.8% |
| int8 | 28.5 | 28.5 | 93.4% | 80.1% | 98.6% |
| binary | 3.6 | 3.6 | 85.8% | 66.4% | 45.6% |
| binary + int8 rescore (4x) | 32.1 | 3.6 | 91.1% | 75.5% | 72.8% |

Search latency (ms/query, fp32 / fp16 / int8 / binary / binary+rescore): 5.50 / 81.36 / 84.08 / 3.28 / 2.89. These measure numpy, not the formats: numpy has no fast fp16 or int8 matrix multiply, so those variants are cast to fp32 on every search. Do not read them as a speed comparison.

An ONNX int8 *model* variant is not measured yet (see roadmap).

## Findings

- **fp16 and int8 are free.** Section hit@5 is identical to fp32 (80.1%) at 2x and 4x smaller storage. With 1000 questions, one point is about 10 questions and the standard error is about 1.3 points, so differences of 0.1 are noise.
- **Binary is cheap but lossy.** 32x smaller (3.6 MB), but section hit@5 drops from 80.1% to 66.4%, and its top 5 shares only 46% of chunks with fp32.
- **Rescoring wins back most of the loss.** Fetching 20 binary candidates and rescoring them against int8 vectors gives 75.5% section hit@5 (4.6 points below fp32) with only 3.6 MB in RAM. The int8 copy (28.5 MB) still has to be stored, e.g. on disk.
- **Corpus size matters.** On a 50-paper subset (3,809 chunks) binary looked much better (78.5% vs 85.0% for fp32, a 6.5-point gap). On the full corpus the gap is 13.7 points, because more chunks means more near neighbours for coarse 1-bit vectors to confuse.
- **Limits:** one embedding model, one dataset, query vectors not quantized, and the rescore multiplier was fixed at 4x (a larger multiplier is likely to close more of the gap, but was not tested).

## Roadmap

- [x] Project scaffold, fp32 baseline eval
- [x] Embedding storage quantization: fp16, int8, binary (with rescoring)
- [ ] Rescore multiplier sweep (2x, 4x, 8x, 16x)
- [ ] Model quantization: ONNX int8
- [ ] Results table and quality-vs-size plot
- [ ] Write-up of trade-offs and failure cases
