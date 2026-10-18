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

Run the tests with `uv run pytest` (about a second; they use random vectors and need no data).

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
| binary + int8 rescore (2x) | 32.1 | 3.6 | 89.0% | 71.9% | 61.4% |
| binary + int8 rescore (4x) | 32.1 | 3.6 | 91.1% | 75.5% | 72.8% |
| binary + int8 rescore (8x) | 32.1 | 3.6 | 91.6% | 77.3% | 81.3% |
| binary + int8 rescore (16x) | 32.1 | 3.6 | 92.4% | 78.7% | 87.4% |

![Section hit@5 vs index size for every variant](results/quality_vs_size.png)

The raw numbers behind the table and plot are in [`results/full_corpus.json`](results/full_corpus.json) (transcribed from the printed `evaluate.py` table, so values carry the table's rounding). Regenerate the plot with `uv run --group plot plot.py`.

"Nx" means binary search fetches 5 × N candidates, which are then rescored with the fp32 query against their int8 vectors.

Search latency (ms/query): fp32 4.62, fp16 69.14, int8 88.49, binary 3.42, binary + rescore 3.5–3.9 for every multiplier. fp16 and int8 are slow only because numpy has no fast fp16 or int8 matrix multiply, so they are cast to fp32 on every search; do not read that as a property of the formats. The rescore multipliers can be compared with each other (rescoring 80 candidates adds well under 1 ms over plain binary), but not with an optimized int8/popcount implementation.

An ONNX int8 *model* variant is not measured yet (see roadmap).

## Findings

- **fp16 and int8 are free.** Section hit@5 is identical to fp32 (80.1%) at 2x and 4x smaller storage. With 1000 questions, one point is about 10 questions and the standard error is about 1.3 points, so differences of 0.1 are noise.
- **Binary is cheap but lossy.** 32x smaller (3.6 MB), but section hit@5 drops from 80.1% to 66.4%, and its top 5 shares only 46% of chunks with fp32.
- **Rescoring wins back almost all of the loss, with diminishing returns.** Section hit@5 climbs 66.4% (no rescore), 71.9% (2x), 75.5% (4x), 77.3% (8x), 78.7% (16x). Each doubling gains less (+5.5, +3.6, +1.8, +1.4 points). At 16x the result is 1.4 points below fp32 (80.1%), which is close to the noise level of this evaluation, with only 3.6 MB in RAM. The int8 copy (28.5 MB) still has to be stored, e.g. on disk, and 80 candidate vectors must be read from it per query.
- **Rescoring cost is small here.** The extra latency from 2x to 16x is about 0.4 ms in this numpy setup, but on a disk-backed int8 store the reads would likely dominate. That was not measured.
- **Corpus size matters.** On a 50-paper subset (3,809 chunks) binary looked much better (78.5% vs 85.0% for fp32, a 6.5-point gap). On the full corpus the gap is 13.7 points, because more chunks means more near neighbours for coarse 1-bit vectors to confuse.
- **Limits:** one embedding model, one dataset, query vectors not quantized, and the rescore sweep stops at 16x (the curve was still rising slowly).

## Roadmap

- [x] Project scaffold, fp32 baseline eval
- [x] Embedding storage quantization: fp16, int8, binary (with rescoring)
- [x] Rescore multiplier sweep (2x, 4x, 8x, 16x)
- [ ] Model quantization: ONNX int8
- [x] Results table and quality-vs-size plot
- [ ] Write-up of trade-offs and failure cases
