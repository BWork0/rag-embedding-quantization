# rag-embedding-quantization

How much retrieval quality do you lose when you quantize the embedding model of a RAG system, and how much memory and latency do you save?

Built on the retrieval pipeline of my [very-simple-rag](https://github.com/BWork0/very-simple-rag) project (self-contained here: nothing needs to be installed from it). Uses the [Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench) (arXiv papers) and its gold labels to measure retrieval.

## Setup

```sh
uv sync
uv run ingest.py --max-docs 50   # downloads papers and chunks them into index/chunks.json
uv run evaluate.py --queries 200 # fp32 baseline
```

## Method

- **Model:** `BAAI/bge-small-en-v1.5` via fastembed (ONNX).
- **Chunking:** ~1500-character chunks with 200 overlap, paper title prepended before embedding.
- **Retrieval:** cosine similarity, top 5.
- **Metrics:** gold paper in top 5, gold section in top 5, index size, embedding time, query latency.
- **Same questions for every variant** (fixed seed).

## Results

_To be filled in as variants are implemented._

| Variant | Index size | Embed time | Query latency | Paper hit@5 | Section hit@5 |
|---|---|---|---|---|---|
| fp32 (baseline) | | | | | |
| fp16 | | | | | |
| int8 model (ONNX dynamic quantization) | | | | | |
| int8 embeddings | | | | | |
| binary embeddings | | | | | |

## Findings

_To be written after the results are in._

## Roadmap

- [x] Project scaffold, fp32 baseline eval
- [ ] Embedding storage quantization: fp16, int8, binary (with rescoring)
- [ ] Model quantization: ONNX int8
- [ ] Results table and quality-vs-size plot
- [ ] Write-up of trade-offs and failure cases
