"""Download the Open RAG Benchmark (arXiv) dataset and chunk it into index/chunks.json.

Embedding is done per model variant in evaluate.py, so no vectors are built here.

Usage:
    uv run ingest.py                 # all 1000 papers (~740 MB download)
    uv run ingest.py --max-docs 50   # quick start with a subset
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "vectara/open_ragbench"
DATA_DIR = Path("data")
INDEX_DIR = Path("index")
DATASET_DIR = DATA_DIR / "pdf" / "arxiv"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 200


"""
Download the dataset from Hugging Face into data/.
Always fetches queries.json, qrels.json and answers.json. Then fetches the paper
files: all of them, or, with --max-docs, the first N with the papers that
benchmark questions point to placed first so the subset can still answer them.
"""
def download(max_docs: int | None) -> None:
    meta = ["queries.json", "qrels.json", "answers.json"]
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        local_dir=DATA_DIR,
        allow_patterns=[f"pdf/arxiv/{name}" for name in meta],
    )

    corpus_files = [
        f
        for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
        if f.startswith("pdf/arxiv/corpus/")
    ]
    if max_docs is not None:
        # Prefer papers that queries point at, so the subset is still answerable.
        qrels = json.loads((DATASET_DIR / "qrels.json").read_text(encoding="utf-8"))
        gold = {q["doc_id"] for q in qrels.values()}
        corpus_files.sort(key=lambda f: (Path(f).stem not in gold, f))
        corpus_files = corpus_files[:max_docs]

    print(f"Downloading {len(corpus_files)} papers...")
    snapshot_download(
        REPO_ID, repo_type="dataset", local_dir=DATA_DIR, allow_patterns=corpus_files
    )


"""
Chunker: split text into CHUNK_CHARS-long pieces. Each piece starts CHUNK_OVERLAP
characters before the previous one ended, so text on a boundary appears whole in
at least one chunk.
"""
def split_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


"""
Read every downloaded paper and turn it into chunks. Each section's text is joined
with its markdown tables (images are skipped), split with split_text, and stored
as a dict with doc_id, title, section_id and text.
"""
def load_chunks() -> list[dict]:
    chunks = []
    for path in sorted((DATASET_DIR / "corpus").glob("*.json")):
        paper = json.loads(path.read_text(encoding="utf-8"))
        for section_id, section in enumerate(paper["sections"]):
            # Images are base64 blobs, so only text and markdown tables are indexed.
            text = section["text"]
            for table in section.get("tables", {}).values():
                text += "\n\n" + table
            for piece in split_text(text.strip()):
                chunks.append(
                    {
                        "doc_id": paper["id"],
                        "title": paper["title"],
                        "section_id": section_id,
                        "text": piece,
                    }
                )
    return chunks


"""
Ingestion pipeline: download the data, chunk it, and save the chunks (text and
metadata) to index/chunks.json.
"""
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        download(args.max_docs)

    chunks = load_chunks()
    INDEX_DIR.mkdir(exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {INDEX_DIR}/chunks.json")


if __name__ == "__main__":
    main()
