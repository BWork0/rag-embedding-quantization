"""Checks for the quantization code in evaluate.py and the chunker in ingest.py.

Run with: uv run pytest
Vectors are random unit vectors, which is enough to test the arithmetic; retrieval
quality on real embeddings is measured by evaluate.py itself.
"""

import numpy as np
import pytest

from evaluate import TOP_K, build_variants, quantize_binary, quantize_int8
from ingest import CHUNK_CHARS, CHUNK_OVERLAP, split_text

N, DIM = 500, 384


@pytest.fixture
def vectors() -> np.ndarray:
    v = np.random.default_rng(0).normal(size=(N, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def queries(vectors: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(1)
    q = vectors[rng.choice(N, 20)] + 0.3 * rng.normal(size=(20, DIM)).astype(np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_int8_codes_are_int8_in_range(vectors):
    codes, _ = quantize_int8(vectors)
    assert codes.dtype == np.int8
    assert codes.shape == vectors.shape
    assert codes.min() == -128 and codes.max() == 127  # each dimension's range is fully used


def test_int8_round_trip_error_is_at_most_half_a_step(vectors):
    codes, dequantize = quantize_int8(vectors)
    step = (vectors.max(axis=0) - vectors.min(axis=0)) / 255.0
    error = np.abs(dequantize(codes) - vectors)
    assert (error <= step / 2 + 1e-6).all()


def test_int8_handles_a_constant_dimension(vectors):
    vectors[:, 7] = 0.25
    codes, dequantize = quantize_int8(vectors)
    assert np.isfinite(dequantize(codes)).all()
    assert np.allclose(dequantize(codes)[:, 7], 0.25, atol=1e-6)


def test_binary_packs_signs_into_bytes(vectors):
    packed = quantize_binary(vectors)
    assert packed.dtype == np.uint8
    assert packed.shape == (N, DIM // 8)
    assert (np.unpackbits(packed, axis=1).astype(bool) == (vectors > 0)).all()


def test_binary_treats_zero_as_negative():
    packed = quantize_binary(np.array([[0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    assert packed[0, 0] == 0b01000000


def test_stored_sizes_match_the_format(vectors):
    sizes = {v.name: v.stored_bytes for v in build_variants(vectors, (4,))}
    assert sizes["fp32 (baseline)"] == N * DIM * 4
    assert sizes["fp16"] == N * DIM * 2
    assert sizes["int8"] == N * DIM
    assert sizes["binary"] == N * DIM // 8
    assert sizes["binary + int8 rescore (4x)"] == N * DIM // 8 + N * DIM


def test_rescore_ram_is_only_the_binary_index(vectors):
    by_name = {v.name: v for v in build_variants(vectors, (4,))}
    assert by_name["binary + int8 rescore (4x)"].memory_bytes == by_name["binary"].memory_bytes


def test_fp16_and_int8_nearly_match_fp32(vectors, queries):
    variants = {v.name: v for v in build_variants(vectors, (4,))}
    for name, minimum in [("fp16", 0.99), ("int8", 0.95)]:
        overlap = np.mean(
            [
                len(set(variants[name].search(q)) & set(variants["fp32 (baseline)"].search(q))) / TOP_K
                for q in queries
            ]
        )
        assert overlap >= minimum, name


def test_binary_search_returns_the_smallest_hamming_distances(vectors, queries):
    binary = {v.name: v for v in build_variants(vectors, (4,))}["binary"]
    bits = vectors > 0
    for q in queries:
        distances = (bits != (q > 0)).sum(axis=1)  # slow reference: count differing signs
        got = binary.search(q)
        assert len(got) == TOP_K
        # Ties are broken arbitrarily, so compare distances rather than indices.
        assert sorted(distances[got]) == sorted(distances)[:TOP_K]


def test_rescore_with_the_whole_corpus_as_candidates_equals_int8_search(vectors, queries):
    variants = {v.name: v for v in build_variants(vectors, (N // TOP_K,))}
    rescore, int8 = variants[f"binary + int8 rescore ({N // TOP_K}x)"], variants["int8"]
    for q in queries:
        assert set(rescore.search(q)) == set(int8.search(q))


def test_more_rescore_candidates_never_lowers_overlap_with_int8(vectors, queries):
    variants = {v.name: v for v in build_variants(vectors, (1, 2, 4, 8, 16))}
    overlaps = [
        np.mean(
            [
                len(set(variants[f"binary + int8 rescore ({f}x)"].search(q)) & set(variants["int8"].search(q)))
                for q in queries
            ]
        )
        for f in (1, 2, 4, 8, 16)
    ]
    assert overlaps == sorted(overlaps)


def test_chunks_are_bounded_overlap_and_cover_the_text():
    text = "".join(chr(97 + i % 26) for i in range(5000))
    chunks = split_text(text)
    assert all(len(c) <= CHUNK_CHARS for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev[-CHUNK_OVERLAP:] == nxt[:CHUNK_OVERLAP]
    rebuilt = chunks[0] + "".join(c[CHUNK_OVERLAP:] for c in chunks[1:])
    assert rebuilt == text


def test_short_text_is_a_single_chunk():
    assert split_text("hello") == ["hello"]
