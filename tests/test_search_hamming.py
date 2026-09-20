# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance_ray.pool import clear_global_pool, init_global_pool
from lance_ray.search import _compute_vector_distances, vector_search


@pytest.fixture(scope="module")
def search_pool(ray_context: None) -> Iterator[None]:
    init_global_pool(processes=2)
    try:
        yield
    finally:
        clear_global_pool(close=True)


def _table(ids: list[int], vectors: list[list[int]]) -> pa.Table:
    return pa.table(
        {"id": ids, "vector": pa.array(vectors, type=pa.list_(pa.uint8(), 2))}
    )


@pytest.mark.parametrize("query_kind", ["list", "numpy", "arrow"])
@pytest.mark.parametrize("query_bytes", [[0, 0], [128, 1], [255, 255]])
def test_hamming_distances_match_native_lance(
    tmp_path: Path, query_kind: str, query_bytes: list[int]
) -> None:
    rows = [[255, 0], [1, 1], [0, 0], [128, 127]]
    table = _table([0, 1, 2, 3], rows)
    # Exercise multiple Arrow chunks and a nonzero slice offset.
    values = table["vector"].combine_chunks()
    vectors = pa.chunked_array([values.slice(0, 1), values.slice(1)])
    query: Any = query_bytes
    if query_kind == "numpy":
        query = np.asarray(query, dtype=np.uint8)
    elif query_kind == "arrow":
        query = pa.array(query, type=pa.uint8())
    dataset = lance.write_dataset(table, tmp_path / "distances.lance")
    native = dataset.to_table(
        columns=["id", "_distance"],
        nearest={
            "column": "vector",
            "q": query,
            "metric": "hamming",
            "k": 4,
            "use_index": False,
        },
    ).sort_by("id")

    actual = _compute_vector_distances(vectors, query, "hamming")

    assert actual.dtype == np.float32
    expected = [
        sum((x ^ y).bit_count() for x, y in zip(row, query_bytes, strict=True))
        for row in rows
    ]
    assert actual.tolist() == expected
    assert actual.tolist() == native["_distance"].to_pylist()


def test_hamming_accumulation_does_not_overflow_a_byte() -> None:
    vectors = pa.chunked_array(
        [pa.array([[255] * 512], type=pa.list_(pa.uint8(), 512))]
    )
    assert _compute_vector_distances(vectors, [0] * 512, "hamming").tolist() == [4096.0]


@pytest.mark.parametrize(
    "query",
    [[-1, 0], [256, 0], [1.5, 0], [1.0, 0.0], [True, False], [None, 0], ["1", "0"]],
)
def test_hamming_rejects_invalid_query_bytes(query: Any) -> None:
    with pytest.raises(ValueError, match="Hamming.*integers.*0.*255"):
        _compute_vector_distances(_table([0], [[0, 0]])["vector"], query, "hamming")


@pytest.mark.parametrize("value_type", [pa.float32(), pa.int8(), pa.uint16()])
def test_hamming_rejects_non_uint8_columns(value_type: pa.DataType) -> None:
    vectors = pa.chunked_array([pa.array([[0, 1]], type=pa.list_(value_type, 2))])
    with pytest.raises(ValueError, match="Hamming.*uint8"):
        _compute_vector_distances(vectors, [0, 0], "hamming")


@pytest.mark.parametrize("vectors", [[None], [[0, None]]])
def test_hamming_rejects_null_vectors_and_elements(vectors: Any) -> None:
    column = pa.chunked_array([pa.array(vectors, type=pa.list_(pa.uint8(), 2))])
    with pytest.raises(ValueError, match="null"):
        _compute_vector_distances(column, [0, 0], "hamming")


@pytest.mark.parametrize(
    "query, message", [([[0, 0]], "one-dimensional"), ([0], "dimension")]
)
def test_hamming_validates_query_shape(query: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _compute_vector_distances(_table([0], [[0, 0]])["vector"], query, "hamming")


@pytest.mark.parametrize("filtered", [True, False])
def test_mixed_hamming_search_matches_lance(
    tmp_path: Path, search_pool: None, filtered: bool
) -> None:
    uri = tmp_path / "mixed.lance"
    dataset = lance.write_dataset(
        _table(list(range(256)), [[1, 1], [3, 1], *([[255, 255]] * 254)]), uri
    )
    dataset.create_index(
        "vector", index_type="IVF_FLAT", metric="hamming", num_partitions=1
    )
    dataset = lance.write_dataset(_table([256], [[255, 0]]), uri, mode="append")
    assert len(dataset.get_fragments()) == 2
    assert len(dataset.describe_indices()[0].segments) == 1
    # A later, exact-match append must not leak into this fixed snapshot.
    lance.write_dataset(_table([257], [[0, 0]]), uri, mode="append")
    filter_expr = "id != 0" if filtered else None
    for k in (1, 2):
        nearest = {"column": "vector", "q": [0, 0], "metric": "hamming", "k": k}
        expected = dataset.to_table(
            columns=["id", "_distance"],
            nearest=nearest,
            filter=filter_expr,
            prefilter=True,
        )
        actual = vector_search(
            dataset,
            nearest=nearest,
            columns=["id"],
            filter=filter_expr,
            scanner_options={"prefilter": True},
            num_workers=2,
        )
        assert isinstance(actual, pa.Table)
        assert "vector" not in actual.column_names
        assert actual["id"].to_pylist() == expected["id"].to_pylist()
        assert actual["_distance"].to_pylist() == expected["_distance"].to_pylist()
        assert 257 not in actual["id"].to_pylist()


@pytest.mark.parametrize("filter_expr", [None, "id > 0", "id < 0"])
def test_unindexed_hamming_search_matches_lance(
    tmp_path: Path, search_pool: None, filter_expr: str | None
) -> None:
    uri = tmp_path / "flat.lance"
    lance.write_dataset(_table([0], [[255, 0]]), uri)
    snapshot = lance.write_dataset(_table([1, 2], [[1, 1], [0, 1]]), uri, mode="append")
    lance.write_dataset(_table([3], [[0, 0]]), uri, mode="append")
    nearest = {"column": "vector", "q": [0, 0], "metric": "hamming", "k": 3}
    expected = snapshot.to_table(
        columns=["id", "_distance"], nearest=nearest, filter=filter_expr, prefilter=True
    )
    actual = vector_search(
        snapshot,
        nearest=nearest,
        columns=["id"],
        filter=filter_expr,
        scanner_options={"prefilter": True},
        num_workers=2,
    )
    assert isinstance(actual, pa.Table)
    assert actual["id"].to_pylist() == expected["id"].to_pylist()
    assert actual["_distance"].to_pylist() == expected["_distance"].to_pylist()


@pytest.mark.parametrize(
    "options", [{"fast_search": True}, {"include_unindexed": False}]
)
def test_indexed_only_hamming_search_excludes_appends(
    tmp_path: Path, search_pool: None, options: dict[str, Any]
) -> None:
    uri = tmp_path / "indexed.lance"
    dataset = lance.write_dataset(
        _table(list(range(256)), [[1, 1], *([[255, 255]] * 255)]), uri
    )
    dataset.create_index(
        "vector", index_type="IVF_FLAT", metric="hamming", num_partitions=1
    )
    dataset = lance.write_dataset(_table([256], [[0, 0]]), uri, mode="append")
    actual = vector_search(
        dataset,
        nearest={"column": "vector", "q": [0, 0], "metric": "hamming", "k": 1},
        columns=["id"],
        num_workers=2,
        **options,
    )
    assert isinstance(actual, pa.Table)
    assert actual["id"].to_pylist() == [0]
    assert actual["_distance"].to_pylist() == [2.0]
