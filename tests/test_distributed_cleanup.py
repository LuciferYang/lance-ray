"""Tests for lance_ray.cleanup."""

import os
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import lance
import lance_ray as lr
import pyarrow as pa
import pytest
import ray
from lance_ray import cleanup as cleanup_mod

requires_distributed_cleanup = pytest.mark.skipif(
    not cleanup_mod._supports_distributed_cleanup(),
    reason="installed pylance cannot distribute a cleanup",
)

_TABLE = pa.table({"id": list(range(100)), "value": [f"v{i}" for i in range(100)]})


def _write_versions(uri: str) -> None:
    """Old versions with an index, deletions and an aged orphan, then one more."""
    lance.write_dataset(_TABLE, uri, max_rows_per_file=25)
    lance.write_dataset(_TABLE, uri, mode="overwrite", max_rows_per_file=25)
    dataset = lance.dataset(uri)
    dataset.create_scalar_index("id", "BTREE")
    dataset.delete("id < 10")
    lance.write_dataset(_TABLE, uri, mode="overwrite", max_rows_per_file=25)
    orphan = Path(uri) / "data" / "orphan.lance"
    orphan.write_bytes(b"orphan")
    eight_days_ago = time.time() - timedelta(days=8).total_seconds()
    os.utime(orphan, (eight_days_ago, eight_days_ago))


def _files(uri: str) -> set[str]:
    return {str(path) for path in Path(uri).rglob("*") if path.is_file()}


@pytest.fixture
def uri(tmp_path: Path) -> str:
    uri = str(tmp_path / "dataset")
    _write_versions(uri)
    return uri


def _expected(uri: str) -> tuple[Any, set[str]]:
    """Single-process explanation, and the files left after cleaning by it."""
    explanation = lance.dataset(uri).explain_cleanup_old_versions(
        older_than=timedelta(0), include_files=True, max_files=100_000
    )
    removed = {"/" + file.path for file in explanation.candidate_files}
    assert removed <= _files(uri)
    return explanation, _files(uri) - removed


@requires_distributed_cleanup
@pytest.mark.parametrize("max_tasks", [1, 64])
def test_matches_single_process_cleanup(uri: str, max_tasks: int) -> None:
    expected, remaining = _expected(uri)
    assert expected.stats.old_versions == 4
    assert expected.stats.data_files_removed > 0
    assert expected.stats.index_files_removed > 0
    assert expected.stats.deletion_files_removed > 0

    explanation = lr.explain_cleanup_old_versions(
        uri, older_than=timedelta(0), num_workers=2, max_tasks=max_tasks
    )
    assert explanation.stats.bytes_removed == expected.stats.bytes_removed
    assert sorted(f.path for f in explanation.candidate_files) == sorted(
        f.path for f in expected.candidate_files
    )

    stats = lr.cleanup_old_versions(
        uri, older_than=timedelta(0), num_workers=2, max_tasks=max_tasks
    )
    assert stats.old_versions == expected.stats.old_versions
    assert stats.data_files_removed == expected.stats.data_files_removed
    assert stats.bytes_removed == expected.stats.bytes_removed
    assert _files(uri) == remaining
    assert lance.dataset(uri).to_table() == _TABLE


@requires_distributed_cleanup
def test_cleanup_with_directory_namespace(tmp_path: Path) -> None:
    import lance_namespace as ln
    from lance_namespace import DescribeTableRequest

    table_id = ["cleanup_table"]
    namespace: dict[str, Any] = {
        "namespace_impl": "dir",
        "namespace_properties": {"root": str(tmp_path)},
    }
    modes: list[Literal["create", "overwrite"]] = ["create", "overwrite"]
    for mode in modes:
        lr.write_lance(
            ray.data.from_arrow(_TABLE), table_id=table_id, mode=mode, **namespace
        )
    location = (
        ln.connect("dir", {"root": str(tmp_path)})
        .describe_table(DescribeTableRequest(id=table_id))
        .location
    )
    assert len(lance.dataset(location).versions()) == 2

    stats = lr.cleanup_old_versions(
        table_id=table_id, older_than=timedelta(0), num_workers=2, **namespace
    )
    assert stats.old_versions == 1
    assert len(lance.dataset(location).versions()) == 1
    assert lance.dataset(location).to_table().sort_by("id") == _TABLE


def _failing_first_attempt(marker_dir: Path) -> Callable[..., Any]:
    """Wraps the sweep handler so each task's first attempt fails."""
    real = cleanup_mod._sweep_handler
    path = str(marker_dir)

    def create(*args: Any) -> Any:
        run = real(*args)

        def handler(task: Any) -> dict[str, Any]:
            marker = os.path.join(path, str(task.index))
            if not os.path.exists(marker):
                open(marker, "w").close()
                return {"status": "error", "error": "injected"}
            result: dict[str, Any] = run(task)
            return result

        return handler

    return create


@requires_distributed_cleanup
def test_failed_task_is_retried(uri: str, tmp_path: Path) -> None:
    expected, remaining = _expected(uri)
    with patch.object(cleanup_mod, "_sweep_handler", _failing_first_attempt(tmp_path)):
        stats = lr.cleanup_old_versions(
            uri, older_than=timedelta(0), num_workers=2, max_task_retries=1
        )
    assert stats.old_versions == expected.stats.old_versions
    assert _files(uri) == remaining


@requires_distributed_cleanup
def test_failed_task_raises(uri: str, tmp_path: Path) -> None:
    _, remaining = _expected(uri)
    with (
        patch.object(cleanup_mod, "_sweep_handler", _failing_first_attempt(tmp_path)),
        pytest.raises(RuntimeError, match="files these tasks left"),
    ):
        lr.cleanup_old_versions(
            uri, older_than=timedelta(0), num_workers=2, max_task_retries=0
        )
    # The old versions went with the plan; the dataset is intact.
    assert len(lance.dataset(uri).versions()) == 1
    assert lance.dataset(uri).to_table() == _TABLE
    # No version references what the failed tasks left, so only an unverified
    # cleanup removes it before it is seven days old.
    stats = lr.cleanup_old_versions(
        uri, older_than=timedelta(0), delete_unverified=True
    )
    assert stats.old_versions == 0
    assert _files(uri) == remaining


@requires_distributed_cleanup
@pytest.mark.parametrize(
    ("rate", "num_workers", "expected"), [(None, 2, None), (100, 4, 25), (3, 4, 1)]
)
def test_rate_limit_is_split_between_workers(
    uri: str, rate: Any, num_workers: int, expected: Any
) -> None:
    rates: list[Any] = []
    real = cleanup_mod._sweep_handler

    def recording(dataset_args: Any, concurrency: Any, worker_rate: Any) -> Any:
        rates.append(worker_rate)
        return real(dataset_args, concurrency, worker_rate)

    with patch.object(cleanup_mod, "_sweep_handler", recording):
        lr.cleanup_old_versions(
            uri,
            older_than=timedelta(0),
            num_workers=num_workers,
            delete_rate_limit=rate,
        )
    assert rates == [expected]


@pytest.mark.parametrize("explain", [False, True])
def test_falls_back_without_distributed_cleanup(uri: str, explain: bool) -> None:
    expected, remaining = _expected(uri)
    with patch.object(cleanup_mod, "_supports_distributed_cleanup", return_value=False):
        if explain:
            result = lr.explain_cleanup_old_versions(uri, older_than=timedelta(0))
            assert result.stats.bytes_removed == expected.stats.bytes_removed
            return
        stats = lr.cleanup_old_versions(uri, older_than=timedelta(0))
    assert stats.old_versions == expected.stats.old_versions
    assert _files(uri) == remaining


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_workers": 0}, "num_workers"),
        ({"max_tasks": 0}, "max_tasks"),
        ({"max_task_retries": -1}, "max_task_retries"),
    ],
)
def test_rejects_invalid_arguments(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        lr.cleanup_old_versions(str(tmp_path / "missing"), **kwargs)
