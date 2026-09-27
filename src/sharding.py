"""Deterministic document sharding shared by all GPU stages."""

from collections.abc import Iterable


def assigned_record_ids(
    record_ids: Iterable[str], worker_id: int, num_workers: int = 5
) -> list[str]:
    """Assign sorted record IDs by position modulo the worker count."""
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    if not 0 <= worker_id < num_workers:
        raise ValueError("worker_id must be in [0, num_workers)")

    unique_ids = sorted(set(record_ids))
    return [
        record_id
        for index, record_id in enumerate(unique_ids)
        if index % num_workers == worker_id
    ]
