"""Router wrapper for the collected-receipt Qwen external-test attack."""

from pathlib import Path


def preflight(config, root: Path):
    from stages.collected_attacks import preflight as check
    return check(config, root, "qwen")


def run(config, config_path, user_id, worker_id, num_workers):
    from stages.collected_attacks import run_qwen
    return run_qwen(config, config_path, user_id, worker_id, num_workers)

