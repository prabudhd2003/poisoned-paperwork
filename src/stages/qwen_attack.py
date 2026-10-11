"""Router for the two independently owned Qwen attack tracks.

``config["track"]`` selects the implementation:

* ``"receipt"`` -> ``stages.qwen_attack_receipt`` (SROIE, CORD; primary study)
* ``"resume"``  -> ``stages.qwen_attack_resume``  (resume degrees; case study)

Both track modules must use the shared attack core in ``attacks.qwen`` and the
shared sharding helper. Put track logic in the track module, never here. See
docs/06_qwen_attacks.md.
"""

from __future__ import annotations

import importlib

TRACKS = {
    "receipt": ("stages.qwen_attack_receipt", {"sroie", "cord_v2"}),
    "resume": ("stages.qwen_attack_resume", {"resume_parsing_vision"}),
}


def _validate(config) -> tuple[str, str]:
    if config.get("stage") != "qwen_attack":
        raise ValueError("config['stage'] must be 'qwen_attack'")
    track = config.get("track")
    if track not in TRACKS:
        raise ValueError(f"config['track'] must be one of {sorted(TRACKS)}, got {track!r}")
    module_name, allowed = TRACKS[track]
    datasets = set(config.get("datasets") or [])
    if not datasets or not datasets <= allowed:
        raise ValueError(
            f"track {track!r} accepts datasets {sorted(allowed)}, got {sorted(datasets)}"
        )
    return track, module_name


def run(config, config_path, user_id, worker_id, num_workers):
    track, module_name = _validate(config)
    if num_workers != 5 and config.get("mode") != "smoke":
        raise ValueError("Only smoke configs may run with fewer than five workers")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise SystemExit(
                f"Qwen {track} track is not implemented yet; its owner must create "
                f"src/{module_name.replace('.', '/')}.py"
            ) from exc
        raise
    return module.run(
        config=config,
        config_path=config_path,
        user_id=user_id,
        worker_id=worker_id,
        num_workers=num_workers,
    )
