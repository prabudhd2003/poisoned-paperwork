"""Full public-dataset multi-defense EOT/BPDA experiment for Qwen or Donut."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from checkpoints import atomic_json, sha256_file, sha256_text
from experiment_utils import (
    initialize_run,
    merge_json_records,
    repo_root_from_config,
    write_worker_complete,
)
from models.donut import (
    DEFAULT_MODEL_ID as DONUT_MODEL_ID,
    DEFAULT_REVISION as DONUT_REVISION,
)
from models.qwen import DEFAULT_MODEL_ID as QWEN_MODEL_ID, DEFAULT_REVISION as QWEN_REVISION


QWEN_DATASETS = ("cord_v2", "resume_parsing_vision", "sroie")
QWEN_DOCUMENTS = {"cord_v2": 96, "resume_parsing_vision": 74, "sroie": 123}
DONUT_DATASETS = ("cord_v2", "sroie")
DONUT_DOCUMENTS = {"cord_v2": 75, "sroie": 46}
EPSILONS = [4, 8, 16]


def _protocol(config: dict[str, Any]):
    model_key = str(config.get("model_key", "qwen"))
    if model_key == "qwen":
        return model_key, QWEN_DATASETS, QWEN_DOCUMENTS, QWEN_MODEL_ID, QWEN_REVISION
    if model_key == "donut":
        return model_key, DONUT_DATASETS, DONUT_DOCUMENTS, DONUT_MODEL_ID, DONUT_REVISION
    raise ValueError("adaptive attack model_key must be qwen or donut")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("stage") != "jpeg_eot_attack":
        raise ValueError("JPEG-EOT config requires stage=jpeg_eot_attack")
    if config.get("ready_for_production") is not True:
        raise RuntimeError("JPEG-EOT run is gated until source hashes are frozen")
    model_key, datasets, expected_documents, model_id, revision = _protocol(config)
    if tuple(config.get("datasets", ())) != datasets:
        raise ValueError(f"datasets must equal {list(datasets)}")
    if config.get("model", {}).get("model_id") != model_id:
        raise ValueError(f"adaptive attack requires the pinned {model_key} model")
    if config.get("model", {}).get("revision") != revision:
        raise ValueError(f"adaptive attack requires the pinned {model_key} revision")
    if config.get("expected_documents_by_dataset") != expected_documents:
        raise ValueError(f"expected document counts must equal {expected_documents}")
    attack = config.get("attack", {})
    method = attack.get("method")
    if method not in {"uniform_multiview_eot", "frequency_hardview_minimax_eot"}:
        raise ValueError("unknown adaptive attack method")
    base_track = {
        "uniform_multiview_eot": "uniform_eot",
        "frequency_hardview_minimax_eot": "hardview_eot",
    }[method]
    expected_track = base_track if model_key == "qwen" else f"donut_{base_track}"
    if config.get("track") != expected_track:
        raise ValueError(f"adaptive attack track must equal {expected_track}")
    sampling = attack.get("view_sampling")
    if sampling not in {"uniform", "hard_view"}:
        raise ValueError("view_sampling must be uniform or hard_view")
    if attack.get("epsilons_255") != EPSILONS:
        raise ValueError("JPEG-EOT study must use epsilons [4,8,16]; epsilon 2 is the existing negative control")
    if int(attack.get("max_steps", 0)) < 1:
        raise ValueError("max_steps must be positive")
    if int(attack.get("checkpoint_interval", 0)) < 1:
        raise ValueError("checkpoint_interval must be positive")
    if (
        int(attack.get("evaluation_interval", 0)) < 1
        or int(attack["evaluation_interval"]) % int(attack["checkpoint_interval"])
    ):
        raise ValueError("evaluation_interval must be a checkpoint multiple")
    steps = attack.get("step_size_255", {})
    if any(float(steps.get(str(epsilon), 0)) <= 0 for epsilon in EPSILONS):
        raise ValueError("step_size_255 must define every epsilon")
    views = attack.get("training_views", [])
    if len(views) < 5 or any(
        not isinstance(view, dict)
        or not isinstance(view.get("transform"), str)
        or not isinstance(view.get("prompt_index"), int)
        or not 0 <= view["prompt_index"] < 5
        for view in views
    ):
        raise ValueError("training_views must define valid visual/prompt EOT pairs")
    required = {"identity", "jpeg_q85", "resize_0.85", "blur_0.5", "brightness_0.95"}
    if not required.issubset({view["transform"] for view in views}):
        raise ValueError("training_views do not cover every completed visual defense")
    if attack.get("include_exact_resize_vote_views") is not True:
        raise ValueError("the adaptive attack must include exact resize-vote views")
    samples = int(attack.get("eot_samples_per_step", 0))
    if not 1 <= samples <= len(views):
        raise ValueError("eot_samples_per_step is outside the view distribution")
    frequency = attack.get("frequency_aware", {})
    if not isinstance(frequency.get("enabled"), bool):
        raise ValueError("frequency_aware.enabled must be boolean")
    if not 0 < float(frequency.get("cutoff", 0)) <= 1:
        raise ValueError("frequency cutoff must be in (0,1]")
    if any(
        not 0 <= float(frequency.get(name, -1)) <= 1
        for name in ("gradient_mix", "delta_mix")
    ):
        raise ValueError("frequency mixes must be in [0,1]")
    if method == "uniform_multiview_eot" and (
        sampling != "uniform" or frequency["enabled"]
    ):
        raise ValueError("uniform EOT method drifted")
    if method == "frequency_hardview_minimax_eot":
        if sampling != "hard_view" or not frequency["enabled"]:
            raise ValueError("frequency hard-view method drifted")
        temperature = float(attack.get("hard_view_temperature", 0))
        floor = float(attack.get("hard_view_probability_floor", -1))
        if temperature <= 0 or not 0 <= floor < 1 / len(views):
            raise ValueError("hard-view temperature or probability floor is invalid")
    prompts = config.get("semantic_prompts", {})
    tasks = ("receipt", "resume") if model_key == "qwen" else ("receipt",)
    if any(len(prompts.get(task, [])) != 5 for task in tasks):
        raise ValueError("five frozen semantic prompts are required per task")
    defense = config.get("defense", {})
    if defense.get("semantic_transforms") != [
        "identity", "jpeg_q85", "resize_0.85", "blur_0.5", "brightness_0.95"
    ]:
        raise ValueError("semantic defense transforms drifted from the completed study")
    limit = config.get("document_limit_per_dataset")
    if limit is not None:
        raise ValueError("the full study requires document_limit_per_dataset=null")
    tolerance = config.get("processor_tolerance", {})
    if any(float(tolerance.get(key, -1)) < 0 for key in ("max", "mean", "surrogate_mean")):
        raise ValueError("processor_tolerance is incomplete")


def _load_plan(root: Path, config: dict[str, Any]):
    import pandas as pd

    frames = []
    for source in config.get("attack_sources", []):
        path = Path(source["records"])
        path = path if path.is_absolute() else root / path
        expected_hash = source.get("sha256")
        if not path.is_file():
            raise FileNotFoundError(f"JPEG-EOT source missing: {path}")
        if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
            raise RuntimeError(f"freeze the exact JPEG-EOT source hash: {path}")
        frame = pd.read_parquet(path)
        frame["source_name"] = source["name"]
        frames.append(frame)
    if not frames:
        raise RuntimeError("JPEG-EOT has no frozen attack sources")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "record_id", "dataset_name", "experiment_split", "clean_target",
        "attack_target", "prompt_text", "status", "epsilon_255", "model_id",
        "model_revision",
    }
    missing = sorted(required - set(combined.columns))
    if missing:
        raise ValueError(f"JPEG-EOT source tables are missing: {missing}")
    _, datasets, expected_documents, model_id, revision = _protocol(config)
    combined = combined.loc[
        combined.status.eq("complete") & combined.dataset_name.isin(datasets)
    ].copy()
    if set(combined.model_id.astype(str)) != {model_id}:
        raise RuntimeError("adaptive source model ID differs from the pinned model")
    if set(combined.model_revision.astype(str)) != {revision}:
        raise RuntimeError("adaptive source revision differs from the pinned model")
    observed = combined.groupby("dataset_name").record_id.nunique().to_dict()
    if observed != expected_documents:
        raise RuntimeError(f"source document counts differ: {observed} != {expected_documents}")
    for record_id, group in combined.groupby("record_id"):
        for field in (
            "dataset_name", "experiment_split", "clean_target", "attack_target",
            "prompt_text",
        ):
            if group[field].nunique(dropna=False) != 1:
                raise RuntimeError(f"source rows disagree for {record_id}: {field}")
        source_epsilons = sorted(group.epsilon_255.astype(int).unique())
        if source_epsilons != [2, 4, 8, 16]:
            raise RuntimeError(f"source attack grid differs for {record_id}: {source_epsilons}")
    representatives = (
        combined.sort_values(["record_id", "epsilon_255"])
        .drop_duplicates("record_id")
        .set_index("record_id", drop=False)
    )
    limit = config.get("document_limit_per_dataset")
    if limit is not None:
        selected = []
        for dataset, group in representatives.groupby("dataset_name"):
            ranked = group.copy()
            ranked["selection_hash"] = ranked.record_id.map(
                lambda record_id: sha256_text(
                    f"jpeg-eot|{config['seed']}|{dataset}|{record_id}"
                )
            )
            selected.append(ranked.sort_values("selection_hash").head(int(limit)))
        representatives = (
            pd.concat(selected, ignore_index=True)
            .sort_values("record_id")
            .set_index("record_id", drop=False)
        )
    return representatives


def preflight(config: dict[str, Any], root: Path) -> None:
    validate_config(config)
    selected = _load_plan(Path(root).resolve(), config)
    counts = selected.groupby("dataset_name").record_id.nunique().to_dict()
    print(
        f"JPEG-EOT PREFLIGHT PASSED: {len(selected)} documents, "
        f"epsilons={EPSILONS}, datasets={counts}"
    )


def _record_valid(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"JPEG-EOT record provenance mismatch in {key}: {path}")
    if value.get("status") != "complete":
        raise RuntimeError(f"JPEG-EOT record is not complete: {path}")
    images = value.get("images")
    if not isinstance(images, list) or not images:
        raise RuntimeError(f"JPEG-EOT record has no PNG images: {path}")
    for image in images:
        image_path = Path(image["path"])
        if not image_path.is_file() or sha256_file(image_path) != image["sha256"]:
            raise RuntimeError(f"JPEG-EOT PNG is missing or changed: {image_path}")
    return True


def _key(record_id: str, epsilon: int, seed: int) -> str:
    return sha256_text(f"jpeg-eot|{record_id}|{epsilon}|{seed}")


def run(config, config_path, user_id, worker_id, num_workers):
    import lpips
    import torch

    from attacks.jpeg_eot import attack
    from models.donut import DonutDocVQA
    from models.qwen import QwenVL
    from stages.defenses import _load_clean_pages, _normalizer

    validate_config(config)
    if num_workers != 5 or not torch.cuda.is_available():
        raise RuntimeError("JPEG-EOT requires five CARC L40S workers")
    config_path = Path(config_path).resolve()
    root = repo_root_from_config(config_path)
    selected = _load_plan(root, config)
    splits = selected.experiment_split.to_dict()
    output, shard, config_hash, commit, ids = initialize_run(
        root=root,
        config_path=config_path,
        config=config,
        stage="jpeg_eot_attack",
        worker_id=worker_id,
        num_workers=num_workers,
        record_ids=selected.index,
    )
    pages_by_id = _load_clean_pages(root, config, {record_id: splits[record_id] for record_id in ids})
    model = config["model"]
    model_key, _, _, _, _ = _protocol(config)
    if model_key == "qwen":
        wrapper = QwenVL(
            device=torch.device("cuda"),
            model_id=model["model_id"],
            revision=model["revision"],
            precision=model.get("precision", "bfloat16"),
            min_pixels=int(model.get("min_pixels", 200704)),
            max_pixels=int(model.get("max_pixels", 1003520)),
            max_new_tokens=int(model.get("max_new_tokens", 32)),
        )
    else:
        wrapper = DonutDocVQA(
            device=torch.device("cuda"),
            model_id=model["model_id"],
            revision=model["revision"],
            precision=model.get("precision", "float32"),
            max_length=int(model.get("max_length", 64)),
        )
    metric = lpips.LPIPS(net="alex").to(wrapper.device).eval()
    metric.requires_grad_(False)
    expected = 0
    try:
        for record_id in ids:
            row = selected.loc[record_id].to_dict()
            pages = [page.convert("RGB") for page in pages_by_id[record_id]]
            normalize = _normalizer(str(row["dataset_name"]))
            for epsilon in EPSILONS:
                key = _key(record_id, epsilon, int(config["seed"]))
                provenance = {
                    "run_id": config["run_id"],
                    "stage": "jpeg_eot_attack",
                    "config_sha256": config_hash,
                    "git_commit": commit,
                    "worker_id": worker_id,
                }
                expected_record = {
                    **provenance,
                    "experiment_key": key,
                    "record_id": record_id,
                    "epsilon_255": epsilon,
                }
                path = shard / "records" / f"{key}.json"
                expected += 1
                if _record_valid(path, expected_record):
                    continue
                result = attack(
                    wrapper,
                    pages,
                    row,
                    config,
                    epsilon,
                    shard,
                    key,
                    provenance,
                    metric,
                    normalize,
                )
                atomic_json(path, {
                    **expected_record,
                    **result,
                    "dataset_name": str(row["dataset_name"]),
                    "experiment_split": str(row["experiment_split"]),
                    "model_key": model_key,
                    "model_id": wrapper.model_id,
                    "model_revision": wrapper.revision,
                    "clean_target": str(row["clean_target"]),
                    "attack_target": str(row["attack_target"]),
                    "prompt_text": str(row["prompt_text"]),
                    "attack_family": str(config["attack"]["method"]),
                    "user_id": user_id,
                    "status": "complete",
                    "error": None,
                })
                _record_valid(path, expected_record)
                atomic_json(shard / "progress.json", {
                    "completed_experiments": len(list((shard / "records").glob("*.json"))),
                    "last_completed": key,
                })
    finally:
        wrapper.unload()
    write_worker_complete(
        shard,
        config=config,
        config_hash=config_hash,
        commit=commit,
        worker_id=worker_id,
        expected_records=expected,
    )
    if merge_json_records(
        output,
        config=config,
        config_hash=config_hash,
        num_workers=num_workers,
        expected_records=len(selected) * len(EPSILONS),
        sort_columns=["dataset_name", "record_id", "epsilon_255"],
    ):
        _summarize(output)


def _summarize(output: Path) -> None:
    import pandas as pd

    frame = pd.read_parquet(output / "merged" / "records.parquet")
    groups = frame.groupby(["dataset_name", "epsilon_255"]).agg(
        attempted=("experiment_key", "size"),
        png_successes=("png_target_success", "sum"),
        jpeg85_successes=("primary_jpeg_success", "sum"),
        resize_vote_successes=("resize_vote_success", "sum"),
        selective_multiview_successes=("selective_multiview_success", "sum"),
        semantic_consistency_successes=("semantic_consistency_success", "sum"),
        all_defenses_successes=("all_defenses_success", "sum"),
        mean_defense_survival=("defense_success_fraction", "mean"),
        median_lpips=("lpips", "median"),
        median_ssim=("ssim", "median"),
        median_runtime_seconds=("runtime_seconds", "median"),
    ).reset_index()
    groups["jpeg85_asr"] = groups.jpeg85_successes / groups.attempted
    groups["resize_vote_asr"] = groups.resize_vote_successes / groups.attempted
    groups["selective_multiview_asr"] = (
        groups.selective_multiview_successes / groups.attempted
    )
    groups["semantic_consistency_asr"] = (
        groups.semantic_consistency_successes / groups.attempted
    )
    groups["all_defenses_asr"] = groups.all_defenses_successes / groups.attempted
    atomic_json(output / "merged" / "jpeg_eot_summary.json", {
        "attack_method": str(frame.attack_method.iloc[0]),
        "groups": json.loads(groups.to_json(orient="records")),
        "interpretation": (
            "The optimized artifact is a bounded PNG. Each ASR is measured after "
            "the exact completed JPEG, resize-vote, selective-multiview, or "
            "cross-view/cross-prompt semantic-consistency defense."
        ),
    })
