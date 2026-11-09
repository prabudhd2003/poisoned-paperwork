"""Isolated final evaluation stage with full clean cohorts and exact source identity."""

from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path
from PIL import Image
from checkpoints import atomic_json, sha256_file, sha256_text
from experiment_utils import (
    initialize_run,
    merge_json_records,
    repo_root_from_config,
    write_worker_complete,
)
from final_protocol.evidence import TesseractOCR, infer
from final_protocol.reporting import summarize
from stages.defenses import _normalizer

VARIANTS = [
    "none",
    "jpeg",
    "resize_vote",
    "selective_multiview",
    "semantic_consistency",
    "ocr_only",
    "full_ocr",
    "full_crop",
    "evidence",
]


def runtime_versions():
    import importlib.metadata
    import platform

    return dict(
        python=platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "torchvision",
                "transformers",
                "pandas",
                "numpy",
                "Pillow",
            ]
        },
    )


def load_pages(items, root):
    pages = []
    for item in items:
        path = root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"image drift: {path}")
        with Image.open(path) as image:
            pages.append(image.convert("RGB").copy())
    if not pages:
        raise ValueError("empty document")
    return pages


def validate_config(config):
    if config.get("stage") != "final_protocol" or config.get("protocol_version") != 1:
        raise ValueError("requires final_protocol version 1")
    if not config.get("ready_for_production"):
        raise ValueError("freeze inputs before execution")
    if config["mode"] not in {"defense", "transfer", "adaptive"}:
        raise ValueError("unknown mode")
    if config["model_key"] not in {"qwen", "donut", "internvl"}:
        raise ValueError("unsupported model")
    if config["mode"] == "transfer" and config["model_key"] != "internvl":
        raise ValueError("transfer target must be independent InternVL")
    if config["mode"] != "transfer" and config["model_key"] == "internvl":
        raise ValueError("InternVL must not provide attack feedback")
    if config["mode"] == "defense" and config["conditions"] != VARIANTS:
        raise ValueError("frozen comparison suite differs")
    if config["mode"] == "transfer":
        from stages.transfer_robustness import TRANSFORMS

        if config["conditions"] != list(TRANSFORMS):
            raise ValueError("transfer transformation suite differs")
    if config["mode"] == "adaptive":
        if config["conditions"] != ["evidence"]:
            raise ValueError("adaptive must evaluate the actual evidence defense")
        if (
            config["whitebox"]["steps"] < 1
            or not 0 < config["whitebox"]["step_fraction"] <= 1
        ):
            raise ValueError("invalid whitebox budget")
        a = config["adaptive"]
        if (
            a["queries"] < 1
            or a["restart_interval"] < 1
            or a["objective"] not in {"targeted", "availability"}
        ):
            raise ValueError("invalid adaptive budget/objective")
    expected = {
        "qwen": (
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "66285546d2b821cf421d4f5eb2576359d3770cd3",
        ),
        "donut": (
            "naver-clova-ix/donut-base-finetuned-docvqa",
            "b19d2e332684b0e2d35d9144ce34047767335cf8",
        ),
        "internvl": (
            "OpenGVLab/InternVL2_5-4B",
            "2cf4a8158bbc40d35015e7c63b527890de4d27b3",
        ),
    }
    if (config["model"]["model_id"], config["model"]["revision"]) != expected[
        config["model_key"]
    ]:
        raise ValueError("unpinned model")


def read_plan(config, root):
    validate_config(config)
    path = root / config["plan"]["path"]
    if sha256_file(path) != config["plan"]["sha256"]:
        raise RuntimeError("plan hash drift")
    plan = json.loads(path.read_text())
    if len({d["record_id"] for d in plan}) != len(plan):
        raise ValueError("duplicate documents")
    counts = {}
    for document in plan:
        dataset = document["dataset_name"]
        counts[dataset] = counts.get(dataset, 0) + 1
        if config["model_key"] == "donut" and dataset == "resume_parsing_vision":
            raise ValueError("Donut checkpoint does not support resume comparison")
        keys = [a["source_id"] for a in document["attacks"]]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate source attack identity")
    if counts != config["expected_clean_counts"]:
        raise ValueError(f"clean cohort differs: {counts}")
    return plan


def preflight(config, root):
    plan = read_plan(config, root)
    if config.get("runtime_versions") != runtime_versions():
        raise RuntimeError(
            "runtime package versions differ from frozen preparation environment"
        )
    import numpy as np

    for document in plan:
        clean = load_pages(document["images"], root)
        for attack in document["attacks"]:
            pages = load_pages(attack["images"], root)
            if len(pages) != len(clean) or any(
                p.size != c.size for p, c in zip(pages, clean)
            ):
                raise ValueError("attack page dimensions differ")
            if any(
                np.abs(
                    np.asarray(p, dtype=np.int16) - np.asarray(c, dtype=np.int16)
                ).max()
                > attack["epsilon_255"]
                for p, c in zip(pages, clean)
            ):
                raise ValueError("saved PNG exceeds epsilon")
    if config["mode"] != "transfer":
        o = config["ocr"]
        version = subprocess.check_output(
            [o["executable"], "--version"], text=True
        ).splitlines()[0]
        if (
            version != o["version"]
            or sha256_file(o["executable"]) != o["executable_sha256"]
        ):
            raise ValueError("OCR executable/version drift")
        for language in o["language"].split("+"):
            if (
                sha256_file(Path(o["tessdata_dir"]) / (language + ".traineddata"))
                != o["traineddata_sha256"][language]
            ):
                raise ValueError("OCR weights drift")
    print(
        f"FINAL PREFLIGHT PASSED: {len(plan)} full-cohort documents; PNG hashes and bounds checked"
    )


def model(config):
    import torch

    key = config["model_key"]
    if key == "qwen":
        from models.qwen import QwenVL as Model
    elif key == "donut":
        from models.donut import DonutDocVQA as Model
    else:
        from models.internvl import InternVL as Model
    return Model(device=torch.device("cuda"), **config["model"])


def evaluate(pages, document, condition, wrapper, config):
    from defenses import evaluate_defense, evaluate_semantic_consistency
    from stages.transfer_robustness import _transform

    dataset = document["dataset_name"]
    task = "resume" if dataset == "resume_parsing_vision" else "receipt"
    prompt = config["prompts"][task]
    if config["mode"] == "transfer":
        started = time.perf_counter()
        prediction = _normalizer(dataset)(
            wrapper.predict(_transform(pages, condition), prompt)
        )
        return dict(
            prediction=prediction,
            abstained=prediction is None,
            vlm_calls=1,
            ocr_pages=0,
            runtime_seconds=time.perf_counter() - started,
        )
    if condition in {
        "jpeg",
        "resize_vote",
        "selective_multiview",
        "semantic_consistency",
    }:
        started = time.perf_counter()
        if condition == "semantic_consistency":
            result, _ = evaluate_semantic_consistency(
                pages=pages,
                record_id=document["record_id"],
                predict=wrapper.predict,
                normalize=_normalizer(dataset),
                transforms=config["defense"]["semantic_transforms"],
                prompts=config["semantic_prompts"][task],
                minimum_votes=config["defense"]["semantic_minimum_votes"],
            )
        else:
            result, _ = evaluate_defense(
                pages=pages,
                defense_id=condition,
                record_id=document["record_id"],
                predict=lambda value: wrapper.predict(value, prompt),
                normalize=_normalizer(dataset),
                config=config["defense"],
            )
        return dict(
            prediction=result.final_prediction,
            abstained=result.final_prediction is None,
            vlm_calls=1 + result.additional_vlm_passes,
            ocr_pages=0,
            runtime_seconds=time.perf_counter() - started,
        )
    return infer(
        pages,
        dataset,
        prompt,
        wrapper.predict,
        TesseractOCR(config["ocr"]),
        config["evidence"],
        condition,
    )


def trial_key(document, source_id, condition, config):
    return sha256_text(
        json.dumps(
            [
                config["protocol_version"],
                document["dataset_name"],
                document["record_id"],
                config["model_key"],
                source_id,
                condition,
            ],
            separators=(",", ":"),
        )
    )


def valid_record(path, provenance, root):
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if (
        any(record.get(k) != v for k, v in provenance.items())
        or record.get("status") != "complete"
    ):
        raise RuntimeError(f"resume record provenance differs: {path}")
    for item in record.get("images", []):
        if sha256_file(root / item["path"]) != item["sha256"]:
            raise RuntimeError("saved adaptive artifact drift")
    return True


def run(config, config_path, user_id, worker_id, num_workers):
    import torch
    import pandas as pd
    from final_protocol.adaptive import search

    preflight(config, repo_root_from_config(config_path))
    if (
        num_workers != 5
        or not torch.cuda.is_available()
        or "L40S" not in torch.cuda.get_device_name()
    ):
        raise RuntimeError("requires five worker shards on L40S")
    root = repo_root_from_config(config_path)
    plan = read_plan(config, root)
    output, shard, config_hash, commit, ids = initialize_run(
        root=root,
        config_path=Path(config_path),
        config=config,
        stage="final_protocol",
        worker_id=worker_id,
        num_workers=num_workers,
        record_ids=[d["record_id"] for d in plan],
    )
    wrapper = model(config)
    metric = None
    if config["mode"] == "adaptive":
        import lpips

        metric = lpips.LPIPS(net="alex").to(wrapper.device).eval()
        metric.requires_grad_(False)
    conditions = config["conditions"]
    expected = 0

    def work(document):
        nonlocal expected
        pages = load_pages(document["images"], root)
        # Independent full-document clean evaluation is also the transfer denominator.
        clean_cache = (
            shard / "clean_cache" / f"{sha256_text(document['record_id'])}.json"
        )
        cache_contract = dict(config_sha256=config_hash, git_commit=commit)
        if clean_cache.exists():
            cached = json.loads(clean_cache.read_text())
            if any(cached[k] != v for k, v in cache_contract.items()):
                raise RuntimeError("clean cache drift")
            target_correct = cached["correct"]
            clean_accepted = cached.get("clean_defense_accepted")
        else:
            task = (
                "resume"
                if document["dataset_name"] == "resume_parsing_vision"
                else "receipt"
            )
            raw = wrapper.predict(pages, config["prompts"][task])
            target_correct = (
                _normalizer(document["dataset_name"])(raw) == document["clean_target"]
            )
            clean_accepted = None
            if config["mode"] == "adaptive":
                clean_accepted = not evaluate(
                    pages, document, "evidence", wrapper, config
                )["abstained"]
            atomic_json(
                clean_cache,
                dict(
                    cache_contract,
                    correct=target_correct,
                    raw=raw,
                    clean_defense_accepted=clean_accepted,
                ),
            )
        if config["mode"] == "adaptive" and document["attacks"] and not target_correct:
            raise RuntimeError(
                "source clean prediction drift before adaptive optimization"
            )
        sources = [None] + document["attacks"]
        if config["mode"] == "adaptive":
            # One attack per source model/document/epsilon, seeds from all frozen families.
            sources = []
            for epsilon in [4, 8, 16]:
                matching = [
                    a
                    for a in document["attacks"]
                    if a["epsilon_255"] == epsilon
                    and a["source_model"] == config["model_key"]
                ]
                if matching:
                    sources.append(
                        dict(
                            matching[0],
                            source_id=f"adaptive-{epsilon}",
                            attack_type="evidence_query_"
                            + config["adaptive"]["objective"],
                            seeds=matching,
                        )
                    )
        for attack in sources:
            source_id = "clean" if attack is None else attack["source_id"]
            input_pages = (
                pages if attack is None else load_pages(attack["images"], root)
            )
            for condition in conditions:
                key = trial_key(document, source_id, condition, config)
                provenance = dict(
                    experiment_key=key,
                    config_sha256=config_hash,
                    git_commit=commit,
                    worker_id=worker_id,
                )
                path = shard / "records" / f"{key}.json"
                expected += 1
                if valid_record(path, provenance, root):
                    continue
                extra = {}
                if config["mode"] == "adaptive":
                    settings = dict(
                        config["adaptive"],
                        seed=int(sha256_text(key + str(config["seed"]))[:8], 16),
                    )
                    from final_protocol.whitebox import optimize

                    task = (
                        "resume"
                        if document["dataset_name"] == "resume_parsing_vision"
                        else "receipt"
                    )
                    wb_started = time.perf_counter()
                    wb_seed, wb_state = optimize(
                        wrapper,
                        pages,
                        config["prompts"][task],
                        attack["attack_target"],
                        document["dataset_name"],
                        TesseractOCR(config["ocr"]),
                        config["evidence"],
                        attack["epsilon_255"],
                        config["whitebox"],
                        shard / "checkpoints" / f"{key}_whitebox.pkl",
                        provenance,
                    )
                    wb_seconds = time.perf_counter() - wb_started
                    final, state = search(
                        pages,
                        [wb_seed]
                        + [load_pages(a["images"], root) for a in attack["seeds"]],
                        lambda p: evaluate(p, document, "evidence", wrapper, config),
                        attack["attack_target"],
                        attack["epsilon_255"],
                        settings,
                        shard / "checkpoints" / f"{key}.pkl",
                        provenance,
                    )
                    saved = []
                    for index, page in enumerate(final):
                        image_path = shard / "images" / key / f"{index}.png"
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        page.save(image_path)
                        saved.append(
                            dict(
                                path=str(image_path.relative_to(root)),
                                sha256=sha256_file(image_path),
                            )
                        )
                    input_pages = load_pages(saved, root)
                    from attacks.qwen import _perceptibility
                    from models.donut import pil_to_rgb_tensor
                    import numpy as np

                    perceptual = []
                    for clean_page, adv_page in zip(pages, input_pages):
                        linf = int(
                            np.abs(
                                np.asarray(clean_page, dtype=np.int16)
                                - np.asarray(adv_page, dtype=np.int16)
                            ).max()
                        )
                        if linf > attack["epsilon_255"]:
                            raise RuntimeError("adaptive saved PNG exceeds epsilon")
                        lp, ss = _perceptibility(
                            pil_to_rgb_tensor(clean_page, wrapper.device),
                            pil_to_rgb_tensor(adv_page, wrapper.device),
                            metric,
                        )
                        perceptual.append(dict(linf_255=linf, lpips=lp, ssim=ss))
                    extra = dict(
                        images=saved,
                        per_page_metrics=perceptual,
                        queries=state["queries"],
                        budget_curve=state["curve"],
                        checkpoint_score=list(state["best_score"]),
                        attack_runtime_seconds=state["runtime_seconds"],
                        attack_vlm_calls=state["vlm_calls"],
                        attack_ocr_pages=state["ocr_pages"],
                        whitebox_steps=wb_state["step"],
                        whitebox_losses=wb_state["losses"],
                        whitebox_invocation_seconds=wb_seconds,
                        whitebox_runtime_seconds=wb_state["runtime_seconds"],
                        whitebox_gradient_model_passes=wb_state[
                            "gradient_model_passes"
                        ],
                        whitebox_ocr_pages=wb_state["ocr_pages"],
                    )
                result = evaluate(input_pages, document, condition, wrapper, config)
                row = dict(
                    provenance,
                    **result,
                    **extra,
                    run_id=config["run_id"],
                    stage="final_protocol",
                    record_id=document["record_id"],
                    dataset_name=document["dataset_name"],
                    experiment_split=document["experiment_split"],
                    model_key=config["model_key"],
                    condition=condition,
                    example_type="clean" if attack is None else "attack",
                    clean_target=document["clean_target"],
                    attack_target="" if attack is None else attack["attack_target"],
                    source_model="none" if attack is None else attack["source_model"],
                    attack_type="clean" if attack is None else attack["attack_type"],
                    source_id=source_id,
                    epsilon_255=0 if attack is None else attack["epsilon_255"],
                    source_clean_correct=False if attack is None else True,
                    target_clean_correct=target_correct,
                    clean_defense_accepted=clean_accepted,
                    source_attack_success=(
                        None
                        if attack is None or config["mode"] == "adaptive"
                        else attack["source_attack_success"]
                    ),
                    status="complete",
                    user_id=user_id,
                )
                if config["mode"] == "adaptive":
                    row["png_target_success"] = (
                        result["full_prediction"] == attack["attack_target"]
                    )
                    row["accepted_target_success"] = (
                        result["prediction"] == attack["attack_target"]
                        and not result["abstained"]
                    )
                atomic_json(path, row)

    try:
        for document in plan:
            if document["record_id"] in ids:
                work(document)
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

    def count(d):
        if config["mode"] == "adaptive":
            return len(
                {
                    a["epsilon_255"]
                    for a in d["attacks"]
                    if a["source_model"] == config["model_key"]
                    and a["epsilon_255"] in [4, 8, 16]
                }
            )
        return 1 + len(d["attacks"])

    if all(
        (output / "shards" / f"worker_{i:02d}" / "WORKER_COMPLETE.json").exists()
        for i in range(5)
    ):
        for i in range(5):
            for path in (output / "shards" / f"worker_{i:02d}" / "records").glob(
                "*.json"
            ):
                valid_record(
                    path,
                    dict(
                        config_sha256=config_hash,
                        git_commit=commit,
                        worker_id=i,
                        experiment_key=path.stem,
                    ),
                    root,
                )
    if merge_json_records(
        output,
        config=config,
        config_hash=config_hash,
        num_workers=5,
        expected_records=sum(count(d) for d in plan) * len(conditions),
        sort_columns=["dataset_name", "record_id", "source_id", "condition"],
    ):
        frame = pd.read_parquet(output / "merged/records.parquet")
        summarize(frame).to_csv(output / "merged/final_summary.csv", index=False)
