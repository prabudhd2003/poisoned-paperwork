"""Five-shard execution and dependency-aware strict merge for CSEV."""

import copy
import fcntl
import json
import time
from pathlib import Path

from checkpoints import atomic_json, sha256_file
from csev.defense import Verifier, prediction
from csev.study import digest, calibrate
from stages.final_protocol import load_pages, evaluate, model


def output(config):
    return Path(config["output"])


def calibration(config):
    p = output(config) / "calibration.json"
    result = json.loads(p.read_text())
    if result["study_hash"] != digest(config):
        raise ValueError("calibration belongs to another study")
    dev = output(config) / "development/merged.json"
    if sha256_file(dev) != result["development_sha256"]:
        raise ValueError("development results changed")
    return result


def tasks(config, plan, phase):
    result = []
    if phase == "development":
        result = [
            dict(document=d, family="clean", epsilon=0, images=d["images"])
            for d in plan["development"]
        ]
    elif phase == "evaluation":
        for d in plan["evaluation"]:
            result.append(
                dict(document=d, family="clean", epsilon=0, images=d["images"])
            )
            result.extend(dict(document=d, **a) for a in d["attacks"])
    elif phase == "adaptive":
        result = [
            dict(
                document=d,
                family="direct_adaptive",
                epsilon=e,
                restart=r,
                images=d["images"],
            )
            for d in plan["evaluation"]
            if d["record_id"] in plan["pilot_ids"]
            for e in config["settings"]["adaptive"]["epsilons"]
            for r in range(config["settings"]["adaptive"]["restarts"])
        ]
    elif phase == "replay":
        merged = json.loads((output(config) / "adaptive/merged.json").read_text())
        if merged["signature"] != signature(config, "adaptive"):
            raise ValueError("adaptive merge drift")
        groups = {}
        docs = {d["record_id"]: d for d in plan["evaluation"]}
        for r in merged["records"]:
            groups.setdefault((r["record_id"], r["epsilon"]), []).append(r)
        for (rid, e), rs in sorted(groups.items()):
            # Best restart chosen by CSEV target success, then frozen actual rank.
            chosen = max(rs, key=lambda r: tuple(r["best_rank"]))
            result.append(
                dict(
                    document=docs[rid],
                    family="direct_adaptive",
                    epsilon=e,
                    images=chosen["images"],
                    selected_restart=chosen["restart"],
                )
            )
    else:
        raise ValueError(phase)
    for t in result:
        t["key"] = digest(
            [
                phase,
                t["document"]["record_id"],
                t["family"],
                t["epsilon"],
                t.get("restart"),
            ]
        )
    return sorted(result, key=lambda t: t["key"])


def signature(config, phase):
    parts = [config, phase]
    if phase != "development":
        parts.append(calibration(config))
    if phase == "replay":
        parts.append(sha256_file(output(config) / "adaptive/merged.json"))
    return digest(parts)


def score_record(pred, d):
    return dict(
        prediction=pred,
        accepted=pred is not None,
        correct=pred is not None and pred == d["clean_target"],
        targeted=pred is not None and pred == d["frozen_attack_target"],
    )


def comparisons(image, d, verifier, wrapper, config, tau):
    obs = verifier(image, d["dataset_name"])
    legacy = copy.deepcopy(config["legacy"])
    if d["dataset_name"] == "collected_receipts":
        legacy["ocr"]["language"] = "eng"
    from types import SimpleNamespace

    inference_wrapper = SimpleNamespace(predict=verifier.predict)
    results = []
    for condition in config["settings"]["conditions"]:
        if condition == "csev":
            pred = prediction(obs, tau)
            cost = obs
        elif condition == "crop_only":
            pred = obs["full_prediction"] if obs["crop_agrees"] else None
            cost = dict(vlm_calls=min(obs["vlm_calls"], 3), runtime_seconds=None)
        elif condition == "none":
            pred = obs["full_prediction"]
            cost = dict(vlm_calls=1, runtime_seconds=None)
        else:
            cost = evaluate([image], d, condition, inference_wrapper, legacy)
            pred = cost["prediction"]
        results.append(
            dict(
                condition=condition,
                **score_record(pred, d),
                runtime_seconds=cost["runtime_seconds"],
                vlm_calls=cost["vlm_calls"],
            )
        )
    # Shared full/crop calls are reused. Do not fabricate isolated latency for them.
    return results, obs


def run(config, plan, root, phase, worker):
    import torch

    if worker not in range(5):
        raise ValueError("worker must be 0..4")
    if not torch.cuda.is_available() or "L40S" not in torch.cuda.get_device_name():
        raise RuntimeError("CSEV production requires L40S")
    directory = output(config) / phase
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / f"worker_{worker}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sig = signature(config, phase)
    assigned = tasks(config, plan, phase)[worker::5]
    pending = []
    for t in assigned:
        path = directory / "records" / (t["key"] + ".json")
        if path.exists():
            r = json.loads(path.read_text())
            if r["signature"] != sig or r["status"] != "complete":
                raise ValueError("record drift")
            if "images" in r:
                load_pages(r["images"], root)
        else:
            pending.append(t)
    wrapper = model(config["legacy"]) if pending else None
    verifier = (
        Verifier(wrapper, config["settings"], config["legacy"]["prompts"]["receipt"])
        if pending
        else None
    )
    tau = calibration(config)["threshold"] if phase != "development" else 0.0
    for t in pending:
        started = time.perf_counter()
        d = t["document"]
        image = load_pages(t["images"], root)[0]
        r = dict(
            key=t["key"],
            signature=sig,
            worker=worker,
            status="complete",
            record_id=d["record_id"],
            dataset_name=d["dataset_name"],
            family=t["family"],
            epsilon=t["epsilon"],
            clean_target=d["clean_target"],
            attack_target=d["frozen_attack_target"],
        )
        if phase == "development":
            r.update(verifier(image, d["dataset_name"]))
        elif phase in {"evaluation", "replay"}:
            r["results"], r["observation"] = comparisons(
                image, d, verifier, wrapper, config, tau
            )
            if phase == "evaluation" and t["family"] == "clean":
                if (r["observation"]["full_prediction"] == d["clean_target"]) != d[
                    "eligible"
                ]:
                    raise RuntimeError(
                        "clean eligibility drift from frozen Qwen baseline"
                    )
            r["images"] = t["images"]
        else:
            from csev.adaptive import search, gradient_step
            import lpips
            from attacks.qwen import _perceptibility
            from models.donut import pil_to_rgb_tensor

            contract = digest([sig, t["key"]])
            state = search(
                image,
                d["dataset_name"],
                d["frozen_attack_target"],
                t["epsilon"],
                t["restart"],
                config["settings"],
                tau,
                verifier,
                lambda im, obs: gradient_step(
                    wrapper,
                    im,
                    __import__("numpy").asarray(image),
                    d["frozen_attack_target"],
                    verifier.prompt,
                    obs,
                    tau,
                    t["epsilon"],
                    config["settings"],
                ),
                directory / "checkpoints" / t["key"] / "state.pkl",
                contract,
            )
            if not hasattr(verifier, "lpips"):
                verifier.lpips = lpips.LPIPS(net="alex").to(wrapper.device).eval()
            final = load_pages(state["images"], root)[0]
            lp, ssim = _perceptibility(
                pil_to_rgb_tensor(image, wrapper.device),
                pil_to_rgb_tensor(final, wrapper.device),
                verifier.lpips,
            )
            r.update(
                restart=t["restart"],
                images=state["images"],
                best_rank=state["best_rank"],
                observation=state["final_observation"],
                queries=state["queries"],
                steps=state["step"],
                trace=state["trace"],
                seen_rejection=state["seen_rejection"],
                linf_255=state["linf_255"],
                lpips=lp,
                ssim=ssim,
                attack_runtime_seconds=state["runtime_seconds"],
                **score_record(state["final_prediction"], d),
            )
        r["task_wall_seconds"] = time.perf_counter() - started
        atomic_json(directory / "records" / (t["key"] + ".json"), r)
        print(
            json.dumps(
                dict(
                    phase=phase,
                    worker=worker,
                    record_id=d["record_id"],
                    family=t["family"],
                    epsilon=t["epsilon"],
                    status="complete",
                )
            ),
            flush=True,
        )
    atomic_json(
        directory / f"worker_{worker}_complete.json",
        dict(signature=sig, keys=[t["key"] for t in assigned]),
    )
    if wrapper:
        wrapper.unload()
    lock.close()
    # Last worker performs strict merge before the array can satisfy afterok.
    with (directory / "merge.lock").open("a") as merge_lock:
        fcntl.flock(merge_lock, fcntl.LOCK_EX)
        if all((directory / f"worker_{w}_complete.json").exists() for w in range(5)):
            merge(config, plan, phase)


def merge(config, plan, phase):
    directory = output(config) / phase
    sig = signature(config, phase)
    expected = tasks(config, plan, phase)
    records = []
    for w in range(5):
        marker = json.loads((directory / f"worker_{w}_complete.json").read_text())
        if marker != dict(signature=sig, keys=[t["key"] for t in expected[w::5]]):
            raise ValueError("worker marker mismatch")
    observed = {p.stem for p in (directory / "records").glob("*.json")}
    if observed != {t["key"] for t in expected}:
        raise ValueError("missing or extra result records")
    for i, t in enumerate(expected):
        r = json.loads((directory / "records" / (t["key"] + ".json")).read_text())
        if (
            r["key"] != t["key"]
            or r["signature"] != sig
            or r["status"] != "complete"
            or r["worker"] != i % 5
        ):
            raise ValueError("result provenance mismatch")
        if phase in {"evaluation", "replay"}:
            if [v["condition"] for v in r["results"]] != config["settings"][
                "conditions"
            ]:
                raise ValueError("condition grid differs")
        if "images" in r:
            load_pages(r["images"], Path("/"))
        records.append(r)
    artifact = dict(signature=sig, records=records)
    dest = directory / "merged.json"
    if dest.exists() and json.loads(dest.read_text()) != artifact:
        raise ValueError("immutable merge differs")
    atomic_json(dest, artifact)
    if phase == "development":
        result = dict(
            **calibrate(records, config["settings"]),
            study_hash=digest(config),
            development_sha256=sha256_file(dest),
        )
        target = output(config) / "calibration.json"
        if target.exists() and json.loads(target.read_text()) != result:
            raise ValueError("calibration drift")
        atomic_json(target, result)
        print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))
    print(f"{phase}: strict merge complete ({len(records)} inputs)", flush=True)
    return artifact
