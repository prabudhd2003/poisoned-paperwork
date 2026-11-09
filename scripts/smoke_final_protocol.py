#!/usr/bin/env python3
"""Real L40S smoke, isolated from full experiment outputs; no job submission."""
import argparse
import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from checkpoints import atomic_json, sha256_file
from stages.final_protocol import preflight, read_plan, load_pages, model, evaluate


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    preflight(c, ROOT)
    if not torch.cuda.is_available() or "L40S" not in torch.cuda.get_device_name():
        raise RuntimeError("smoke requires L40S")
    wrapper = model(c)
    rows = []
    try:
        seen = set()
        for d in read_plan(c, ROOT):
            if d["dataset_name"] in seen:
                continue
            if c["mode"] == "adaptive" and not any(
                a["source_model"] == c["model_key"] for a in d["attacks"]
            ):
                continue
            pages = load_pages(d["images"], ROOT)
            for condition in c["conditions"]:
                rows.append(
                    dict(
                        record_id=d["record_id"],
                        condition=condition,
                        result=evaluate(pages, d, condition, wrapper, c),
                    )
                )
            if c["mode"] == "adaptive":
                from final_protocol.whitebox import optimize
                from final_protocol.adaptive import search
                from final_protocol.evidence import TesseractOCR

                attack = next(
                    (a for a in d["attacks"] if a["source_model"] == c["model_key"]),
                    None,
                )
                if attack is None:
                    raise RuntimeError("smoke document has no eligible attack seed")
                task = (
                    "resume"
                    if d["dataset_name"] == "resume_parsing_vision"
                    else "receipt"
                )
                smoke_root = ROOT / "outputs/smoke" / c["run_id"] / d["dataset_name"]
                contract = {
                    "config_sha256": sha256_file(args.config),
                    "smoke_version": 1,
                }
                seed, state = optimize(
                    wrapper,
                    pages,
                    c["prompts"][task],
                    attack["attack_target"],
                    d["dataset_name"],
                    TesseractOCR(c["ocr"]),
                    c["evidence"],
                    attack["epsilon_255"],
                    dict(c["whitebox"], steps=2),
                    smoke_root / "whitebox.pkl",
                    contract,
                )
                result, state = search(
                    pages,
                    [seed],
                    lambda p: evaluate(p, d, "evidence", wrapper, c),
                    attack["attack_target"],
                    attack["epsilon_255"],
                    dict(c["adaptive"], queries=3, seed=566),
                    smoke_root / "query.pkl",
                    contract,
                )
                rows.append(
                    dict(
                        record_id=d["record_id"],
                        gradient_steps=2,
                        queries=state["queries"],
                    )
                )
            # Verify multi-page assembly independently when a one-page example was first.
            if c["model_key"] == "internvl":
                wrapper.predict(pages[:1] * 2, c["prompts"]["receipt"])
            seen.add(d["dataset_name"])
        atomic_json(
            ROOT / "outputs/smoke" / f"{c['run_id']}.json",
            dict(
                config_sha256=sha256_file(args.config),
                git_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                mode=c["mode"],
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                rows=rows,
            ),
        )
    finally:
        wrapper.unload()
    print(
        "Smoke passed for the configured mode. Full runtime and robustness remain unverified."
    )


if __name__ == "__main__":
    main()
