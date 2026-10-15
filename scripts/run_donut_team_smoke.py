#!/usr/bin/env python3
"""Run the 20-document validation smoke after a successful restart audit."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from donut_readiness import ROOT, preflight, validate_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--restart-report", type=Path, required=True)
    parser.add_argument("--user-id", default="user4")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    from team import get_team_member
    get_team_member(args.user_id)
    config_path = args.config.resolve()
    config, baseline = preflight(config_path)
    if config.get("smoke_per_dataset") != 10 or len(baseline) != 20:
        raise ValueError("This runner requires exactly ten receipts per dataset")
    if args.preflight:
        return
    audit = json.loads(args.restart_report.read_text())
    if audit.get("passed") is not True or len(audit["results"]) != 2:
        raise RuntimeError("Restart audit has not passed")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Baseline production repeats only a subset: verify every smoke receipt now.
    import gc
    import tempfile
    from PIL import Image
    from data import load_processed_rows
    from models.donut import DonutDocVQA, pil_to_rgb_tensor, rgb_tensor_to_pil
    from attacks.donut import project_linf
    from normalization import normalize_receipt_total
    from stages.donut_attack import _processor_parity
    device = torch.device("cuda")
    model = DonutDocVQA(device=device, model_id=config["model_id"],
                       revision=config["model_revision"], max_length=config["max_length"])
    rows = load_processed_rows(ROOT, "validation", set(baseline.index))
    checks = []
    with tempfile.TemporaryDirectory(prefix="donut-clean-") as temporary:
        for record_id in sorted(baseline.index):
            row = rows[record_id]
            image = row["image"].convert("RGB")
            clean = pil_to_rgb_tensor(image, device=device)
            zero = project_linf(clean, clean, 0, quantize=True)
            if not torch.equal(clean, zero):
                raise RuntimeError("Zero epsilon changed clean pixels")
            png = Path(temporary) / "zero.png"
            rgb_tensor_to_pil(zero).save(png)
            first = model.predict_pil(image, config["question"])
            second = model.predict_pil(image, config["question"])
            with Image.open(png) as reloaded:
                third = model.predict_pil(reloaded.convert("RGB"), config["question"])
            parity = _processor_parity(model, image, device)
            if first != second or first != third:
                raise RuntimeError(f"Clean repeat/PNG mismatch: {record_id}")
            if normalize_receipt_total(first, row["dataset_name"]) != row["clean_target"]:
                raise RuntimeError(f"Frozen clean-correct result no longer reproduces: {record_id}")
            if parity["mean_absolute_error"] > config["processor_parity_mean_tolerance"]:
                raise RuntimeError(f"Processor parity failed: {record_id}")
            checks.append(dict(record_id=record_id, clean_prediction=first,
                               repeat_and_zero_png_match=True, processor_parity=parity))
            print(f"Clean validation passed: {record_id}", flush=True)
    output = ROOT / "outputs/donut_attack" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "clean_validation.json").write_text(json.dumps(checks, indent=2))
    del model, clean, zero
    gc.collect()
    torch.cuda.empty_cache()
    from stages.donut_attack import run
    run(config, config_path, user_id=args.user_id, worker_id=0, num_workers=1)
    validate_results(config_path, expected_workers=1)


if __name__ == "__main__":
    main()
