#!/usr/bin/env python3
"""Compare fresh-process Donut reference and resumed runs; preserve old audits."""
import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from attacks.donut import checkpoint_state, load_checkpoint_state, targeted_image_attack
from attacks.donut_validation import validate_initial_gradient
from data import load_processed_rows
from models.donut import DonutDocVQA, pil_to_rgb_tensor, rgb_tensor_to_pil
from normalization import normalize_receipt_total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="donut_paper_smoke_khalid_v1")
    parser.add_argument("--audit-id", default="restart_audit_paper_v2")
    parser.add_argument("--phase", choices=["reference", "partial", "resume"])
    args = parser.parse_args()
    for value in (args.source_run, args.audit_id):
        if Path(value).name != value or value in (".", ".."):
            raise ValueError("Run IDs must be single directory names")
    source = ROOT / "outputs/donut_attack" / args.source_run
    out = ROOT / "outputs/donut_attack" / args.audit_id
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    records = [json.loads(p.read_text()) for p in sorted(source.glob("shards/*/records/*.json"))]
    if len(records) != 2 or {r["dataset_name"] for r in records} != {"sroie", "cord_v2"}:
        raise ValueError("Restart audit requires one source record per receipt dataset")
    attack = config["attack"]
    if len(attack["epsilons_255"]) != 1:
        raise ValueError("Restart audit requires a single epsilon")
    total_steps = int(attack["max_steps"])
    resume_step = total_steps // 2
    if resume_step < 1:
        raise ValueError("At least two steps are required")
    epsilon = float(attack["epsilons_255"][0])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.phase is None:
        # A new audit ID prevents accidental reuse or replacement of prior evidence.
        out.mkdir(parents=True, exist_ok=False)
        for phase in ("reference", "partial", "resume"):
            subprocess.run([sys.executable, __file__, "--source-run", args.source_run,
                            "--audit-id", args.audit_id, "--phase", phase], check=True)
        results = []
        for record in records:
            key = record["dataset_name"]
            a, b = (np.load(out / f"{key}_{phase}.npy") for phase in ("reference", "resume"))
            ra, rb = (json.loads((out / f"{key}_{phase}.json").read_text())
                      for phase in ("reference", "resume"))
            results.append(dict(dataset=key, identical_final_tensor=bool(np.array_equal(a, b)),
                                max_absolute_difference_255=float(np.max(np.abs(a-b))*255),
                                same_decoded_prediction=ra["prediction"] == rb["prediction"],
                                reference=ra, resumed=rb))
        passed = all(r["identical_final_tensor"] and r["same_decoded_prediction"] for r in results)
        report = dict(passed=passed, source_commit=commit, source_config_sha256=config_hash,
                      audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      resume_step=resume_step, total_steps=total_steps, results=results)
        (out / "restart_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        if not passed:
            raise SystemExit("RESTART MISMATCH: do not expand the run")
        print("FRESH-PROCESS RESTART CHECK PASSED", flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Run the audit in a CARC GPU allocation")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = load_processed_rows(ROOT, "validation", {r["record_id"] for r in records})
    device = torch.device("cuda")
    model = DonutDocVQA(device=device, model_id=config["model_id"],
                       revision=config["model_revision"], max_length=config["max_length"])
    for record in records:
        key = record["dataset_name"]
        row = rows[record["record_id"]]
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        clean = pil_to_rgb_tensor(row["image"].convert("RGB"), device=device)
        clean_hash = hashlib.sha256(clean.mul(255).round().to(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        if clean_hash != record["clean_image_sha256"]:
            raise RuntimeError("Clean input changed since the source run")
        state = None
        if args.phase == "resume":
            state = load_checkpoint_state(out / f"{key}_step{resume_step}.pt",
                clean_sha256=clean_hash, config_sha256=config_hash,
                model_revision=config["model_revision"], device=device)
            if state["completed_steps"] != resume_step:
                raise RuntimeError("Incorrect checkpoint step")
            torch.set_rng_state(state["torch_rng_state"])
            np.random.set_state(state["numpy_rng_state"])
            random.setstate(state["python_rng_state"])
            torch.cuda.set_rng_state_all(state["extra_state"]["cuda_rng_states"])

        gradient_checks = []
        initial_losses = []
        def objective(image):
            loss, metadata = model.attack_objective(image, question=config["question"],
                target_text=row["attack_target"], loss_type=attack["loss_type"])
            if not initial_losses:
                initial_losses.append(float(loss.detach()))
                def check(grad):
                    if not gradient_checks:
                        gradient_checks.append(dict(finite=bool(torch.isfinite(grad).all()),
                            nonzero=bool(torch.count_nonzero(grad)), max_abs=float(grad.abs().max())))
                image.register_hook(check)
            return loss, metadata

        def save(step, image, best, loss, metadata, optimizer):
            checkpoint_state(out / f"{key}_step{resume_step}.pt", adversarial=image,
                clean_sha256=clean_hash, completed_steps=step, best_loss=best,
                optimizer=optimizer, config_sha256=config_hash,
                model_revision=config["model_revision"],
                extra_state={"cuda_rng_states": torch.cuda.get_rng_state_all()})

        result = targeted_image_attack(clean, objective, epsilon_255=epsilon,
            step_size_255=float(attack["step_size_255"][str(int(epsilon))]),
            max_steps=resume_step if args.phase == "partial" else total_steps,
            optimizer_name=attack.get("optimizer", "adam"),
            quantize_each_step=attack.get("quantize_each_step", True),
            start_adversarial=state["adversarial"] if state else None,
            start_step=resume_step if state else 0,
            start_best_loss=state["best_loss"] if state else float("inf"),
            optimizer_state=state["optimizer_state"] if state else None,
            checkpoint_interval=resume_step, checkpoint_callback=save if args.phase == "partial" else None)
        validate_initial_gradient(gradient_checks[0] if gradient_checks else None,
            resumed=state is not None, loss_type=attack["loss_type"], initial_loss=initial_losses[0])
        if any(p.grad is not None for p in model.model.parameters()):
            raise RuntimeError("Model parameter received a gradient")
        np.save(out / f"{key}_{args.phase}.npy", result.adversarial.cpu().numpy())
        image_path = out / f"{key}_{args.phase}.png"
        rgb_tensor_to_pil(result.adversarial).save(image_path)
        with Image.open(image_path) as image:
            reloaded = image.convert("RGB")
            prediction = model.predict_pil(reloaded, config["question"])
            linf = float((pil_to_rgb_tensor(reloaded, device=device)-clean).abs().max()*255)
        if linf > epsilon + 1e-4:
            raise RuntimeError("Reloaded image violates epsilon")
        details = dict(phase=args.phase, completed_steps=result.completed_steps,
            prediction=prediction, target=row["attack_target"], linf_255=linf,
            success=normalize_receipt_total(prediction, key) == row["attack_target"],
            first_gradient=gradient_checks[0], initial_loss=initial_losses[0],
            model_parameter_gradients_absent=True, gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__, deterministic_algorithms=True)
        (out / f"{key}_{args.phase}.json").write_text(json.dumps(details, indent=2))
        print(key, details, flush=True)
        del result, clean, state
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
