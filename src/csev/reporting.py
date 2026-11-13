"""Report exports with document-paired intervals and diagnostic coverage curves."""

import json
from pathlib import Path
import numpy as np
from checkpoints import atomic_json
from csev.defense import prediction
from csev.runner import merge, output, calibration, score_record


def metrics(frame):
    n = len(frame)
    accepted = int(frame.accepted.sum())
    correct = int(frame.correct.sum())
    return dict(
        n=n,
        accepted=accepted,
        coverage=accepted / n,
        correct_answer_rate=correct / n,
        selective_accuracy=correct / accepted if accepted else None,
        accepted_target_asr=(
            float(frame.targeted.mean()) if frame.family.iloc[0] != "clean" else None
        ),
    )


def analyze(config, plan):
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged = {
        p: merge(config, plan, p)["records"]
        for p in ("evaluation", "adaptive", "replay")
    }
    dest = output(config) / "analysis"
    dest.mkdir(exist_ok=True)
    rows = []
    for phase in ("evaluation", "replay"):
        for r in merged[phase]:
            rows.extend(
                dict(
                    record_id=r["record_id"],
                    family=r["family"],
                    epsilon=r["epsilon"],
                    condition=v["condition"],
                    prediction=v["prediction"],
                    accepted=v["accepted"],
                    correct=v["correct"],
                    targeted=v["targeted"],
                )
                for v in r["results"]
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(dest / "records.csv", index=False)
    summary = []
    paired = []
    rng = np.random.default_rng(config["settings"]["seed"])
    for (family, epsilon), group in frame.groupby(["family", "epsilon"]):
        for condition, part in group.groupby("condition"):
            summary.append(
                dict(
                    family=family, epsilon=epsilon, condition=condition, **metrics(part)
                )
            )
        candidate = group.loc[group.condition.eq("csev")].set_index("record_id")
        for condition in config["settings"]["conditions"]:
            if condition == "csev":
                continue
            baseline = group.loc[group.condition.eq(condition)].set_index("record_id")
            if (
                set(baseline.index) != set(candidate.index)
                or baseline.index.duplicated().any()
            ):
                raise ValueError("paired comparison has unmatched documents")
            baseline = baseline.loc[candidate.index]
            for metric in ("correct", "accepted", "targeted"):
                if metric == "targeted" and family == "clean":
                    continue
                delta = (
                    candidate[metric].astype(float).to_numpy()
                    - baseline[metric].astype(float).to_numpy()
                )
                boot = delta[
                    rng.integers(
                        len(delta),
                        size=(config["settings"]["bootstrap_repeats"], len(delta)),
                    )
                ].mean(axis=1)
                paired.append(
                    dict(
                        family=family,
                        epsilon=epsilon,
                        baseline=condition,
                        metric=metric,
                        difference=float(delta.mean()),
                        ci_low=float(np.quantile(boot, 0.025)),
                        ci_high=float(np.quantile(boot, 0.975)),
                        documents=len(delta),
                    )
                )
    pd.DataFrame(summary).to_csv(dest / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(dest / "paired_bootstrap.csv", index=False)
    # Descriptive threshold sweep only. The deployed point remains dev-calibrated.
    all_observations = [r for p in ("evaluation", "replay") for r in merged[p]]
    gaps = [
        r["observation"]["gap"]
        for r in all_observations
        if r["observation"]["gap"] is not None
    ]
    thresholds = sorted(set([calibration(config)["threshold"], 0.0] + gaps))
    curve = []
    for tau in thresholds:
        sweep = []
        for r in all_observations:
            d = dict(
                clean_target=r["clean_target"], frozen_attack_target=r["attack_target"]
            )
            sweep.append(
                dict(
                    family=r["family"],
                    epsilon=r["epsilon"],
                    **score_record(prediction(r["observation"], tau), d),
                )
            )
        for (family, epsilon), part in pd.DataFrame(sweep).groupby(
            ["family", "epsilon"]
        ):
            curve.append(
                dict(threshold=tau, family=family, epsilon=epsilon, **metrics(part))
            )
    curves = pd.DataFrame(curve)
    curves.to_csv(dest / "coverage_curves_descriptive.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (family, e), part in curves.groupby(["family", "epsilon"]):
        part = part.sort_values("coverage")
        if family == "clean":
            axes[0].plot(part.coverage, 1 - part.selective_accuracy, label="Clean")
        else:
            axes[1].plot(
                part.coverage, part.accepted_target_asr, label=f"{family}, eps={e}"
            )
    axes[0].set(xlabel="Coverage", ylabel="Error among accepted clean answers")
    axes[1].set(xlabel="Coverage", ylabel="Accepted-target ASR")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(dest / "coverage_tradeoff.png", dpi=200)
    plt.close(fig)
    clean = frame[(frame.family == "clean") & (frame.condition == "csev")].set_index(
        "record_id"
    )
    availability = []
    for (rid, e), rs in __import__("itertools").groupby(
        sorted(merged["adaptive"], key=lambda r: (r["record_id"], r["epsilon"])),
        key=lambda r: (r["record_id"], r["epsilon"]),
    ):
        rs = list(rs)
        availability.append(
            dict(
                record_id=rid,
                epsilon=e,
                clean_accepted=bool(clean.loc[rid, "accepted"]),
                encountered_rejection=any(r["seen_rejection"] for r in rs),
                selected_rejection=any(not r["accepted"] for r in rs),
                targeted_success=any(r["targeted"] for r in rs),
            )
        )
    pd.DataFrame(availability).to_csv(dest / "adaptive_trials.csv", index=False)
    traces = [
        dict(record_id=r["record_id"], epsilon=r["epsilon"], restart=r["restart"], **t)
        for r in merged["adaptive"]
        for t in r["trace"]
    ]
    pd.DataFrame(traces).to_csv(dest / "adaptive_traces.csv", index=False)
    pd.DataFrame(
        [
            {
                k: r[k]
                for k in (
                    "record_id",
                    "epsilon",
                    "restart",
                    "linf_255",
                    "lpips",
                    "ssim",
                    "queries",
                    "steps",
                    "attack_runtime_seconds",
                )
            }
            for r in merged["adaptive"]
        ]
    ).to_csv(dest / "adaptive_cost_quality.csv", index=False)
    from PIL import Image, ImageDraw
    from stages.final_protocol import load_pages
    from csev.defense import mask

    for r in [r for r in merged["evaluation"] if r["family"] == "clean"][:5]:
        im = load_pages(r["images"], Path("/"))[0]
        obs = r["observation"]
        annotated = im.copy()
        draw = ImageDraw.Draw(annotated)
        if obs["box"]:
            draw.rectangle(obs["box"], outline="red", width=4)
        if obs["control"]:
            draw.rectangle(obs["control"], outline="blue", width=4)
        panels = [annotated]
        for key in ("box", "control"):
            panels.append(
                mask(im, obs[key], config["settings"]["mask_value"])
                if obs[key]
                else im.copy()
            )
        canvas = Image.new("RGB", (900, 450), "white")
        for i, p in enumerate(panels):
            p.thumbnail((300, 420))
            canvas.paste(p, (300 * i, 30))
        ImageDraw.Draw(canvas).text(
            (5, 5),
            "Evidence (red), control (blue) | Evidence masked | Control masked",
            fill="black",
        )
        canvas.save(dest / (r["key"][:12] + "_regions.png"))
    atomic_json(
        dest / "analysis_complete.json",
        dict(
            study=config["run_id"],
            calibration=calibration(config),
            evaluation_records=len(frame),
            adaptive_attempts=len(merged["adaptive"]),
            adaptive_trials=len(merged["replay"]),
            availability_note="Rejections encountered during targeted search; not a separately optimized availability attack.",
            curve_note="Descriptive test curves; no external threshold selection. Compare coverage as well as ASR.",
            novelty_note="Empirical proposed method; neither calibration nor masking certifies robustness.",
        ),
    )
    print(pd.DataFrame(summary).to_string(index=False))
    return dest
