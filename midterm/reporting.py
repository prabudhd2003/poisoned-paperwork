"""CPU-only, publication-quality reporting helpers for Poisoned Paperwork.

The notebooks in this directory deliberately contain very little logic.  This
module is the single source of truth for record discovery, metric definitions,
confidence intervals, and figure/table export.  It never loads a VLM or needs a
GPU, so the notebooks are safe to run while CARC jobs are active.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np


MIDTERM_DIR = Path(__file__).resolve().parent
REPO_ROOT = MIDTERM_DIR.parent
GENERATED = MIDTERM_DIR / "generated"
FIGURES = GENERATED / "figures"
TABLES = GENERATED / "tables"

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#222222",
}


def setup_style() -> None:
    """Use a readable, color-blind-safe style suitable for reports."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_output_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def save_figure(figure, stem: str) -> Path:
    """Save a report-ready 300 DPI PNG."""
    ensure_output_dirs()
    png = FIGURES / f"{stem}.png"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {png.relative_to(REPO_ROOT)}")
    return png


def export_table(frame, stem: str) -> Path:
    """Export the notebook's displayed report table as CSV."""
    ensure_output_dirs()
    csv = TABLES / f"{stem}.csv"
    frame.to_csv(csv, index=False)
    print(f"Saved {csv.relative_to(REPO_ROOT)}")
    return csv


def _candidate_paths(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return [value]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("path") or item.get("image_path") or item.get("saved_path")
        if item:
            result.append(str(item))
    return result


def resolve_run_dir(candidates: Iterable[str | Path], env_var: str | None = None) -> Path | None:
    """Return the first existing run directory, allowing a CARC override."""
    expanded = []
    if env_var and os.environ.get(env_var):
        expanded.append(Path(os.environ[env_var]).expanduser())
    expanded.extend(Path(path).expanduser() for path in candidates)
    for path in expanded:
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.is_dir():
            return path.resolve()
    return None


def _read_json_records(paths: Iterable[Path]):
    import pandas as pd

    rows = []
    for path in sorted(paths):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping unreadable record {path}: {error}")
            continue
        values = value if isinstance(value, list) else [value]
        for row in values:
            if isinstance(row, dict):
                row = dict(row)
                row["_record_path"] = str(path)
                rows.append(row)
    return pd.DataFrame(rows)


def load_attack_records(run_dir: str | Path):
    """Load merged results when available, otherwise read live worker shards."""
    import pandas as pd

    run_dir = Path(run_dir)
    merged_names = (
        "merged/attacks.parquet",
        "merged/records.parquet",
        "merged/results.parquet",
        "merged/attacks.csv",
        "merged/records.csv",
        "merged/results.csv",
    )
    for relative in merged_names:
        path = run_dir / relative
        if path.is_file():
            frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            frame["_record_path"] = str(path)
            return normalize_attack_records(frame, run_dir)

    record_paths = list(run_dir.glob("shards/worker_*/records/*.json"))
    if not record_paths:
        record_paths = list(run_dir.glob("**/records/*.json"))
    if not record_paths:
        raise FileNotFoundError(f"No merged table or shard records found under {run_dir}")
    return normalize_attack_records(_read_json_records(record_paths), run_dir)


def load_baseline_records(run_dir: str | Path):
    """Load the authoritative merged clean-baseline predictions."""
    import pandas as pd

    run_dir = Path(run_dir)
    for relative in ("merged/predictions.parquet", "merged/predictions.csv"):
        path = run_dir / relative
        if path.is_file():
            return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    raise FileNotFoundError(f"No merged baseline predictions found under {run_dir}")


def summarize_baseline(frame):
    """Clean accuracy and parse failures by model and dataset."""
    grouped = frame.groupby(["model_key", "dataset_name"], dropna=False)
    rows = []
    for (model, dataset), group in grouped:
        correct = int(group["clean_correct"].fillna(False).astype(bool).sum())
        parse_failures = int(group.get("parse_status", "").astype(str).eq("failed").sum()) if "parse_status" in group else 0
        rows.append(
            {
                "model": str(model).title(),
                "dataset": {"sroie": "SROIE", "cord_v2": "CORD v2", "resume_parsing_vision": "Resume"}.get(str(dataset), str(dataset)),
                "correct": correct,
                "total": len(group),
                "accuracy": correct / len(group),
                "parse_failures": parse_failures,
            }
        )
    import pandas as pd

    return pd.DataFrame(rows).sort_values(["model", "dataset"]).reset_index(drop=True)


def _first_column(frame, names: Iterable[str], default=np.nan):
    for name in names:
        if name in frame.columns:
            return frame[name]
    return default


def normalize_attack_records(frame, run_dir: Path):
    """Map historical and current result schemas to stable report columns."""
    import pandas as pd

    result = frame.copy()
    result["epsilon"] = pd.to_numeric(
        _first_column(result, ("epsilon_255", "epsilon", "eps", "attack_epsilon")),
        errors="coerce",
    )
    success = _first_column(
        result,
        (
            "reloaded_success",
            "targeted_success_reloaded",
            "saved_image_success",
            "attack_success",
            "success",
        ),
        False,
    )
    if hasattr(success, "astype"):
        if success.dtype == object:
            success = success.astype(str).str.lower().isin(("true", "1", "yes"))
        else:
            success = success.fillna(False).astype(bool)
    result["success"] = success
    result["lpips"] = pd.to_numeric(
        _first_column(result, ("lpips", "lpips_distance", "perceptual_distance")),
        errors="coerce",
    )
    result["ssim"] = pd.to_numeric(
        _first_column(result, ("ssim", "structural_similarity")), errors="coerce"
    )
    result["runtime_seconds"] = pd.to_numeric(
        _first_column(result, ("runtime_seconds", "attack_runtime_seconds", "elapsed_seconds")),
        errors="coerce",
    )
    result["steps"] = pd.to_numeric(
        _first_column(result, ("completed_steps", "selected_step", "steps", "attack_steps")),
        errors="coerce",
    )
    dataset = _first_column(result, ("dataset_name", "dataset", "source_dataset"), "unknown")
    if not hasattr(dataset, "astype"):
        dataset = pd.Series([dataset] * len(result), index=result.index)
    result["dataset"] = dataset.astype(str)
    record_id = _first_column(
        result, ("record_id", "sample_id", "document_id", "key", "id"), "unknown"
    )
    if not hasattr(record_id, "astype"):
        record_id = pd.Series([record_id] * len(result), index=result.index)
    result["record_id"] = record_id.astype(str)
    result["_run_dir"] = str(run_dir)
    return result


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials**2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize_attacks(frame):
    """One row per epsilon with ASR, uncertainty, quality, and cost."""
    import pandas as pd

    rows = []
    for epsilon, group in frame.dropna(subset=["epsilon"]).groupby("epsilon"):
        trials = len(group)
        successes = int(group["success"].sum())
        low, high = wilson_interval(successes, trials)
        rows.append(
            {
                "epsilon": int(epsilon) if float(epsilon).is_integer() else float(epsilon),
                "trials": trials,
                "successes": successes,
                "asr": successes / trials if trials else np.nan,
                "asr_ci_low": low,
                "asr_ci_high": high,
                "median_lpips": group["lpips"].median(),
                "median_ssim": group["ssim"].median(),
                "median_runtime_seconds": group["runtime_seconds"].median(),
                "median_steps": group["steps"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)


def plot_attack_summary(summary, *, title: str, stem: str):
    """Create the four main quantitative panels for a model/dataset pair."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    setup_style()
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    eps = summary["epsilon"].to_numpy()
    asr = summary["asr"].to_numpy()
    error = np.vstack((asr - summary["asr_ci_low"], summary["asr_ci_high"] - asr))

    axes[0, 0].errorbar(eps, asr, yerr=error, marker="o", lw=2.2, capsize=5, color=COLORS["blue"])
    for x, y, successes, trials in zip(eps, asr, summary["successes"], summary["trials"]):
        axes[0, 0].annotate(f"{successes}/{trials}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center")
    axes[0, 0].set(title="Targeted attack success", xlabel="Perturbation budget ε (pixel levels)", ylabel="Attack success rate")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].set_ylim(-0.04, 1.08)

    axes[0, 1].plot(eps, summary["median_ssim"], marker="o", lw=2.2, color=COLORS["green"], label="SSIM (higher is closer)")
    axes[0, 1].plot(eps, summary["median_lpips"], marker="s", lw=2.2, color=COLORS["orange"], label="LPIPS (lower is closer)")
    axes[0, 1].set(title="Perceptual similarity", xlabel="Perturbation budget ε", ylabel="Median score")
    axes[0, 1].legend()

    axes[1, 0].scatter(summary["median_lpips"], asr, s=95, c=eps, cmap="viridis", edgecolor="white", linewidth=0.8)
    for _, row in summary.iterrows():
        axes[1, 0].annotate(f"ε={row.epsilon:g}", (row.median_lpips, row.asr), xytext=(6, 5), textcoords="offset points")
    axes[1, 0].set(title="Effectiveness–visibility trade-off", xlabel="Median LPIPS", ylabel="Attack success rate")
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))

    runtime = summary["median_runtime_seconds"] / 60
    axes[1, 1].bar(eps.astype(str), runtime, color=COLORS["purple"], alpha=0.9)
    axes[1, 1].set(title="Compute cost", xlabel="Perturbation budget ε", ylabel="Median runtime (minutes)")
    for index, (minutes, steps) in enumerate(zip(runtime, summary["median_steps"])):
        label = f"{minutes:.1f} min"
        if not np.isnan(steps):
            label += f"\n{steps:.0f} steps"
        axes[1, 1].text(index, minutes, label, ha="center", va="bottom", fontsize=9)

    figure.suptitle(title, fontsize=16, y=1.01)
    figure.tight_layout()
    save_figure(figure, stem)
    return figure


def _resolve_path(raw: str, run_dir: Path) -> Path | None:
    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((run_dir / path, REPO_ROOT / path))
        # Result metadata is sometimes relative to the worktree rather than
        # the run directory. This covers the main, Sakeeth, and Khalid layouts.
        candidates.extend(parent / path for parent in list(run_dir.parents)[:6])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def record_image_paths(row, run_dir: Path) -> list[Path]:
    for column in ("images", "saved_images", "adversarial_images", "attacked_images"):
        if column in row and _candidate_paths(row[column]):
            return [path for raw in _candidate_paths(row[column]) if (path := _resolve_path(raw, run_dir))]
    for column in ("image_path", "saved_image_path", "attacked_image_path", "adversarial_path"):
        if column in row and isinstance(row[column], str):
            path = _resolve_path(row[column], run_dir)
            if path:
                return [path]
    record_path = Path(str(row.get("_record_path", "")))
    if record_path.is_file():
        worker_dir = record_path.parents[1]
        stem = record_path.stem
        conventional = worker_dir / "images" / stem / "page_000.png"
        if conventional.is_file():
            return [conventional]
        matches = sorted((worker_dir / "images").glob(f"**/{stem}*.png"))
        if matches:
            return matches
    return []


def _public_clean_images(row):
    try:
        from src.data import load_processed_rows, row_images

        record_id = str(row.get("record_id"))
        parts = record_id.split("/")
        split = parts[1] if len(parts) > 2 else "validation"
        rows = load_processed_rows(REPO_ROOT, split=split, record_ids={record_id})
        return [image.convert("RGB") for image in row_images(rows[record_id])]
    except Exception:
        pass
    return []


def clean_images(row):
    from PIL import Image

    for column in ("clean_images", "original_images", "source_images", "clean_image_paths"):
        if column in row and _candidate_paths(row[column]):
            found = []
            for raw in _candidate_paths(row[column]):
                path = _resolve_path(raw, REPO_ROOT)
                if path:
                    with Image.open(path) as image:
                        found.append(image.convert("RGB").copy())
            if found:
                return found
    for column in ("clean_image_path", "original_image_path", "source_path", "processed_path"):
        if column in row and isinstance(row[column], str):
            path = _resolve_path(row[column], REPO_ROOT)
            if path:
                with Image.open(path) as image:
                    return [image.convert("RGB").copy()]
    return _public_clean_images(row)


def show_qualitative_example(frame, *, preferred_epsilon: float | None = None, success: bool = True, page: int = 0, stem: str | None = None):
    """Show native-resolution original, attack, and an interpretable heatmap."""
    import matplotlib.pyplot as plt
    from PIL import Image

    candidates = frame[frame["success"] == success]
    if preferred_epsilon is not None:
        at_epsilon = candidates[np.isclose(candidates["epsilon"], preferred_epsilon)]
        if not at_epsilon.empty:
            candidates = at_epsilon
    if candidates.empty:
        print(f"No {'successful' if success else 'unsuccessful'} record is available yet.")
        return None

    selected = None
    for _, row in candidates.iterrows():
        run_dir = Path(row["_run_dir"])
        clean = clean_images(row)
        attacked = record_image_paths(row, run_dir)
        if clean and attacked:
            selected = (row, clean, attacked)
            break
    if selected is None:
        print("Metrics loaded, but the referenced images are not present on this machine. Run this cell on CARC or copy the images locally.")
        return None

    row, clean_paths, attacked_paths = selected
    page = min(page, len(clean_paths) - 1, len(attacked_paths) - 1)
    clean = np.asarray(clean_paths[page], dtype=np.int16)
    with Image.open(attacked_paths[page]) as image:
        attacked = np.asarray(image.convert("RGB"), dtype=np.int16)
    if clean.shape != attacked.shape:
        print(f"Cannot compare different shapes: {clean.shape} and {attacked.shape}")
        return None

    magnitude = np.abs(attacked - clean).max(axis=2)
    vmax = max(1, float(np.percentile(magnitude, 99.5)))
    setup_style()
    figure, axes = plt.subplots(1, 3, figsize=(18, 8))
    axes[0].imshow(clean.astype(np.uint8), interpolation="nearest")
    axes[0].set_title("Original")
    axes[1].imshow(attacked.astype(np.uint8), interpolation="nearest")
    axes[1].set_title(f"Attacked (ε={row.epsilon:g}, success={bool(row.success)})")
    heat = axes[2].imshow(magnitude, cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
    axes[2].set_title(f"Pixel-change heatmap\nmax={magnitude.max():.0f}/255, 99.5th percentile={vmax:.0f}/255")
    figure.colorbar(heat, ax=axes[2], fraction=0.046, pad=0.04, label="Max RGB-channel change")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(str(row.record_id), fontsize=14)
    figure.tight_layout()
    if stem:
        save_figure(figure, stem)
    return figure


def run_attack_report(
    *,
    title: str,
    stem: str,
    run_candidates: Iterable[str | Path],
    env_var: str,
    dataset_filter: str | None = None,
    qualitative_epsilon: float | None = None,
    extra_run_candidates: Iterable[Iterable[str | Path]] = (),
):
    """Notebook entry point: discover, audit, tabulate, plot, and export."""
    from IPython.display import display

    run_dir = resolve_run_dir(run_candidates, env_var)
    if run_dir is None:
        print(f"Results are not available yet. After the run finishes, rerun this notebook. Optional override: export {env_var}=/absolute/run/path")
        return None
    print(f"Reading {run_dir}")
    frames = [load_attack_records(run_dir)]
    for candidate_group in extra_run_candidates:
        extra = resolve_run_dir(candidate_group)
        if extra is not None and extra != run_dir:
            print(f"Also reading {extra}")
            frames.append(load_attack_records(extra))
    import pandas as pd

    records = pd.concat(frames, ignore_index=True, sort=False)
    if dataset_filter:
        mask = records["dataset"].astype(str).str.contains(dataset_filter, case=False, regex=False)
        if mask.any():
            records = records[mask].copy()
    summary = summarize_attacks(records)
    if summary.empty:
        print("No completed attack records are available yet.")
        return None
    display(summary.style.format({"asr": "{:.1%}", "asr_ci_low": "{:.1%}", "asr_ci_high": "{:.1%}", "median_lpips": "{:.3f}", "median_ssim": "{:.3f}", "median_runtime_seconds": "{:.1f}"}))
    export_table(summary, f"{stem}_metrics")
    plot_attack_summary(summary, title=title, stem=f"{stem}_quantitative")
    show_qualitative_example(records, preferred_epsilon=qualitative_epsilon, success=True, stem=f"{stem}_successful_example")
    return records, summary


def run_defense_report(
    run_candidates: Iterable[str | Path] = (
        "/project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork/outputs/defenses/defenses_validation_v1",
        "outputs/defenses/defenses_validation_v1",
    ),
    env_var: str = "PP_DEFENSE_RUN_DIR",
):
    """Analyze future non-adaptive and adaptive defense results."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display
    from matplotlib.ticker import PercentFormatter

    run_dir = resolve_run_dir(run_candidates, env_var)
    if run_dir is None:
        print(f"Defense results are not available yet. Rerun after completion, or export {env_var}=/absolute/run/path")
        return None
    path = run_dir / "merged/records.parquet"
    if not path.is_file():
        csv = run_dir / "merged/records.csv"
        if not csv.is_file():
            print(f"No merged defense records found under {run_dir}")
            return None
        records = pd.read_csv(csv)
    else:
        records = pd.read_parquet(path)
    summary = (
        records.groupby(["example_type", "defense_id"], dropna=False)
        .agg(
            trials=("experiment_key", "size"),
            clean_accuracy=("clean_correct", "mean"),
            target_asr=("adversarial_target_success", "mean"),
            abstention=("abstained", "mean"),
            agreement=("agreement", "mean"),
            extra_passes=("additional_vlm_passes", "mean"),
            median_runtime_seconds=("runtime_seconds", "median"),
        )
        .reset_index()
    )
    display(summary.style.format({"clean_accuracy": "{:.1%}", "target_asr": "{:.1%}", "abstention": "{:.1%}", "agreement": "{:.1%}", "extra_passes": "{:.2f}", "median_runtime_seconds": "{:.1f}"}))
    export_table(summary, "defense_results")

    nonadaptive = summary[summary["example_type"].eq("non_adaptive_attack")].copy()
    if nonadaptive.empty:
        nonadaptive = summary.copy()
    setup_style()
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = (
        ("target_asr", "Post-defense targeted ASR", "Lower is better", COLORS["red"], True),
        ("clean_accuracy", "Clean accuracy retained", "Higher is better", COLORS["blue"], True),
        ("abstention", "Abstention rate", "Flagged / rejected examples", COLORS["orange"], True),
        ("extra_passes", "Compute overhead", "Mean additional VLM passes", COLORS["purple"], False),
    )
    for axis, (column, title, ylabel, color, percentage) in zip(axes.flat, panels):
        bars = axis.bar(nonadaptive["defense_id"].astype(str), nonadaptive[column], color=color)
        axis.set(title=title, ylabel=ylabel)
        axis.tick_params(axis="x", rotation=25)
        if percentage:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.set_ylim(0, max(1.0, float(nonadaptive[column].max()) * 1.15))
        for bar, value in zip(bars, nonadaptive[column]):
            label = f"{value:.1%}" if percentage else f"{value:.2f}"
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=9)
    figure.suptitle("Defense effectiveness, utility, and cost", fontsize=16)
    figure.tight_layout()
    save_figure(figure, "defense_overview")

    adaptive = records[records["example_type"].eq("adaptive_attack")].copy()
    adaptive_summary = None
    if not adaptive.empty:
        adaptive_summary = (
            adaptive.groupby(["defense_id", "epsilon_255"], dropna=False)
            .agg(trials=("record_id", "size"), adaptive_asr=("adversarial_target_success", "mean"), abstention=("abstained", "mean"), agreement=("agreement", "mean"))
            .reset_index()
        )
        display(adaptive_summary.style.format({"adaptive_asr": "{:.1%}", "abstention": "{:.1%}", "agreement": "{:.1%}"}))
        export_table(adaptive_summary, "adaptive_defense_results")
        adaptive_figure, axis = plt.subplots(figsize=(8, 5))
        for defense, group in adaptive_summary.groupby("defense_id"):
            axis.plot(group["epsilon_255"], group["adaptive_asr"], marker="o", lw=2, label=defense)
        axis.set(title="Adaptive attack success against defenses", xlabel="Perturbation budget ε", ylabel="Adaptive targeted ASR", ylim=(-0.04, 1.04))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0)); axis.legend()
        adaptive_figure.tight_layout()
        save_figure(adaptive_figure, "adaptive_defense_results")
    return records, summary, adaptive_summary


def render_authoritative_overview():
    """Rebuild the cross-experiment overview from authoritative CARC outputs."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.ticker import PercentFormatter

    main = Path("/project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork")
    donut = Path("/project2/yzhao010_1531/csci_699_new_arch/kbali-donut-worktree")
    baseline_dir = resolve_run_dir(
        [main / "outputs/baseline/baseline_validation_v1", "outputs/baseline/baseline_validation_v1"],
        "PP_BASELINE_RUN_DIR",
    )
    if baseline_dir is None:
        raise FileNotFoundError("Run this notebook on CARC, or set PP_BASELINE_RUN_DIR to the merged baseline run.")
    baseline = summarize_baseline(load_baseline_records(baseline_dir))

    specs = [
        (
            "Qwen",
            [Path("/project2/yzhao010_1531/qwen-receipt-runs/qwen_attack/qwen_receipt_validation_v1"), "outputs/qwen_attack/qwen_receipt_validation_v1"],
        ),
        ("Qwen", [main / "outputs/qwen_attack/qwen_resume_validation_v1", "outputs/qwen_attack/qwen_resume_validation_v1"]),
        ("Donut", [donut / "outputs/donut_attack/donut_paper_full_validation_v1"]),
        ("Donut", [donut / "outputs/donut_attack/donut_paper_eps8_16_20_v1"]),
    ]
    attack_frames = []
    for model, candidates in specs:
        run_dir = resolve_run_dir(candidates)
        if run_dir is None:
            print(f"Skipping unavailable {model} run: {candidates[0]}")
            continue
        frame = load_attack_records(run_dir)
        frame["model"] = model
        attack_frames.append(frame)
    if not attack_frames:
        raise FileNotFoundError("No authoritative attack result directories were found.")
    attacks = pd.concat(attack_frames, ignore_index=True, sort=False)
    summaries = []
    for (model, dataset), group in attacks.groupby(["model", "dataset"]):
        summary = summarize_attacks(group)
        summary.insert(0, "dataset", {"sroie": "SROIE", "cord_v2": "CORD v2", "resume_parsing_vision": "Resume"}.get(str(dataset), str(dataset)))
        summary.insert(0, "model", model)
        summaries.append(summary)
    attack_summary = pd.concat(summaries, ignore_index=True)
    export_table(baseline, "current_baseline_results")
    export_table(attack_summary, "current_attack_results")
    setup_style()

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    labels = baseline["model"] + "\n" + baseline["dataset"]
    bar_colors = baseline["model"].map({"Qwen": COLORS["blue"], "Donut": COLORS["orange"]}).fillna(COLORS["black"])
    bars = axes[0].bar(labels, baseline["accuracy"], color=bar_colors)
    axes[0].set(title="Clean baseline accuracy", ylabel="Accuracy", ylim=(0, 1.08))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].tick_params(axis="x", rotation=20)
    for bar, row in zip(bars, baseline.itertuples()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row.correct}/{row.total}\n{row.accuracy:.1%}", ha="center", va="bottom", fontsize=9)

    palette = (COLORS["blue"], COLORS["green"], COLORS["purple"], COLORS["orange"], COLORS["red"])
    for ((model, dataset), group), color in zip(attack_summary.groupby(["model", "dataset"]), palette):
        axes[1].plot(group["epsilon"], group["asr"], marker="o", lw=2.2, label=f"{model} · {dataset}", color=color)
    axes[1].set(title="Targeted attack success", xlabel="Perturbation budget ε", ylabel="Attack success rate", ylim=(-0.04, 1.04))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend(title="Model · dataset", fontsize=9)
    figure.tight_layout()
    save_figure(figure, "current_results_overview")

    tradeoff, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (model, dataset), group in attack_summary.groupby(["model", "dataset"]):
        label = f"{model} · {dataset}"
        axes[0].plot(group["median_ssim"], group["asr"], marker="o", lw=2, label=label)
        axes[1].plot(group["median_lpips"], group["asr"], marker="o", lw=2, label=label)
    axes[0].set(xlabel="Median SSIM (higher is more similar)", ylabel="ASR", title="Success versus structural similarity")
    axes[1].set(xlabel="Median LPIPS (lower is more similar)", ylabel="ASR", title="Success versus perceptual distance")
    for axis in axes:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0)); axis.legend()
    tradeoff.tight_layout()
    save_figure(tradeoff, "current_attack_tradeoff")
    return baseline, attack_summary

