"""Strict denominators and document-paired uncertainty for final experiments."""

import numpy as np
import pandas as pd

GROUPS = [
    "dataset_name",
    "model_key",
    "source_model",
    "attack_type",
    "epsilon_255",
    "condition",
    "example_type",
]


def summarize(frame):
    required = GROUPS + [
        "prediction",
        "clean_target",
        "attack_target",
        "abstained",
        "runtime_seconds",
        "vlm_calls",
        "ocr_pages",
        "status",
        "source_clean_correct",
        "target_clean_correct",
    ]
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"missing outcome columns: {sorted(missing)}")
    if frame[GROUPS].isna().any().any():
        raise ValueError("missing grouping/outcome data")
    if not frame.status.eq("complete").all():
        raise ValueError("failed evaluations cannot be treated as attack failures")
    rows = []
    for key, g in frame.groupby(GROUPS, dropna=False):
        n = len(g)
        accepted = ~g.abstained.astype(bool)
        correct = accepted & g.prediction.eq(g.clean_target)
        targeted = accepted & g.prediction.eq(g.attack_target)
        both = g.source_clean_correct.eq(True) & g.target_clean_correct.eq(True)
        source = g.source_clean_correct.eq(True)
        for col in ["abstained", "source_clean_correct", "target_clean_correct"]:
            if g[col].isna().any() or not g[col].isin([True, False]).all():
                raise ValueError(f"missing or invalid {col}")
        original = g.get("source_attack_success", pd.Series(None, index=g.index))
        recovery_pool = original.eq(True)
        clean_accepted = g.get(
            "clean_defense_accepted", pd.Series(None, index=g.index)
        ).eq(True)
        rows.append(
            dict(
                zip(GROUPS, key),
                n=n,
                accepted=int(accepted.sum()),
                clean_accuracy=float(correct.mean()),
                coverage=float(accepted.mean()),
                selective_accuracy=(
                    float(correct.sum() / accepted.sum()) if accepted.any() else None
                ),
                selective_risk=(
                    float(1 - correct.sum() / accepted.sum())
                    if accepted.any()
                    else None
                ),
                abstention_rate=float((~accepted).mean()),
                accepted_targeted_successes=int(targeted.sum()),
                accepted_targeted_asr=float(targeted.mean()),
                targeted_rate_among_accepted=(
                    float(targeted.sum() / accepted.sum()) if accepted.any() else None
                ),
                source_eligible_n=int(source.sum()),
                source_eligible_asr=(
                    float(targeted[source].mean()) if source.any() else None
                ),
                both_clean_correct_n=int(both.sum()),
                both_clean_correct_asr=(
                    float(targeted[both].mean()) if both.any() else None
                ),
                originally_successful_n=int(recovery_pool.sum()),
                clean_defense_accepted_n=int(clean_accepted.sum()),
                induced_abstention_rate=(
                    float((~accepted)[clean_accepted].mean())
                    if clean_accepted.any()
                    else None
                ),
                targeted_asr_on_clean_accepted=(
                    float(targeted[clean_accepted].mean())
                    if clean_accepted.any()
                    else None
                ),
                source_outcome_missing=int(original.isna().sum()),
                correct_recovery=(
                    float(correct[recovery_pool].mean())
                    if recovery_pool.any()
                    else None
                ),
                mean_seconds=float(g.runtime_seconds.mean()),
                mean_vlm_calls=float(g.vlm_calls.mean()),
                mean_ocr_pages=float(g.ocr_pages.mean()),
            )
        )
    return pd.DataFrame(rows)


def paired_bootstrap(frame, first, second, metric="targeted", seed=566, repeats=2000):
    """Call within one dataset/model/source/epsilon stratum. Cluster by document."""
    keys = ["record_id", "source_model", "attack_type", "epsilon_255", "example_type"]
    work = frame.copy()
    work["value"] = (
        ~work.abstained
        & work.prediction.eq(
            work.attack_target if metric == "targeted" else work.clean_target
        )
    ).astype(float)
    a = work.loc[work.condition.eq(first), keys + ["value"]]
    b = work.loc[work.condition.eq(second), keys + ["value"]]
    joined = a.merge(b, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all() or joined.empty:
        raise ValueError("comparison requires identical nonempty trial sets")
    delta = (
        (joined.value_x - joined.value_y).groupby(joined.record_id).mean().to_numpy()
    )
    rng = np.random.default_rng(seed)
    draws = np.array(
        [rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)]
    )
    return dict(
        difference=float(delta.mean()),
        low=float(np.quantile(draws, 0.025)),
        high=float(np.quantile(draws, 0.975)),
        documents=len(delta),
    )


def matched_source_intersection(frame):
    """Keep documents attempted by both source models, within each attack stratum."""
    attack = frame.loc[frame.example_type.eq("attack")]
    parts = []
    for _, group in attack.groupby(
        ["dataset_name", "attack_type", "epsilon_255", "model_key", "condition"],
        dropna=False,
    ):
        a = set(group.loc[group.source_model.eq("qwen"), "record_id"])
        b = set(group.loc[group.source_model.eq("donut"), "record_id"])
        parts.append(group.loc[group.record_id.isin(a & b)])
    return pd.concat(parts, ignore_index=True) if parts else attack.iloc[:0]


def source_outcome(row):
    """Saved-PNG outcome; JPEG-EOT's legacy saved_image_success means JPEG."""
    for name in [
        "png_target_success",
        "saved_png_target_success",
        "saved_image_success",
        "targeted_success_reloaded",
        "reloaded_success",
        "attack_success",
    ]:
        value = row.get(name)
        if value is None or pd.isna(value):
            continue
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"nonboolean source outcome: {name}")
        return bool(value)
    raise ValueError("missing explicit saved-PNG source outcome")
