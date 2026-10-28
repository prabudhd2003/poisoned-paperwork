from __future__ import annotations

import pandas as pd
import pytest

from midterm.reporting import summarize_attacks, summarize_baseline, wilson_interval


def test_attack_summary_keeps_sample_sizes_and_metrics():
    frame = pd.DataFrame(
        {
            "epsilon": [8, 8, 16],
            "success": [True, False, True],
            "lpips": [0.1, 0.2, 0.3],
            "ssim": [0.9, 0.8, 0.7],
            "runtime_seconds": [10, 20, 30],
            "steps": [25, 50, 75],
        }
    )
    summary = summarize_attacks(frame)
    epsilon_8 = summary.loc[summary.epsilon.eq(8)].iloc[0]
    assert epsilon_8.trials == 2
    assert epsilon_8.successes == 1
    assert epsilon_8.asr == 0.5
    assert epsilon_8.median_lpips == pytest.approx(0.15)
    assert epsilon_8.median_ssim == pytest.approx(0.85)


def test_baseline_summary_is_conditional_on_each_model_dataset_pair():
    frame = pd.DataFrame(
        {
            "model_key": ["qwen", "qwen", "donut"],
            "dataset_name": ["sroie", "sroie", "cord_v2"],
            "clean_correct": [True, False, True],
            "parse_status": ["ok", "failed", "ok"],
        }
    )
    summary = summarize_baseline(frame)
    qwen = summary.loc[summary.model.eq("Qwen")].iloc[0]
    assert qwen.correct == 1
    assert qwen.total == 2
    assert qwen.accuracy == 0.5
    assert qwen.parse_failures == 1


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(17, 74)
    assert low < 17 / 74 < high
    assert 0 <= low <= high <= 1
