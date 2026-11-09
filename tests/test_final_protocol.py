import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from final_protocol.evidence import Line, localize, infer
from final_protocol.adaptive import search
from final_protocol.reporting import summarize, paired_bootstrap
from stages.final_protocol import trial_key, valid_record


def line(text, page=0, confidence=90):
    return Line(page, text, (0, 0, 90, 15), confidence)


def test_total_not_subtotal_tax_tender():
    answer, regions, _ = localize(
        [
            line("SUBTOTAL 9.00"),
            line("TAX 1.00"),
            line("TOTAL 10.00"),
            line("CASH 20.00"),
        ],
        "sroie",
    )
    assert answer == "10.00" and len(regions) == 1


def test_conflicts_and_low_confidence_abstain():
    assert localize([line("TOTAL 10.00"), line("TOTAL 100.00")], "sroie")[0] is None
    assert localize([line("TOTAL 10.00", confidence=59)], "sroie")[0] is None
    assert localize([line("TOTAL 10.00 100.00")], "sroie")[0] is None


def test_receipt_locales():
    assert localize([line("TOTAL 10.000")], "cord_v2")[0] == "10000"
    assert localize([line("TOTAL 1,000.25")], "collected_receipts")[0] == "1000.25"


def test_resume_all_pages_and_uncertain_degree():
    lines = [
        line("Education"),
        line("Bachelor of Science 2018"),
        line("Master of Science 2020", page=1),
    ]
    assert localize(lines, "resume_parsing_vision")[0] == "master"
    assert (
        localize(lines + [line("PhD expected 2027", page=2)], "resume_parsing_vision")[
            0
        ]
        is None
    )
    assert (
        localize([line("Teaching master students")], "resume_parsing_vision")[0] is None
    )
    assert localize([line("PhD required")], "resume_parsing_vision")[0] is None


def test_inference_agreement_and_label_free_signature():
    import inspect

    assert not {"clean_target", "attack_target", "row", "annotation"} & set(
        inspect.signature(infer).parameters
    )
    page = Image.new("RGB", (100, 100))
    answers = iter(["10.00", "100.00"])
    result = infer(
        [page],
        "sroie",
        "total?",
        lambda p, q: next(answers),
        lambda p: [line("TOTAL 10.00")],
        {"minimum_confidence": 60, "crop_padding_pixels": 16},
    )
    assert result["abstained"] and result["vlm_calls"] == 2


def fixture_frame():
    return pd.DataFrame(
        [
            dict(
                dataset_name="sroie",
                model_key="internvl",
                source_model="qwen",
                attack_type="standard",
                epsilon_255=8,
                condition="png_reload",
                example_type="attack",
                prediction=p,
                clean_target="10.00",
                attack_target="100.00",
                abstained=p is None,
                runtime_seconds=1,
                vlm_calls=1,
                ocr_pages=0,
                status="complete",
                source_clean_correct=True,
                target_clean_correct=i < 2,
                source_attack_success=True,
                record_id=str(i),
            )
            for i, p in enumerate(["100.00", None, "10.00"])
        ]
    )


def test_denominators_and_recovery_do_not_count_abstention():
    result = summarize(fixture_frame()).iloc[0]
    assert result.accepted_targeted_asr == pytest.approx(1 / 3)
    assert result.both_clean_correct_n == 2
    assert result.both_clean_correct_asr == 0.5
    assert result.correct_recovery == pytest.approx(1 / 3)
    assert result.coverage == pytest.approx(2 / 3)


def test_missing_fields_and_failed_evaluations_rejected():
    with pytest.raises(ValueError):
        summarize(fixture_frame().drop(columns="abstained"))
    f = fixture_frame()
    f.loc[0, "status"] = "failed"
    with pytest.raises(ValueError):
        summarize(f)
    f = fixture_frame()
    f["target_clean_correct"] = None
    with pytest.raises(ValueError):
        summarize(f)


def test_no_dataset_pooling_and_paired_matching():
    f = fixture_frame()
    second = f.copy()
    second["dataset_name"] = "cord_v2"
    assert len(summarize(pd.concat([f, second]))) == 2
    second = f.copy()
    second["condition"] = "evidence"
    assert (
        paired_bootstrap(pd.concat([f, second]), "png_reload", "evidence")["difference"]
        == 0
    )
    with pytest.raises(ValueError):
        paired_bootstrap(pd.concat([f, second.iloc[:1]]), "png_reload", "evidence")


def test_source_identity_cannot_collide():
    d = {"dataset_name": "sroie", "record_id": "x"}
    c = {"protocol_version": 1, "model_key": "internvl"}
    assert trial_key(d, "qwen:A", "png_reload", c) != trial_key(
        d, "donut:A", "png_reload", c
    )


def test_query_budget_bounds_and_resume(tmp_path):
    pages = [
        Image.new("RGB", (20, 15), (120, 120, 120)),
        Image.new("RGB", (17, 19), (30, 30, 30)),
    ]
    seen = []

    def evaluate(p):
        seen.append(p)
        return dict(
            prediction=None,
            abstained=True,
            full_prediction="10",
            crop_prediction=None,
            ocr_prediction="10",
        )

    settings = dict(
        seed=566,
        queries=7,
        restart_interval=3,
        square_fraction=0.25,
        report_queries=[3, 7],
        objective="targeted",
    )
    out, state = search(
        pages, [], evaluate, "100", 4, settings, tmp_path / "c.pkl", {"config": "a"}
    )
    assert len(seen) == 7 and state["queries"] == 7
    assert all(
        np.abs(np.asarray(a, dtype=int) - np.asarray(b, dtype=int)).max() <= 4
        for a, b in zip(out, pages)
    )
    search(pages, [], evaluate, "100", 4, settings, tmp_path / "c.pkl", {"config": "a"})
    assert len(seen) == 7
    with pytest.raises(RuntimeError):
        search(
            pages, [], evaluate, "100", 4, settings, tmp_path / "c.pkl", {"config": "b"}
        )


def test_interrupted_search_is_identical(tmp_path):
    pages = [Image.new("RGB", (10, 10), (100, 100, 100))]

    def evaluate(p):
        return dict(
            prediction=None,
            abstained=True,
            full_prediction=None,
            crop_prediction=None,
            ocr_prediction=None,
        )

    settings = dict(
        seed=566,
        queries=9,
        restart_interval=3,
        square_fraction=0.25,
        report_queries=[3, 9],
        objective="targeted",
    )
    full, _ = search(pages, [], evaluate, "100", 8, settings, tmp_path / "full.pkl", {})
    n = 0

    def interrupt(p):
        nonlocal n
        n += 1
        if n == 5:
            raise RuntimeError("timeout")
        return evaluate(p)

    with pytest.raises(RuntimeError):
        search(pages, [], interrupt, "100", 8, settings, tmp_path / "resume.pkl", {})
    resumed, _ = search(
        pages, [], evaluate, "100", 8, settings, tmp_path / "resume.pkl", {}
    )
    assert np.array_equal(np.asarray(full[0]), np.asarray(resumed[0]))


def test_resume_rejects_provenance_drift(tmp_path):
    p = tmp_path / "record.json"
    p.write_text(json.dumps({"status": "complete", "config_sha256": "a"}))
    with pytest.raises(RuntimeError):
        valid_record(p, {"config_sha256": "b"}, tmp_path)


def test_internvl_tiling_bound_and_thumbnail():
    from models.internvl import _dynamic_preprocess

    tiles = _dynamic_preprocess(Image.new("RGB", (900, 1800)))
    assert 1 <= len(tiles) <= 13 and all(p.size == (448, 448) for p in tiles)


def test_whitebox_refreshes_regions_and_resumes(tmp_path):
    import torch
    from final_protocol.whitebox import optimize

    class Objective:
        def parity(self, xs):
            return {}

        def loss(self, xs, grad_page=None):
            loss = sum(x.mean() for x in xs)
            return loss, -loss

    class Wrapper:
        device = torch.device("cpu")

        def make_adaptive_objective(self, *args, **kwargs):
            return Objective()

    pages = [Image.new("RGB", (100, 50), (120, 120, 120))]
    calls = []

    def ocr(p):
        calls.append(1)
        return [line("TOTAL 10.00")]

    settings = dict(steps=3, step_fraction=0.25, processor_tolerance={})
    result, state = optimize(
        Wrapper(),
        pages,
        "total",
        "100",
        "sroie",
        ocr,
        {"minimum_confidence": 60, "crop_padding_pixels": 16},
        4,
        settings,
        tmp_path / "w.pkl",
        {},
    )
    assert len(calls) == 3 and state["step"] == 3
    assert np.asarray(result[0]).min() == 117
    optimize(
        Wrapper(),
        pages,
        "total",
        "100",
        "sroie",
        ocr,
        {"minimum_confidence": 60, "crop_padding_pixels": 16},
        4,
        settings,
        tmp_path / "w.pkl",
        {},
    )
    assert len(calls) == 3


def test_full_stage_five_shard_merge_with_fake_model(tmp_path, monkeypatch):
    import torch
    import experiment_utils
    from checkpoints import sha256_file
    from stages import final_protocol as stage
    from stages.transfer_robustness import TRANSFORMS

    base = json.loads((ROOT / "configs/final_protocol/protocol_v1.json").read_text())
    config = dict(
        base,
        mode="transfer",
        model_key="internvl",
        model=base["models"]["internvl"],
        conditions=list(TRANSFORMS),
        ready_for_production=True,
        run_id="test_full",
        seed=566,
        prompts={"receipt": "total?"},
        expected_clean_counts={"sroie": 2},
    )
    config["runtime_versions"] = stage.runtime_versions()
    page = tmp_path / "p.png"
    Image.new("RGB", (20, 20)).save(page)
    images = [{"path": "p.png", "sha256": sha256_file(page)}]
    plan = []
    for index in range(2):
        plan.append(
            dict(
                record_id=f"sroie/{index}",
                dataset_name="sroie",
                experiment_split="validation",
                clean_target="10.00",
                images=images,
                attacks=[
                    dict(
                        source_id=f"{model}:same-key:{index}",
                        source_model=model,
                        attack_type="standard",
                        epsilon_255=4,
                        attack_target="100.00",
                        source_attack_success=True,
                        images=images,
                    )
                    for model in ["qwen", "donut"]
                ],
            )
        )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    config["plan"] = {"path": "plan.json", "sha256": sha256_file(plan_path)}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    class Wrapper:
        def predict(self, p, q):
            return "10.00"

        def unload(self):
            pass

    monkeypatch.setattr(stage, "model", lambda c: Wrapper())
    monkeypatch.setattr(stage, "repo_root_from_config", lambda c: tmp_path)
    monkeypatch.setattr(
        experiment_utils, "git_commit", lambda *a, **k: "fake-unit-test-commit"
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda: "fake L40S for unit test"
    )
    for worker in range(5):
        stage.run(config, path, f"user{worker+1}", worker, 5)
    records = pd.read_parquet(
        tmp_path / "outputs/final_protocol/test_full/merged/records.parquet"
    )
    assert len(records) == 66 and records.experiment_key.nunique() == 66
    assert records.target_clean_correct.all()
    # Repeating all workers must preserve the exact records without duplication.
    for worker in range(5):
        stage.run(config, path, f"user{worker+1}", worker, 5)
    assert (
        len(
            pd.read_parquet(
                tmp_path / "outputs/final_protocol/test_full/merged/records.parquet"
            )
        )
        == 66
    )


def test_receipt_crops_are_single_page_for_donut():
    page = Image.new("RGB", (100, 100))
    calls = []

    def predict(p, q):
        assert len(p) == 1
        calls.append(1)
        return "10.00"

    result = infer(
        [page],
        "sroie",
        "total?",
        predict,
        lambda p: [line("TOTAL 10.00"), line("AMOUNT DUE 10.00")],
        {"minimum_confidence": 60, "crop_padding_pixels": 16},
    )
    assert result["prediction"] == "10.00" and result["vlm_calls"] == 3


def test_availability_uses_clean_accepted_denominator():
    frame = fixture_frame()
    frame["clean_defense_accepted"] = [False, True, True]
    result = summarize(frame).iloc[0]
    assert result.clean_defense_accepted_n == 2
    assert result.induced_abstention_rate == 0.5


def test_adaptive_source_uses_png_not_jpeg_outcome():
    from final_protocol.reporting import source_outcome

    assert (
        source_outcome({"png_target_success": False, "saved_image_success": True})
        is False
    )
    with pytest.raises(ValueError):
        source_outcome({"saved_image_success": "False"})
    with pytest.raises(ValueError):
        source_outcome({})


def test_preparation_freezes_full_cohort_and_split_sources(tmp_path, monkeypatch):
    import importlib.util
    import shutil

    spec = importlib.util.spec_from_file_location(
        "prepare_final", ROOT / "scripts/prepare_final_protocol.py"
    )
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    for relative in [
        "configs/final_protocol/protocol_v1.json",
        "configs/defenses/qwen_public_final.json",
    ]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, target)
    rows = []
    for dataset, n in [("cord_v2", 100), ("sroie", 126), ("resume_parsing_vision", 75)]:
        rows.extend(
            dict(
                record_id=f"{dataset}/{i}",
                dataset_name=dataset,
                experiment_split="validation",
                clean_target="1.00",
                attack_target="10.00",
                usable=True,
            )
            for i in range(n)
        )
    manifest = tmp_path / "data/processed/master_manifest.csv"
    manifest.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    eligible = {"sroie/0", "resume_parsing_vision/0"}
    baseline = tmp_path / "baseline.parquet"
    pd.DataFrame(
        [
            dict(
                model_key="qwen",
                model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                model_revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
                record_id=r["record_id"],
                clean_correct=r["record_id"] in eligible,
            )
            for r in rows
        ]
    ).to_parquet(baseline)
    page = Image.new("RGB", (20, 20))
    image = tmp_path / "image.png"
    page.save(image)
    sources = []
    for dataset in ["sroie", "resume_parsing_vision"]:
        path = tmp_path / f"{dataset}.parquet"
        pd.DataFrame(
            [
                dict(
                    record_id=f"{dataset}/0",
                    dataset_name=dataset,
                    experiment_split="validation",
                    clean_target="1.00",
                    attack_target="10.00",
                    epsilon_255=e,
                    status="complete",
                    saved_image_success=False,
                    image_path=str(image),
                )
                for e in [2, 4, 8, 16]
            ]
        ).to_parquet(path)
        sources += ["--source", f"qwen:standard:{path}"]
    monkeypatch.setattr(
        prepare,
        "_load_clean_pages",
        lambda root, c, rows: {rid: [page] for rid in rows},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_final",
            "--mode",
            "transfer",
            "--model",
            "internvl",
            "--cohort",
            "public",
            "--run-id",
            "fixture_full",
            "--baseline",
            str(baseline),
        ]
        + sources,
    )
    prepare.main()
    config = json.loads(
        (tmp_path / "outputs/runtime_configs/fixture_full.json").read_text()
    )
    from stages.final_protocol import read_plan, preflight

    plan = read_plan(config, tmp_path)
    assert len(plan) == 301 and sum(len(d["attacks"]) for d in plan) == 8
    preflight(config, tmp_path)
    # Frozen preparation refuses an accidental overwrite of the same run.
    with pytest.raises(SystemExit):
        prepare.main()


def test_collected_donut_export_matches_followup_source_paths(tmp_path):
    from checkpoints import sha256_file
    from stages.collected_attacks import _write_attack_summary
    from stages.defenses import _load_attack_sources
    run = tmp_path / 'outputs/donut_attack/donut_collected_external_v1'
    merged = run / 'merged'
    merged.mkdir(parents=True)
    pd.DataFrame([dict(record_id='collected_receipts/fixture',dataset_name='collected_receipts',
        experiment_split='external_test',clean_target='10.00',attack_target='100.00',
        status='complete',epsilon_255=8,saved_image_success=True,lpips=.1,ssim=.9,
        runtime_seconds=1)]).to_parquet(merged/'records.parquet')
    (merged/'summary.json').write_text('{}')
    _write_attack_summary(run,'donut')
    for template in ['configs/defenses/donut_collected_final.json',
                     'configs/jpeg_eot_attack/donut_collected_full.json',
                     'configs/jpeg_eot_attack/donut_collected_frequency_hardview.json']:
        config=json.loads((ROOT/template).read_text())
        source=config['attack_sources'][0]
        source['sha256']=sha256_file(tmp_path/source['records'])
        source['artifact_root']=str(tmp_path)
        assert len(_load_attack_sources(tmp_path,config))==1
