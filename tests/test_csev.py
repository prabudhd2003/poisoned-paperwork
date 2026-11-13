import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from csev.defense import parse_box, control_box, mask, infer, prediction
from csev.study import select, calibrate, bounds, digest
from csev.adaptive import search, project


@pytest.fixture
def settings():
    return json.loads((ROOT / "configs/csev/protocol_v1.json").read_text())


def test_invalid_localization_is_rejection_not_arbitrary_crop(settings):
    for text in (
        "garbage",
        '{"box":null}',
        '{"box":[0,0,1001,1]}',
        '{"box":[900,0,1,50]}',
        '{"box":[true,0,40,50]}',
    ):
        assert parse_box(text, (1000, 1000), settings) is None
    assert parse_box('{"box":[0,0,1000,1000]}', (1000, 1000), settings) is None
    b = parse_box('```json\n{"box":[900,900,910,910]}\n```', (100, 100), settings)
    assert b and b[2] - b[0] >= 28


def test_equal_control_is_disjoint_and_mask_local(settings):
    box = (10, 10, 40, 40)
    control = control_box(box, (100, 100))
    assert control[2] - control[0] == 30 and control[3] - control[1] == 30
    assert control[0] >= box[2] or control[1] >= box[3]
    im = Image.new("RGB", (100, 100), "white")
    m = mask(im, box, 128)
    assert m.getpixel((10, 10)) == (128, 128, 128) and m.getpixel((0, 0)) == (
        255,
        255,
        255,
    )
    assert im.getpixel((10, 10)) == (255, 255, 255)


def test_label_free_verifier_direction_and_crop_ablation(settings):
    answers = iter(["10.00", '{"box":[100,100,300,300]}', "10.00"])
    scores = iter([-1.0, -3.0])
    obs = infer(
        Image.new("RGB", (200, 200)),
        "sroie",
        "total",
        lambda *_: next(answers),
        lambda *_: next(scores),
        settings,
    )
    assert (
        obs["gap"] == 2 and prediction(obs, 1) == "10.00" and prediction(obs, 3) is None
    )
    assert obs["crop_agrees"]
    answers = iter(["10.00", '{"box":[100,100,300,300]}', "20.00"])
    obs = infer(
        Image.new("RGB", (200, 200)),
        "sroie",
        "total",
        lambda *_: next(answers),
        lambda *_: pytest.fail("must not score disagreement"),
        settings,
    )
    assert prediction(obs, 0) is None


def test_model_errors_do_not_become_rejections(settings):
    def bad(*_):
        raise RuntimeError("CUDA OOM")

    with pytest.raises(RuntimeError, match="CUDA"):
        infer(Image.new("RGB", (100, 100)), "sroie", "?", bad, bad, settings)


def test_selection_does_not_use_outcomes(settings):
    docs = [
        dict(record_id=str(i), clean_target="1", eligible=i % 2 == 0)
        for i in range(100)
    ]
    first = select(docs, 60, 566, "external")
    altered = [
        dict(d, eligible=not d["eligible"], clean_target="2") for d in reversed(docs)
    ]
    assert [d["record_id"] for d in first] == [
        d["record_id"] for d in select(altered, 60, 566, "external")
    ]


def test_calibration_reports_infeasible_goal_and_never_relaxes(settings):
    rows = [
        dict(
            full_prediction="1",
            clean_target="1",
            crop_agrees=i < 4,
            gap=1 if i < 4 else None,
        )
        for i in range(10)
    ]
    c = calibrate(rows, settings)
    assert (
        not c["development_goal_met"] and c["threshold"] == 0 and c["coverage"] == 0.4
    )
    rows = [
        dict(full_prediction="1", clean_target="1", crop_agrees=True, gap=i)
        for i in range(10)
    ]
    c = calibrate(rows, settings)
    assert c["development_goal_met"] and c["threshold"] == 2 and c["coverage"] == 0.8


def test_bounds_detect_wrong_pair_and_projection():
    clean = np.full((20, 20, 3), 128, dtype="uint8")
    candidate = project(clean.astype(float) + 100, clean, 4)
    assert bounds([Image.fromarray(clean)], [Image.fromarray(candidate)], 4) == 4
    with pytest.raises(ValueError):
        bounds([Image.fromarray(clean)], [Image.fromarray(candidate)], 2)


class ToyVerifier:
    prompt = "total"

    def __call__(self, im, dataset):
        yes = np.asarray(im).mean() > 129
        return dict(
            full_prediction="100" if yes else "10",
            crop_agrees=True,
            gap=2.0,
            box=(1, 1, 3, 3),
            control=(5, 5, 7, 7),
        )

    def score(self, im, prompt, target):
        return float(np.asarray(im).mean())


def test_checkpoint_resume_query_budget_and_saved_png(tmp_path, settings):
    s = copy.deepcopy(settings)
    s["adaptive"].update(steps=4, queries=8, refresh_interval=2)
    clean = Image.fromarray(np.full((12, 12, 3), 128, dtype="uint8"))

    def step(im, obs):
        return np.asarray(im).astype(float) + 1, 1.0

    v = ToyVerifier()
    contract = digest(["toy"])
    full = search(
        clean,
        "sroie",
        "100",
        4,
        0,
        s,
        1.0,
        v,
        step,
        tmp_path / "full/state.pkl",
        contract,
    )
    calls = [0]

    def interrupted(im, obs):
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError("interrupt")
        return step(im, obs)

    with pytest.raises(RuntimeError):
        search(
            clean,
            "sroie",
            "100",
            4,
            0,
            s,
            1.0,
            v,
            interrupted,
            tmp_path / "resume/state.pkl",
            contract,
        )
    resumed = search(
        clean,
        "sroie",
        "100",
        4,
        0,
        s,
        1.0,
        v,
        step,
        tmp_path / "resume/state.pkl",
        contract,
    )
    assert full["queries"] == resumed["queries"] == 8
    assert full["step"] == resumed["step"] == 4
    assert full["trace"] == resumed["trace"]
    assert np.array_equal(full["best"], resumed["best"])
    assert resumed["linf_255"] <= 4 and resumed["final_prediction"] == "100"
    with pytest.raises(ValueError):
        search(
            clean,
            "sroie",
            "100",
            4,
            0,
            s,
            1.0,
            v,
            step,
            tmp_path / "resume/state.pkl",
            "changed",
        )


def test_gradient_attacks_masking_score_with_toy_model(monkeypatch, settings):
    import torch
    import csev.adaptive as module

    class Obj:
        def parity(self, xs):
            return {}

        def loss(self, xs):
            loss = (xs[0] - 0.8).square().mean()
            return loss, -loss

    monkeypatch.setattr(module, "answer_objective", lambda *a, **kw: Obj())

    class Wrapper:
        device = torch.device("cpu")

    image = Image.fromarray(np.full((50, 50, 3), 128, dtype="uint8"))
    obs = dict(box=(5, 5, 20, 20), control=(30, 30, 45, 45))
    updated, loss = module.gradient_step(
        Wrapper(), image, np.asarray(image), "100", "total", obs, 1.0, 4, settings
    )
    assert np.isfinite(loss) and not np.array_equal(updated, np.asarray(image))
    assert bounds([image], [Image.fromarray(updated)], 4) <= 4


def test_strict_merge_and_external_calibration_separation(tmp_path, settings):
    from checkpoints import atomic_json
    from csev.runner import tasks, signature, merge

    config = dict(settings=settings, output=str(tmp_path), run_id="test")
    dev = [
        dict(
            record_id=f"dev{i}",
            dataset_name="cord_v2",
            clean_target="10",
            frozen_attack_target="100",
            images=[],
            attacks=[],
        )
        for i in range(60)
    ]
    plan = dict(development=dev, evaluation=[], pilot_ids=[])
    todo = tasks(config, plan, "development")
    sig = signature(config, "development")
    for i, t in enumerate(todo):
        atomic_json(
            tmp_path / "development/records" / (t["key"] + ".json"),
            dict(
                key=t["key"],
                signature=sig,
                worker=i % 5,
                status="complete",
                clean_target="10",
                full_prediction="10",
                crop_agrees=True,
                gap=1.0,
            ),
        )
    for w in range(4):
        atomic_json(
            tmp_path / f"development/worker_{w}_complete.json",
            dict(signature=sig, keys=[t["key"] for t in todo[w::5]]),
        )
    with pytest.raises(FileNotFoundError):
        merge(config, plan, "development")
    atomic_json(
        tmp_path / "development/worker_4_complete.json",
        dict(signature=sig, keys=[t["key"] for t in todo[4::5]]),
    )
    merge(config, plan, "development")
    c = json.loads((tmp_path / "calibration.json").read_text())
    assert c["development_goal_met"]
    before = (tmp_path / "calibration.json").read_bytes()
    plan["evaluation"] = [dict(record_id="test", clean_target="wrong")]
    merge(config, plan, "development")
    assert before == (tmp_path / "calibration.json").read_bytes()
    atomic_json(tmp_path / "development/records/extra.json", {})
    with pytest.raises(ValueError, match="missing or extra"):
        merge(config, plan, "development")


def test_prepare_freezes_full_grid_and_preserves_originals(
    tmp_path, monkeypatch, settings
):
    import pandas as pd
    import csev.study as study
    import stages.defenses as defenses
    from checkpoints import atomic_json, sha256_file

    root = tmp_path / "repo"
    root.mkdir()
    for relative in (
        "configs/csev/protocol_v1.json",
        "configs/defenses/qwen_public_final.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    pinned = json.loads((ROOT / "configs/defenses/qwen_public_final.json").read_text())
    image = Image.new("RGB", (32, 32), (128, 128, 128))
    ip = root / "clean.png"
    image.save(ip)
    items = [dict(path=str(ip), sha256=sha256_file(ip))]
    public = [
        dict(
            record_id=f"{ds}/{i}",
            dataset_name=ds,
            clean_target="10",
            images=items,
            attacks=[],
        )
        for ds in ("cord_v2", "sroie")
        for i in range(30)
    ]
    pp = root / "public.json"
    atomic_json(pp, public)
    legacy = dict(
        model_key="qwen",
        model=pinned["model"],
        plan=dict(path=str(pp), sha256=sha256_file(pp)),
    )
    cp = root / "public_config.json"
    atomic_json(cp, legacy)
    frame = pd.DataFrame(
        [
            dict(
                record_id=f"collected_receipts/{i}",
                dataset_name="collected_receipts",
                experiment_split="external",
                clean_target="10",
                attack_target="100",
            )
            for i in range(114)
        ]
    )
    mp = root / "manifest.csv"
    frame.to_csv(mp, index=False)
    atomic_json(
        root / "data/processed/collected_receipts/manifest_contract.json",
        dict(manifest=str(mp), manifest_sha256=sha256_file(mp)),
    )
    bp = root / "baseline.parquet"
    pd.DataFrame(
        [
            dict(
                record_id=r.record_id,
                model_key="qwen",
                model_id=pinned["model"]["model_id"],
                model_revision=pinned["model"]["revision"],
                clean_correct=i < 110,
            )
            for i, r in enumerate(frame.itertuples())
        ]
    ).to_parquet(bp)
    attack = []
    for row in frame.iloc[:110].to_dict("records"):
        for e in (4, 8, 16):
            attack.append(
                dict(
                    **row,
                    epsilon_255=e,
                    status="complete",
                    images=[dict(path=str(ip))],
                    model_id=pinned["model"]["model_id"],
                    model_revision=pinned["model"]["revision"],
                )
            )
    sp = root / "source.parquet"
    pd.DataFrame(attack).to_parquet(sp)
    monkeypatch.setattr(study, "git_commit", lambda _: "testcommit")
    monkeypatch.setattr(study, "runtime_versions", lambda: {})
    monkeypatch.setattr(
        defenses,
        "_load_clean_pages",
        lambda root, config, rows: {rid: [image] for rid in rows},
    )
    original = sp.read_bytes()
    dest = study.prepare(
        root,
        "test",
        cp,
        bp,
        {f: sp for f in settings["families"]},
        tmp_path / "scratch",
    )
    c = json.loads(dest.read_text())
    plan = json.loads(Path(c["plan"]).read_text())
    assert (
        len(plan["development"]) == len(plan["evaluation"]) == 60
        and len(plan["pilot_ids"]) == 20
    )
    for d in plan["evaluation"]:
        assert len(d["attacks"]) == (9 if d["eligible"] else 0)
        for a in d["attacks"]:
            assert Path(a["images"][0]["path"]).is_file()
    assert sp.read_bytes() == original
    with pytest.raises(ValueError, match="already exists"):
        study.prepare(
            root,
            "test",
            cp,
            bp,
            {f: sp for f in settings["families"]},
            tmp_path / "scratch",
        )


def test_reporting_exports_matched_intervals_and_figures(
    tmp_path, monkeypatch, settings
):
    import csev.reporting as reporting
    from checkpoints import sha256_file
    from csev.runner import score_record
    import pandas as pd

    s = copy.deepcopy(settings)
    s["conditions"] = ["none", "csev"]
    s["bootstrap_repeats"] = 20
    config = dict(output=str(tmp_path), run_id="toy", settings=s)
    im = tmp_path / "input.png"
    Image.new("RGB", (40, 40), "white").save(im)
    images = [dict(path=str(im), sha256=sha256_file(im))]
    evaluation = []
    replay = []
    adaptive = []
    for i in range(2):
        d = dict(clean_target="10", frozen_attack_target="100")
        obs = dict(
            full_prediction="10",
            crop_agrees=True,
            gap=1.0,
            box=[0, 0, 10, 10],
            control=[20, 20, 30, 30],
        )
        for family, epsilon in [("clean", 0), ("standard", 4), ("direct_adaptive", 4)]:
            r = dict(
                key=str(i),
                record_id=str(i),
                family=family,
                epsilon=epsilon,
                clean_target="10",
                attack_target="100",
                images=images,
                observation=obs,
                results=[
                    dict(condition=c, **score_record("10", d)) for c in s["conditions"]
                ],
            )
            (replay if family == "direct_adaptive" else evaluation).append(r)
        adaptive.append(
            dict(
                record_id=str(i),
                epsilon=4,
                restart=0,
                seen_rejection=False,
                accepted=True,
                targeted=False,
                linf_255=4,
                lpips=0.1,
                ssim=0.9,
                queries=100,
                steps=100,
                attack_runtime_seconds=1.0,
                trace=[],
            )
        )
    monkeypatch.setattr(
        reporting,
        "merge",
        lambda c, p, phase: dict(
            records={"evaluation": evaluation, "replay": replay, "adaptive": adaptive}[
                phase
            ]
        ),
    )
    monkeypatch.setattr(reporting, "calibration", lambda c: dict(threshold=0.0))
    dest = reporting.analyze(config, {})
    summary = pd.read_csv(dest / "summary.csv")
    assert summary.n.eq(2).all() and summary.correct_answer_rate.eq(1).all()
    paired = pd.read_csv(dest / "paired_bootstrap.csv")
    assert paired.difference.eq(0).all()
    assert (dest / "coverage_tradeoff.png").is_file() and (
        dest / "analysis_complete.json"
    ).is_file()


def test_submission_dependency_chain_and_duplicate_guard(tmp_path, monkeypatch):
    import importlib.util
    import subprocess
    import shlex
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "submit_csev_test", ROOT / "scripts/submit_csev.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "outputs/runtime_configs").mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "read_config", lambda *a: ({"run_id": "test"}, {}))
    monkeypatch.setattr(module, "preflight", lambda *a: None)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "")
    commands = []
    actual_run = subprocess.run

    def fake_run(cmd, **kw):
        commands.append(cmd)
        shell = shlex.split(cmd[-1])
        assert shell[:2] == ["bash", "-lc"]
        actual_run(["bash", "-n"], input=shell[2], text=True, check=True)
        return SimpleNamespace(
            returncode=0, stdout=str(100 + len(commands)) + "\n", stderr=""
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["submit", "--config", str(tmp_path / "config.json"), "--submit"]
    )
    module.main()
    assert len(commands) == 4
    assert (
        "--dependency=afterok:101" in commands[1]
        and "--dependency=afterok:101" in commands[2]
    )
    assert "--dependency=afterok:103" in commands[3]
    assert all("--gpus-per-task=l40s:1" in c and "--array=0-4" in c for c in commands)
    with pytest.raises(SystemExit, match="ledger exists"):
        module.main()
    assert len(commands) == 4
