# Qwen receipt attack contract

## Status and ownership

- Contract status: **Phase 0 complete; Phase 1 tasks 1-6 implemented and the
  CARC real-artifact CPU handoff accepted; the Task 7 closure unlocks only the
  owner GPU smoke. No receipt GPU attack has run, and production remains
  locked**.
- Development branch: `qwen-receipt`.
- Branch base commit: `f269fa759e8cfd0f8736fbb27d96bc0c0791d30a`.
- Verification commit: `94a1725e0ae518351064e7cc95ddcbb189def7b7`.
- Phase 1 tasks 1-6 implementation commit:
  `fa5934fd6d3a1345da1fd0130998c297ace2e61a`.
- Track owner: Saaketh (`user3`, production worker `2`).
- Shared attack core owner/reviewer: Shail (`user5`).

This file freezes the receipt track's scientific and provenance contract and
records the implementation boundary. Tasks 1-6 add the code and CPU-verifiable
contracts; they do not run Qwen or an adversarial attack. A completed run ID is
immutable: any later change to code, inputs, or selected settings requires a
new run ID.

## Phase 1 tasks 1-6 implementation

1. **Personal/shared CARC layout.** Receipt configs name
   `PP_SHARED_ARTIFACT_ROOT`; frozen baseline outputs and processed datasets are
   resolved below that separate, read-only checkout. Configs and all new outputs
   remain below the personal `qwen-receipt` checkout. Absolute or escaping
   receipt input paths, an unset variable, and using the personal checkout as
   the shared root are rejected.
2. **Frozen configs.** `receipt_smoke_saaketh.json` fixes the owner smoke at five
   deterministic candidates per dataset, epsilon 8, step size 2, and 25 steps.
   The Task 7 closure pins the accepted complete source-pixel evidence digest
   and changes only its `ready_for_gpu_smoke` gate to true. That authorizes the
   later owner-only smoke; it does not claim that a GPU job has run or authorize
   five-worker validation.
   `receipt_active.json` fixes the intended 219-document validation grid but
   remains deliberately blocked with `ready_for_production: false` until the
   smoke, saved-image review, and checkpoint-resume proof are accepted.
3. **Candidate/data contract.** The receipt stage re-verifies all seven frozen
   artifact hashes, the 527-record baseline provenance, the 226 usable receipt
   rows, the 219 clean-correct target-valid candidates, canonical exact-decimal
   targets, and one-page metadata. CPU preflight additionally loads the ten
   selected smoke images and proves each is one nonempty RGB page.
4. **Shared attack integration.** The receipt adapter uses the existing
   `attacks.qwen.attack` and `models.qwen.QwenVL` implementations unchanged. It
   supplies the frozen receipt prompt and dataset-aware SROIE/CORD output
   normalization; no second optimizer or model wrapper is introduced.
5. **Immutable output and merge contract.** Experiment keys use the full record
   ID SHA-256. Resume validation binds source pixels, target, model/prompt,
   processor settings and parity, epsilon/step size, success recomputation,
   metrics, Git/config/input hashes, package versions, and Python/CUDA/GPU/VRAM.
   The strict merge requires the complete worker/epsilon Cartesian product,
   identical worker runtime contracts, immutable assignments, valid saved RGB
   PNGs, and exact deterministic manifest/output hashes. It publishes via an
   atomic directory rename and reports conditional ASR over 123/96 candidates
   plus harmful-output rate over 126/100 usable documents.
6. **CPU verification surface.** `tests/test_qwen_receipt.py` covers config,
   target, selection, routing, path containment, real-image preflight, sharding,
   record validation, runtime provenance, merge tamper detection, and both
   denominators. `smoke_qwen_attack.py --preflight` loads no Qwen model and
   performs no attack. `freeze_qwen_receipt_sources.py` hashes the decoded RGB
   pixels and dimensions for exactly all 219 candidates into a CPU evidence
   report. Its canonical source-record digest must be pinned in a subsequent
   closure commit before GPU work. `merge_qwen_receipt.py` exposes the same
   strict merge for a CPU login node after workers finish; that merge
   independently recomputes pixel-space L-infinity from the processed clean RGB
   bytes and saved PNG. Processor-parity execution still belongs to the later
   full model-runtime/GPU readiness work.

The five-worker validation config is code-complete but not operationally
unlocked. Before that later run, every worker must write to one common writable
run root; separate personal checkout output directories cannot be merged. This
does not affect the single-owner smoke or the accepted CPU preflight handoff.

## Phase 1 CPU handoff and source-evidence closure

The CARC Phase 1 CPU handoff passed against the tasks 1-6 implementation at
commit `fa5934fd6d3a1345da1fd0130998c297ace2e61a`. The personal worktree was
clean before and after execution. The accepted evidence establishes the exact
receipt source population and confirms that the real processed images can be
loaded without invoking Qwen or an attack:

- Focused receipt CPU tests: 24 of 24 passed. This CARC execution did **not**
  rerun the previously reported 70-test local suite, so this record makes no
  claim that 70 tests ran on CARC.
- Source-evidence verifier status: `pass`.
- Source records: 219 total, comprising 123 SROIE and 96 CORD v2 records.
- Real-image smoke preflight: 10 receipts, comprising 5 SROIE and 5 CORD v2
  records. It printed all ten IDs, clean targets, attack targets, and decoded
  RGB dimensions.
- Canonical source-record SHA-256:
  `77480634c85f37271c1927798bd975858877da6c1ebb06442c549730d58da277`.
- Accepted CPU evidence report SHA-256:
  `7682d9fc5f5601fa1cc9d4dd368b75d9df44950772185d8a9a9a85cb61cf609d`.
- Evidence directory:
  `/project2/yzhao010_1531/kanduri/poisoned-paperwork-qwen-receipt/outputs/phase1_cpu/fa5934f-20261014T220123Z`.
- Source-evidence report: `qwen_receipt_source_evidence.json`.

The canonical source-record digest is the semantic identity of the ordered
219-record source set. It is computed from the record identity, dataset/source
selection, targets, decoded RGB dimensions, and source-pixel hashes, and must
remain stable when the same frozen sources and policy are reverified. The full
report hash instead identifies this historical CARC report byte for byte. A
later report may legitimately have a different report SHA-256 because it can
contain new Git or execution metadata; that does not permit its canonical
source-record digest to change.

The accepted CARC run used Python 3.11.17 through the explicit interpreter

```text
/project2/yzhao010_1531/kanduri/envs/qwen-receipt-phase1-cpu/bin/python
```

with `-E -s`. Commands must continue to name that interpreter for this CPU
verification environment instead of relying on the shell prompt or an
unqualified `python`, because Conda activation previously selected the shared
base interpreter incorrectly. The run read frozen inputs from the shared
artifact root

```text
/project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
```

and kept code, generated reports, and other personal outputs below

```text
/project2/yzhao010_1531/kanduri/poisoned-paperwork-qwen-receipt
```

The Task 7 closure binds the stable source-record digest and records the
historical report hash. It unlocks only `receipt_smoke_saaketh.json` for the
future owner GPU smoke. `receipt_active.json` remains fail-closed with
`ready_for_production: false`. This CPU evidence does not validate the full
Qwen/CUDA environment, processor parity, gradients, attack success, saved
adversarial PNGs, GPU memory, or checkpoint resume, and none of those actions
was performed by this handoff.

## Frozen task scope

- Stage: `qwen_attack`.
- Track: `receipt`.
- Task: `receipt_total`.
- Datasets, in this exact order: `sroie`, `cord_v2`.
- Document shape: exactly one RGB receipt image per record.
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Model revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Prompt ID: `receipt_total_v1`.
- Prompt text: `What is the final total amount shown on this receipt? Return only the amount, with no currency symbol, label, or explanation.`
- Random seed: `566`.
- Pixel-space epsilon grid: `2`, `4`, `8`, and `16` on the 0-255 scale.
- Checkpoint interval: every `10` optimization steps.

Run IDs are reserved as follows:

- Owner smoke: `qwen_receipt_smoke_saaketh_v1`.
- Five-worker validation: `qwen_receipt_validation_v1`.
- Frozen held-out test: `qwen_receipt_test_v1`.

## Candidate and target policy

The baseline candidate pool contains only rows that are usable,
attack-eligible, complete, parseable, and clean-correct for the frozen Qwen
configuration on the declared split.

The approved receipt target policy is:

1. The clean total is canonical for its dataset and strictly positive.
2. The attack target is canonical for its dataset.
3. Using exact decimal arithmetic, the attack target equals ten times the
   clean total.
4. The attack target differs from the clean total.

Rows that fail the positive/distinct-target policy are not silently removed.
They are listed with an exclusion reason, remain part of the clean baseline,
and are excluded from the conditional targeted-attack denominator. The
harmful-output-rate denominator remains all usable documents for the task.

The accepted validation baseline currently reports the following pre-policy
Qwen receipt pool. The verifier must recompute these counts from the frozen
artifacts rather than trusting this text:

| Dataset | Pre-policy candidates | Target-valid candidates |
|---|---:|---:|
| SROIE | 123 | 123 |
| CORD v2 | 96 | 96 |
| Total | 219 | 219 |

The verified usable Qwen receipt baseline contains 126 SROIE and 100 CORD v2
records (226 total). Seven usable records are outside the clean-correct
pre-policy candidate pool: three SROIE and four CORD v2. All 219 pre-policy
candidates satisfy the proposed target policy, so the verifier reported zero
policy exclusions.

## Frozen input artifacts

The CARC verification reads these artifacts without modifying them:

- `configs/baseline/active.json`
- `configs/qwen_attack/resume_active.json` (accepted baseline hash reference)
- `outputs/baseline/baseline_validation_v1/config.json`
- `outputs/baseline/baseline_validation_v1/merged/predictions.parquet`
- `outputs/baseline/baseline_validation_v1/merged/merge_manifest.json`
- `outputs/baseline/baseline_validation_v1/merged/summary.json`
- `outputs/baseline/baseline_validation_v1/merged/qwen_clean_correct_ids.txt`
- `outputs/baseline/baseline_validation_v1/shards/worker_00/environment.json`
- `data/processed/master_manifest.parquet`

The CARC verifier confirmed these hashes against the actual files:

| Artifact | SHA-256 |
|---|---|
| Active and frozen baseline config | `a371655bf97c99e6c08ac5da2b288ca48bf301ad0c2bdb21e4260c588266d05a` |
| Resume attack config (baseline-hash reference) | `31843833539eaf1bdb4c653cf949b30b0df71202032cb0e922d8181f057d99f8` |
| Baseline predictions | `483e12b88167555d94e943d0179314de25016333e409417b34a6f922c46caeb0` |
| Baseline merge manifest | `7a9ab78153ed3dd07d1d543aeda70f055f0b395820dffe8f2236bc797586640e` |
| Baseline summary | `cf491a9b3a044c019f021e328aeb0e93e3e898dca5022329c6a9c72f1ac4d001` |
| Qwen clean-correct IDs | `ea32595020d47058eb54bd2a253ce944492e3accf3c8a7c89599f7a9bb68d4bf` |
| Representative worker environment | `69da33d1992d602936d6e88d99d3cfe1fd7639b81dbee16b030e016e800a2e64` |
| Processed master manifest | `062348979989a7fa10fecb25bb7d0b87fe82984d6d7f37e9477a4b38e0e99162` |

The immutable output copy and active baseline config matched exactly. The
worker-00 environment remains a representative environment rather than proof
that every worker or the future GPU runtime has the same installed packages.

## Verified baseline provenance

- Baseline run ID: `baseline_validation_v1`.
- Baseline Git commit: `5e387062d29b2a9b2b79de03c305c52d7b97f01f`.
- Prediction records: 527 across 301 unique documents.
- Workers represented: `0`, `1`, `2`, `3`, and `4`.
- Saved environment scope: `worker_00_representative`.
- Baseline groups: Donut SROIE 126, Donut CORD v2 100, Qwen SROIE 126,
  Qwen CORD v2 100, and Qwen resume parsing 75.
- Receipt-manifest and Qwen-baseline record IDs matched exactly.
- One-page coverage was verified from metadata only; image pixels were not
  inspected.

## Optimization-setting selection

The owner smoke is a plumbing and feasibility check at epsilon `8`, step size
`2`, and initially `25` steps. It does not select final scientific settings.

The intended production ceiling is `100` steps. If a comparison is required,
the following candidate step-size schedules are evaluated on the same fixed
validation subset:

- Current Qwen/resume candidate for epsilons `[2, 4, 8, 16]`:
  `[0.5, 1, 2, 4]`.
- Documented Donut project default for the same grid: `[1, 1, 2, 2]`.

Settings are selected using validation data only. They are frozen, hashed, and
linked from the test config before the held-out test is run. Test results are
never used for retuning.

## Phase 0 completion record

The passing CARC run used a clean personal checkout and explicit read-only
path overrides for the shared baseline outputs and processed master manifest.
It did not switch the shared checkout to this branch or modify any frozen
artifact. The command pattern was:

```bash
python -m unittest discover -s tests -p 'test_qwen_receipt_phase0.py' -v
python scripts/verify_qwen_receipt_phase0.py \
  --baseline-frozen-config "$BASELINE/config.json" \
  --baseline-predictions "$BASELINE/merged/predictions.parquet" \
  --baseline-merge-manifest "$BASELINE/merged/merge_manifest.json" \
  --baseline-summary "$BASELINE/merged/summary.json" \
  --baseline-correct-ids "$BASELINE/merged/qwen_clean_correct_ids.txt" \
  --baseline-environment "$BASELINE/shards/worker_00/environment.json" \
  --master-manifest "$SHARED/data/processed/master_manifest.parquet" \
  --output "$REPORT_DIR/qwen_receipt_phase0_report.json"
```

The verifier is CPU-only. It reads hashes, JSON, text, and Parquet metadata; it
does not import PyTorch, load Qwen, initialize CUDA, download a model, inspect
image pixels, or run an attack. Its one-page result is explicitly a metadata
check comparing baseline `input_page_count` with manifest `num_pages`.

The passing verification record is:

- Execution time: `2026-10-14T20:48:35Z`.
- Verification Git commit: `94a1725e0ae518351064e7cc95ddcbb189def7b7`.
- Git worktree: clean.
- Focused tests: 10 of 10 passed.
- Verifier status: `pass`.
- Evidence directory relative to the personal checkout:
  `outputs/phase0/94a1725-20261014T204835Z`.
- Verification report SHA-256:
  `8473d2aa070040603656e6244d0a2840ebb8736051dd94860a4f4ee65e1d9a38`.
- Verified usable counts: SROIE 126, CORD v2 100, total 226.
- Verified pre-policy counts: SROIE 123, CORD v2 96, total 219.
- Verified target-valid counts: SROIE 123, CORD v2 96, total 219.
- Excluded record IDs and reasons: none.
- Target policy approval: explicitly approved by the track owner on
  `2026-10-14`.
- Phase 0 closure: the documentation commit that changes this contract status
  to complete; its authoritative SHA is recorded in Git history and the Phase
  0 handoff.

The verification ran in a separate CPU-only `qwen-receipt-phase0`
environment with NumPy 2.4.6, Pandas 3.0.6, and PyArrow 25.0.1 because the
original `paperwork` Conda environment was not discoverable from the owner's
CARC account. This result validates the Phase 0 data contract and evidence; it
does not configure or validate the complete baseline/GPU runtime needed by
later phases.

At Phase 0 closure, no receipt stage, attack config, GPU job, or attack output
was part of the evidence. Phase 1 tasks 1-6 subsequently added the receipt code
and configs described above; no GPU job or attack output has yet been run or
accepted.
