# Verified results ledger

This is the short, authoritative handoff for project results. Read it before
reconstructing results from chat history. The underlying merged artifacts remain
the source of truth. Values marked interim must not be presented as final.

Last updated: 2026-11-09.

## Clean baselines

| Model | Dataset | Correct | Total | Accuracy |
|---|---|---:|---:|---:|
| Qwen | CORD | 96 | 100 | 96.0% |
| Qwen | SROIE | 123 | 126 | 97.6% |
| Qwen | Resumes | 74 | 75 | 98.7% |
| Qwen | Self-collected receipts | 110 | 114 | 96.5% |
| Donut | CORD | 75 | 100 | 75.0% |
| Donut | SROIE | 46 | 126 | 36.5% |
| Donut | Self-collected receipts | 90 | 114 | 78.9% |

The self-collected baseline is
`outputs/baseline/baseline_collected_external_v1/merged/summary.json`.

## Standard targeted attacks on public documents

All values are conditional attack success rates on clean-correct documents.

| Model | Dataset | epsilon 2 | epsilon 4 | epsilon 8 | epsilon 16 |
|---|---|---:|---:|---:|---:|
| Qwen | CORD | 0.0% | 62.5% | 87.5% | 89.6% |
| Qwen | SROIE | 0.0% | 37.4% | 67.5% | 91.1% |
| Qwen | Resumes | 0.0% | 1.4% | 5.4% | 23.0% |
| Donut | CORD | 8.0% | 16.0% | 50.7% | 93.3% |
| Donut | SROIE | 2.2% | 10.9% | 26.1% | 63.0% |

Primary public-result artifacts:

- Qwen receipts: `outputs/imported/qwen_receipt_validation_v1/merged/attacks_for_defenses.parquet`
- Qwen resumes: `outputs/qwen_attack/qwen_resume_validation_v1/merged/attacks.parquet`
- Donut receipts: `outputs/imported/donut_public_validation/merged/attacks_for_defenses.parquet`
- Donut epsilon 32 is an archived stress test and is not part of the matched
  primary protocol: `outputs/imported/donut_eps32_archive/merged/attacks.parquet`

## Qwen defenses against standard attacks

This completed run contains 4,688 records across 293 documents with no logged
failures. Attack success is lower when a defense is more effective.

| Defense | Clean accuracy | ASR at 4 | ASR at 8 | ASR at 16 |
|---|---:|---:|---:|---:|
| No defense | 100.0% | 36.5% | 58.4% | 73.4% |
| JPEG quality 85 | 99.7% | 4.4% | 15.7% | 41.0% |
| Resize vote | 100.0% | 11.6% | 22.5% | 39.9% |
| Visual multi-view | 100.0% | 23.9% | 38.2% | 54.6% |
| Semantic consistency | 99.7% | 15.4% | 31.1% | 44.4% |

Run ID: `defenses_qwen_public_final_v1`.
Summary: `outputs/defenses/defenses_qwen_public_final_v1/merged/summary.json`.

## Donut defenses

Run `defenses_donut_public_final_v1` is complete and merged on CARC with 1,936
records, five worker completion markers, and no failures. The per-defense values
must be read from its merged table before being copied into a report. Do not
infer them from the Qwen defense table.

## Self-collected Qwen attacks

Run ID: `qwen_collected_external_v1`.

Finalized on 2026-11-09 with 440 validated records, 110 unique clean-correct
documents, five complete workers, and zero failures. All records use commit
`8a7b66c`. The repaired merged result is the final source of truth.

| Epsilon | Successes | Attempts | Conditional ASR | Median SSIM | Median LPIPS |
|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 110 | 0.0% | 0.9942 | 0.0079 |
| 4 | 78 | 110 | 70.9% | 0.9392 | 0.1177 |
| 8 | 97 | 110 | 88.2% | 0.8017 | 0.2787 |
| 16 | 109 | 110 | 99.1% | 0.5388 | 0.4513 |

Evidence:
`outputs/qwen_attack/qwen_collected_external_v1/merged/summary.json`

## Adaptive Qwen attacks

These runs are partial and resumable. Their saved records were produced under
commit `97858f607f571453841c609618014a1ba03ace46` and have been approved for exact
reuse through `configs/provenance/jpeg_eot_resume_compatibility.json` after a
scientific-source compatibility check.

| Run ID | Attack | Saved records | State |
|---|---|---:|---|
| `qwen_jpeg_eot_public_full_v1` | Uniform JPEG-aware EOT | 424 | Partial |
| `qwen_frequency_hardview_public_full_v1` | Frequency hard-view EOT | 329 | Partial |

CORD is complete within both partial runs. At epsilon 16, the current CORD
analysis reports that uniform EOT succeeds after JPEG on 91.7% of documents and
under all four defenses on 87.5%. Frequency hard-view reaches 78.1% after JPEG
and 66.7% under all four defenses. These are attack success rates, not defense
accuracies. Resume and SROIE remain incomplete, so the runs as a whole are not
final.

## Operational status

- Do not resubmit `qwen_collected_external_v1`; finalize it on CPU.
- Do not resubmit `defenses_donut_public_final_v1`; it is complete.
- Resume only the two adaptive Qwen runs above using their exact frozen runtime
  configurations.
- Adaptive retries request 12 hours because previous eight-hour workers timed
  out after saving roughly 77 to 92 records each.
- Scratch-backed output links are intentional. Never delete a partial shard.

## Frozen self-collected follow-up protocol

The following runs are configured but are not results until their strict
merged summaries exist.

| Model | Run | Documents | Attacks |
|---|---|---:|---:|
| Qwen | `defenses_qwen_collected_final_v1` | 110 | 330 |
| Qwen | `qwen_collected_jpeg_eot_full_v1` | 110 | 330 |
| Qwen | `qwen_collected_frequency_hardview_full_v1` | 110 | 330 |
| Donut | `defenses_donut_collected_final_v1` | 90 | 270 |
| Donut | `donut_collected_jpeg_eot_full_v1` | 90 | 270 |
| Donut | `donut_collected_frequency_hardview_full_v1` | 90 | 270 |

Each count uses epsilon 4, 8, and 16. Qwen follow-ups use the finalized
440-record standard attack table. Donut follow-ups depend on completion of the
standard Donut attack over its 90 clean-correct self-collected receipts.

## Evidence discipline

Before adding a number to a report, prefer a merged `summary.json` or merged
Parquet table. If only shard records exist, label the value interim and record
the denominator. Update this ledger when a run reaches strict merged completion.

## Final protocol implementation, 2026-11-09

Entry checkout was clean at `35dd6c1270cdb2a271a06a8dbd209a50ba11a5b9`.
New work is local and uncommitted. Nothing was pushed, no CARC connection was
opened, and no job was submitted or changed. No new empirical ASR or clean
accuracy is available. The Donut exact defense table and current queued-job
status still require the read-only CARC snapshot in runbook section 9.

Implemented in the isolated `final_protocol` stage:

- OCR-grounded selective inference, OCR-only, full+OCR and full+crop ablations,
  plus no-defense and all four existing baseline conditions;
- full public/external clean cohorts, source-clean-correct attack eligibility,
  separate dataset/source/family/epsilon denominators, accepted errors,
  abstention, true correct recovery, utility and inference cost;
- adaptive full-page/crop PGD followed by actual-pipeline query search,
  targeted and availability objectives, saved/reloaded bounded PNGs,
  LPIPS/SSIM, per-step/query checkpoints and budget diagnostics;
- full InternVL clean evaluation and direct/transformed transfer, including
  source-eligible and both-models-clean-correct denominators;
- frozen baseline/source/image/runtime/OCR provenance, strict preflight,
  matching L40S smoke gate, scratch definitions, and verified permanent input
  plus result archival tooling;
- analysis notebook `notebooks/12_final_evidence_and_transfer.ipynb`, with
  paired bootstrap differences, matched source intersections, CSV and PNG exports.

Planned standard-only evaluation sizes, not results:

| Run family | Full clean documents | Attack images | Conditions | Evaluation records |
|---|---:|---:|---:|---:|
| Qwen public defense | 301 | 879 | 9 | 10,620 |
| Donut public defense | 226 | 363 | 9 | 5,301 |
| Qwen external defense | 114 | 330 | 9 | 3,996 |
| Donut external defense | 114 | 270 | 9 | 3,456 |
| InternVL public standard transfer | 301 | 1,656 | 11 | 21,527 |
| InternVL external standard transfer | 114 | 800 | 11 | 10,054 |

Adaptive generation plans remain 879/363 public Qwen/Donut and 330/270 external
Qwen/Donut, at epsilon 4/8/16. Each uses 100 surrogate gradient steps and up to
2,000 actual defense queries by default. These can be expensive; measure the
L40S smoke and early shard throughput before estimating aggregate GPU hours.
Do not treat the earlier EOT GPU-hour estimate as a budget for this new search.

Local validation passed:

- 143 CPU tests, including 19 new final-protocol tests;
- synthetic 301-document input freezing, source-grid validation and overwrite
  rejection; five-worker fake-model merge/resume; exact source identity;
- receipt exclusions/conflicts, multipage degree rules, abstention denominators,
  source PNG versus JPEG outcome semantics, bounded query search, deterministic
  interrupted resume, and surrogate gradient updates with a toy objective;
- notebook schema validation and every cell with absent results and synthetic
  results, including paired CSVs and exported PNGs;
- Python compilation, 12 new terminal-block shell syntax checks, launcher shell
  syntax and `git diff --check`.

Test environment was an isolated temporary Mac environment with Python 3.9.6,
torch 2.8.0, torchvision 0.23.0, transformers 4.48.3, pandas 2.3.3, numpy 2.0.2,
and Pillow 11.3.0. It is not the CARC Python 3.11 / torch 2.6 / transformers 5.17
runtime. The first full test attempt had one missing-transformers dependency;
after installing it in the temporary environment, the complete suite passed.
Warnings were library deprecations/LibreSSL and an existing toy-test tensor
conversion warning. No GPU model weights or actual Tesseract binary were run
locally. Synthetic outputs are validation fixtures, not scientific results.

Open gates and limitations:

- Verify current CARC job states, Donut parent permission, full source merges,
  baseline table locations, source counts and project quota.
- Install and hash English/Indonesian OCR data. The actual Tesseract interface
  and its clean coverage remain untested on the real experiment corpus.
- Run separate L40S inference/gradient/multipage smokes for each frozen config.
  InternVL trusted-remote-code compatibility with transformers 5.17 is unresolved.
- Receipt rules require same-line labeled amounts; resume rules are conservative
  and stricter about completion than the existing highest-degree-stated task.
  These can lower coverage and must be reported, not hidden by label changes.
- The adaptive OCR gate is attacked through actual queries, not OCR gradients.
  Discrete search plateaus, final-iterate surrogate seeds and finite budgets
  limit robustness conclusions. Inspect curves and stronger-budget runs.
- The proposed contribution is a falsifiable evidence-agreement evaluation,
  not a claim that OCR, localization, consistency, or abstention is novel.
- No permanent CARC archive has been created by this local work.

Execution order is documented in `docs/CARC_END_TO_END.md` section 9. Detailed
scientific scope and primary-source review are in `docs/FINAL_RESEARCH_PROTOCOL.md`.

## Collected Donut source-path correction

The collected Donut implementation writes `merged/records.parquet`. Corrected
the runbook and three tracked follow-up templates that incorrectly referenced
`merged/attacks.parquet`. Existing runtime configs and CARC runs were not edited.
This correction does not establish that Khalid completed the run; verify current
permissions, job states, five markers, 360 records and zero failures first.

## CARC snapshot supplied by user, 2026-11-09

CARC remains at commit `35dd6c1`, Python 3.11.16 in the paperwork environment.
Khalid's parent permission repair succeeded: `outputs/donut_attack` is mode
2775, and the user's temporary-directory creation probe passed. The project
link `outputs/donut_attack/donut_collected_external_v1` does not yet exist;
there is no collected Donut merged table through that path. The scratch run
was not inspected by this snapshot and must be preserved if it exists.

The shown queue for pkandpal and kbali contains exactly 25 pending Qwen tasks,
all waiting on Priority, across arrays 13871221, 13871227, 13871232, 13871336,
and 13871341. No running or pending Donut job appears in that queue snapshot.
Do not duplicate those Qwen workers or update their checkout.

Public Donut defenses were reconfirmed complete: five markers with shard
counts 400/384/384/384/384, 1,936 records, 121 documents, zero failures, commit
`8a7b66c4f37f169df3050441ba386820132f613a`. Strict merge timestamp was
2026-11-09T13:14:08.668144+00:00. The historical Slurm array 13814788 also has
five COMPLETED workers with exit code 0:0.

Completed-run attack counts printed from the shards (retrieve the merged table
before final report export):

| Dataset | Defense | Successes at 4 | Successes at 8 | Successes at 16 | Attempts per epsilon |
|---|---|---:|---:|---:|---:|
| CORD | JPEG | 6 | 18 | 53 | 75 |
| CORD | Resize vote | 6 | 11 | 38 | 75 |
| CORD | Visual multi-view | 6 | 18 | 53 | 75 |
| CORD | Semantic consistency | 4 | 13 | 38 | 75 |
| SROIE | JPEG | 0 | 4 | 15 | 46 |
| SROIE | Resize vote | 2 | 3 | 11 | 46 |
| SROIE | Visual multi-view | 5 | 5 | 16 | 46 |
| SROIE | Semantic consistency | 0 | 2 | 5 | 46 |

The status display used `adversarial_target_success` for clean rows too. Its
zeros are not zero clean accuracy, and the epsilon labels on those clean rows
come from their representative attack records. Do not report clean accuracy
from that status display.

Both public Donut adaptive project run paths are missing. Historical launch
attempts failed; there are no current Donut adaptive workers in the shown queue.
After verifying the collected scratch link and existing runtime config, the
next production submission is collected Donut standard attacks (90 documents,
360 attacks). Public Donut adaptive sources are independent of that run but
still require preflight and a real adaptive GPU smoke. Collected Donut defenses
and both adaptive methods depend on its strict 360-record standard merge.


## Runtime-only Donut submission bundle prepared (2026-11-09)

Prepared `scripts/ready_donut/queue_ready_donut.py` and
`smoke_ready_donut.py` for copying into ignored
`outputs/runtime_tools/ready_donut_v1/` on CARC at commit `35dd6c1`.
No remote connection, submission, Git push, GPU validation, or new efficacy
result was performed. Five CPU tests pass, including actual launcher dry-runs
for all three job names and explicit L40S/five-worker/afterok contracts.

The bundle checks frozen sources and current queues, preserves partial progress,
and queues missing collected Donut standard attacks (360 records), public
Donut uniform EOT (363), and public Donut frequency hard-view EOT (363).
All production arrays depend on a successful isolated L40S smoke. The smoke
checks saved PNG bounds, hashes, finite metrics, completed-record reuse, and
adaptive checkpoint interruption/resume. Production wrappers verify the
smoke receipt and unchanged code/config/helper hashes before running.
The GPU smoke has NOT been run locally. Existing Qwen jobs are untouched.
Collected Donut defense/EOT followups still require the strict standard merge.


## User-requested code update after cancellation (2026-11-09)

User explicitly chose to cancel pending project jobs and pull all prepared Mac
code into CARC. The local suite passes 149 CPU tests. This does not establish
CARC GPU compatibility. Cancellation and the remote pull have not been observed
by the agent. Preserve runtime configs, partial records, and checkpoints.
Before changing the shared checkout, verify no collaborator worker uses it.
After pulling, run CARC CPU checks and inspect resume compatibility before
resubmission. The runtime-only ready_donut_v1 bundle remains pinned to 35dd6c1;
do not use that old command unchanged after upgrading the checkout. New final
protocol stages remain gated by their actual L40S smoke receipts.
