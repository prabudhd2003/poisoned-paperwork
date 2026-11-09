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
