# Baseline configurations

- `smoke.json`: owner-only interactive validation smoke test. It selects two
  SROIE receipts, two CORD receipts, and two resumes and repeats every model
  prediction. Run it with `python scripts/smoke_baseline.py` on one L40S.
- `active.json`: full five-worker validation run. It selects 301 documents and
  requires exactly 527 prediction records at merge time.

Qwen runs on all three datasets. Donut runs only on SROIE and CORD. Both models
use the same receipt prompt. Every model revision, prompt, processor setting,
seed, and precision is tracked in the JSON and copied into the output folder.

Never change one of these files while its `run_id` is queued or running. Create
a new file and a new `run_id` for any revised experiment. A future test config
must set `split` to `test`, set `frozen_for_test` to `true`, and name the
accepted `validation_run_id`; the expected merged size is 997 records.
