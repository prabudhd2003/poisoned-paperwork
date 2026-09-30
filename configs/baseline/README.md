# Donut receipt baseline configuration

`example_donut_validation.json` is a four-document smoke-test template. It is
not an active or frozen experiment.

`active.json` is the reviewed 20-document smoke run (10 receipts per dataset),
distributed as four receipts per worker. Commit the code and config before
submitting it. The merged output is:

```text
outputs/baseline/<run_id>/merged/predictions.parquet
```

After checking deterministic predictions, parsing, and GPU memory, create a new
validation run configuration with a new `run_id` and remove
`smoke_per_dataset`. Do not reuse the smoke run ID. The resulting full
validation `predictions.parquet` is the input to `donut_attack`.

This first implementation intentionally covers only the pinned Donut model on
SROIE and CORD receipts. Qwen and resume baselines should be added as separate
reviewed configurations rather than silently changing this run.
