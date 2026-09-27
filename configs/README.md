# Experiment configurations

Every GPU stage reads one tracked JSON configuration. The stage owner creates:

```text
configs/<stage>/<descriptive_version>.json
configs/<stage>/active.json
```

`active.json` is the reviewed configuration all five teammates run. It must at
least contain:

```json
{
  "run_id": "descriptive_unique_run_id",
  "frozen_for_test": false,
  "seed": 566
}
```

The complete stage-specific fields are defined in `docs/`. Never change an
active configuration while workers are running. To revise settings, create a
new config and a new `run_id`.
