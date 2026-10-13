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
  "stage": "stage_name",
  "mode": "validation",
  "run_id": "descriptive_unique_run_id",
  "ready_for_production": true,
  "frozen_for_test": false,
  "seed": 566
}
```

The complete stage-specific fields are defined in `docs/`. Never change an
active configuration while workers are running. To revise settings, create a
new config and a new `run_id`.

Stages may commit an active config with `ready_for_production=false` while its
smoke is under review. Production code must refuse that config until the owner
changes only the gate, commits it, and tells all workers the accepted commit.
Configs that depend on saved outputs must record and verify those file hashes.
