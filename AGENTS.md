# Required context for coding agents

Before editing this repository, read these files completely:

1. `README.md` for the scientific question, threat model, completed results,
   attack/defense definitions, and remaining work.
2. `docs/RESULTS_LEDGER.md` for the verified result values, exact run IDs,
   completion state, evidence paths, and results that are still interim.
3. `docs/CARC_END_TO_END.md` for the operational contract, five-worker mapping,
   L40S-only policy, immutable runtime configurations, submission commands,
   checkpoint/restart behavior, and completion checks.

Do not change an experiment denominator, prompt, target policy, model revision,
epsilon grid, defense setting, output schema, or source path without updating
the corresponding frozen configuration, tests, runbook, and analysis notebook.

All production GPU jobs use L40S. Do not add requests, documentation, or
fallbacks for another GPU type. Do not delete partial output shards: restart
the identical worker and runtime config so checkpoints and completed records
are reused.

The root README is scientific documentation. Keep terminal and CARC commands in
`docs/CARC_END_TO_END.md` rather than adding operational instructions to the
root README.
