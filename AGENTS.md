# Required context for coding agents

Before editing this repository, read:

1. `README.md` for the question, threat model, method and findings.
2. `docs/RESULTS.md` for every verified number, its denominator and its source files.
3. `docs/REPRODUCE.md` for how the CARC pipeline and the Colab studies fit together.
4. `docs/CARC_END_TO_END.md` before launching or changing any CARC job.

Do not change an experiment denominator, prompt, target policy, model revision,
epsilon grid, defense setting, output schema, or source path without updating
the corresponding frozen configuration, tests and documentation.

CARC production jobs use L40S GPUs. The Colab studies in `colab/` ran on A100s;
each runner records a hash contract and refuses to resume into an output folder
whose contract differs. To change a Colab study, start a new study ID instead of
editing a frozen runner. Do not delete partial output shards or checkpoints.

Keep cluster and notebook commands in `docs/` rather than in the root README.
