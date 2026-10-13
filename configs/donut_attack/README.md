# Donut attack configuration

Configuration inventory:

- `smoke_khalid.json`: two-document implementation check retained for debugging;
- `smoke_validation.json`: the required 10-SROIE + 10-CORD acceptance smoke;
- `active.json`: the four-epsilon validation sweep, deliberately committed with
  `ready_for_production=false`; and
- `example_validation.json`: a non-runnable template for later variants.

Every runnable config pins the accepted baseline prediction hash, baseline
config hash, merge manifest, and inference Git commit. The stage stops on any
provenance or clean-prediction drift.

The implemented attack supports two targeted teacher-forced losses:

- `paper_margin`: the target-token versus top-logit margin from Equation 10 of
  *Counterfeit Answers*. This is the reproduction default.
- `cross_entropy`: ordinary target-token cross-entropy for comparison with the
  original project plan.

Both modes optimize the original RGB image, project in 0--255 pixel space,
quantize every iteration when configured, freeze all Donut parameters, and
verify success only after lossless PNG save/reload.

## Required GPU validation

1. Run the CPU preflight for `smoke_validation.json`.
2. Submit `scripts/slurm/donut_smoke.sbatch` from the CARC login node.
3. Inspect `environment.json`, especially processor parity.
4. Confirm a nonzero input gradient, no model gradients, the requested L-inf
   bound, optimization traces, and PNG reload predictions.
5. Interrupt the job after a checkpoint once, resubmit unchanged, and confirm
   `checkpoint_resumed=true` with a continuous trace.
6. Review five clean/adversarial/magnified-difference triplets before expanding
   the epsilon grid or running all five worker shards.
7. Change `ready_for_production` to `true`, commit, and only then submit the
   five production workers.

Commands are listed in `docs/imm_next.md` and `docs/05_donut_attack.md`.

The paper's Donut full-document setting is `epsilon=32`, `step_size=2`, and
`max_steps=100`. The project's planned receipt sweep is epsilon 2, 4, 8, and
16 with step sizes 1, 1, 2, and 2 respectively. These must be recorded as
separate run configurations rather than blended into one claimed reproduction.
