# Donut attack configuration

`example_validation.json` is a template, not an active experiment. It cannot
run until `baseline_predictions` points at a completed, frozen Donut receipt
baseline. Do not copy it to `active.json` until the stage owner has completed
the GPU checks below.

The implemented attack supports two targeted teacher-forced losses:

- `paper_margin`: the target-token versus top-logit margin from Equation 10 of
  *Counterfeit Answers*. This is the reproduction default.
- `cross_entropy`: ordinary target-token cross-entropy for comparison with the
  original project plan.

Both modes optimize the original RGB image, project in 0--255 pixel space,
quantize every iteration when configured, freeze all Donut parameters, and
verify success only after lossless PNG save/reload.

## Required GPU validation

1. Point the template at baseline predictions and retain the 10-per-dataset
   smoke selection.
2. Copy it to `active.json` with a unique `run_id`.
3. Request an interactive CARC L40S and activate the `paperwork` environment.
4. Run one worker and inspect `environment.json`, especially processor parity.
5. Confirm a nonzero input gradient, no model gradients, the requested L-inf
   bound, optimization traces, and PNG reload predictions.
6. Interrupt one experiment at a checkpoint and confirm the resumed result
   matches an uninterrupted run.
7. Review five clean/adversarial/magnified-difference triplets before expanding
   the epsilon grid or running all five worker shards.

The paper's Donut full-document setting is `epsilon=32`, `step_size=2`, and
`max_steps=100`. The project's planned receipt sweep is epsilon 2, 4, 8, and
16 with step sizes 1, 1, 2, and 2 respectively. These must be recorded as
separate run configurations rather than blended into one claimed reproduction.
