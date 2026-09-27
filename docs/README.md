# Experiment implementation plans

This directory is the implementation contract for the rest of the project.
Each document describes one notebook, its inputs, outputs, models, metrics,
checkpoint behavior, five-worker CARC execution, and completion criteria.

Read [00 - Execution contract](00_execution_contract.md) before implementing
any notebook. Its rules apply to every stage.

## Notebook roadmap

| Notebook | Main purpose | GPU on CARC? | Depends on |
|---|---|---:|---|
| [04 - Clean baselines](04_clean_baselines.md) | Freeze prompts and measure clean accuracy | Yes | 03 |
| [05 - Donut attack](05_donut_attack.md) | Reproduce and validate targeted PGD | Yes | 04 |
| [06 - Qwen attacks](06_qwen_attacks.md) | Main full-page targeted attack study | Yes | 04, 05 |
| [07 - Transfer and digital robustness](07_transfer_and_digital_robustness.md) | Test InternVL transfer and common transformations | Yes | 06 |
| [08 - EOT patch and physical pilot](08_eot_patch_and_physical_pilot.md) | Optimize robust patches and test printed examples | Yes, plus manual physical work | 07 |
| [09 - Defenses and adaptive attacks](09_defenses_and_adaptive_attacks.md) | Evaluate three defenses and attack them adaptively | Yes; detector fitting alone is CPU | 06-08 |
| [10 - Results and figures](10_results_and_figures.md) | Merge, verify, analyze, and export final results | No | 04-09 |

## Dependency flow

```text
01 download -> 02 audit -> 03 preprocess
                         |
                         v
                 04 clean baselines
                    /          \
                   v            v
          05 Donut validation   06 Qwen attacks
                                      |
                                      v
                         07 transfer + robustness
                                      |
                                      v
                         08 EOT patch + physical pilot
                                      |
                                      v
                         09 defenses + adaptive attacks
                                      |
                                      v
                         10 final tables + figures
```

## What “done” means

A notebook is not complete merely because its cells ran once. It is complete
only when:

1. Its frozen configuration and Git commit are recorded.
2. All five worker shards finish or the run explicitly uses a documented
   smaller worker count.
3. Interrupted jobs resume without repeating completed documents.
4. The merge step finds no missing or duplicate experiment keys.
5. Expected counts and metric ranges pass assertions.
6. The notebook creates the outputs and summary described in its plan.
7. A second person can reproduce the run from the saved configuration.

## Scope and contingency order

The proposal makes receipt attacks the primary study and resume attacks the
secondary study. If time or GPU access is limited, preserve work in this order:

1. Clean Qwen receipt baseline.
2. Donut attack reproduction.
3. Qwen receipt PGD at all four epsilon values.
4. Saved-image verification, transfer, and digital robustness.
5. Three defenses and adaptive evaluation.
6. Resume case study.
7. EOT patch and physical pilot extensions.

Do not quietly drop an experiment. Record any scope reduction and its reason in
the final report.
