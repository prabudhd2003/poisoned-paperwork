# Poisoned Paperwork

Poisoned Paperwork studies targeted adversarial attacks and defenses for
document vision-language models (VLMs). The central question is not merely
whether changing pixels can change a prediction. We ask how reliably an
attacker can force a specific wrong answer, how visible that manipulation is,
whether ordinary document processing removes it, and whether those defenses
remain useful after the attacker learns how they work.

The primary task is receipt-total extraction. The secondary task is identifying
the highest degree stated in a resume. Together they test numerical extraction
and semantic document classification across two model families, public data,
and a prospectively collected external dataset.

> **Required reading for contributors and coding agents:** read
> [`AGENTS.md`](AGENTS.md) and
> [`docs/CARC_END_TO_END.md`](docs/CARC_END_TO_END.md) before changing
> experiment code or launching jobs. Cluster commands, worker ownership,
> environment requirements, restart rules, and completion checks deliberately
> live in that separate operational runbook rather than this scientific README.

## Why this project matters

Document VLMs increasingly read receipts, invoices, applications, resumes, and
other records that can influence financial or administrative decisions. An
image can appear normal to a person while causing a model to return an
attacker-chosen total or education category. At the same time, a defense that
simply destroys image quality, rejects every document, or changes clean answers
is not useful.

The project therefore measures an attack–defense trade-off:

1. Can a bounded perturbation force an exact target answer?
2. At what perturbation budget does this become reliable?
3. Does the attacked page remain visually similar to the original?
4. Do JPEG, resizing, multi-view voting, or prompt consistency remove it?
5. Can an informed attacker adapt to those defenses?
6. Do conclusions hold on newly collected receipts and another model?

## Tasks, datasets, and models

### Receipt-total extraction

The model receives a receipt image and must return only its final transaction
total. Public experiments use:

- **SROIE**, with 123 clean-correct Qwen validation documents out of 126;
- **CORD v2**, with 96 clean-correct Qwen validation documents out of 100.

The targeted public-receipt attack asks the model to return a frozen incorrect
amount equal to ten times the canonical clean total. Exact decimal handling is
used so formatting differences do not silently change the target definition.

### Resume highest-degree classification

The model receives every page of a resume and returns one label from a frozen
education taxonomy. Qwen is clean-correct on 74 of 75 validation resumes. The
target is a fixed upward change, such as `bachelor → master` or
`master → doctorate`.

### Independently collected receipts

The team collected 114 new receipt photographs and is manually auditing their
final-total annotations.
They are converted to PNG, audited one at a time, and kept as a separately
labeled external test set. These documents were not selected from the public
benchmark and are the main response to the concern that an open-source VLM may
have memorized public data. Public and collected results are never pooled into
one denominator.

### Models

- **Qwen2.5-VL-3B-Instruct** is the main differentiable VLM and supports the
  full adaptive attack study.
- **Donut** supplies a second document-model architecture with its own clean
  baselines and targeted attacks.
- A separate vision-language model is used for black-box transfer evaluation;
  it never supplies gradients to the source attack.

## Experimental contract

Attacks are evaluated only on documents the relevant model originally answered
correctly. This conditional denominator prevents ordinary model errors from
being counted as successful attacks.

Every saved adversarial artifact is an RGB PNG constrained by an image-space
L-infinity budget measured in ordinary 0–255 pixel levels. The principal
budgets are ε = 2, 4, 8, and 16. The saved PNG is reopened and evaluated again;
success is never inferred only from an in-memory optimization tensor.

The principal metrics are:

- exact targeted attack success rate (ASR), with the numerator and denominator;
- LPIPS and SSIM for perceptual similarity;
- measured L-infinity distance after saving and reloading;
- clean accuracy, defense coverage, abstention, agreement, and entropy;
- runtime, optimization steps, model passes, and peak GPU memory;
- success after each real defense transformation;
- transfer success on a model that did not generate the gradient.

All production runs use frozen configurations, model revisions, prompts,
dataset counts, source hashes, deterministic sharding, checkpoint contracts,
and strict merged-result validation.

## Attack progression

The attacks deliberately form a sequence rather than a collection of unrelated
demos.

### 1. Standard targeted full-page attack

The first attack uses teacher-forced targeted cross-entropy and projected
gradient descent. Qwen's official image processor is used in the forward pass.
A differentiable surrogate with an exact-forward BPDA construction supplies
the backward path, so optimization sees the same processed pixels used by the
model while still receiving image gradients.

After every update, the image is projected into the ε-bounded RGB region and
quantized. Checkpoints retain the lowest-loss and verified successful
candidates. The final result is accepted only after the saved PNG is reloaded
and produces the exact target.

### 2. Uniform multi-view EOT/BPDA attack

The first adaptive attack assumes the attacker knows the completed defenses.
It uses Expectation over Transformation (EOT) and BPDA across:

- identity;
- JPEG qualities 65, 75, 85, and 95;
- fixed and exact per-document randomized resize views;
- Gaussian blur;
- brightness adjustment;
- five semantically equivalent task prompts.

Three frozen views are sampled per optimization step. Real Pillow transforms
run in the forward pass and an identity straight-through approximation supplies
the transformation gradient. Every twenty steps, the saved PNG is tested
against the exact four defense protocols. Candidate selection first maximizes
the number of defenses defeated and then uses target loss as the tie-breaker.

This full run contains all 293 public clean-correct documents at ε = 4, 8, and
16, for 879 adaptive attacks. ε = 2 remains the completed negative control from
the standard attack, where it achieved 0/293 successes.

### 3. Frequency-aware hard-view EOT/BPDA attack

The second adaptive method uses the same documents, budgets, transformations,
prompts, targets, and final defense evaluation, but strengthens the optimizer
in two ways.

**Hard-view weighting.** At each checkpoint, the attack measures target loss
for every visual/prompt view. A temperature-controlled distribution gives more
sampling probability to the highest-loss views while retaining a nonzero floor
for every view. The optimizer therefore concentrates on whichever defense is
currently hardest instead of spending equal work on views it already defeats.

**Frequency-aware updates.** A smooth FFT low-pass component is mixed into both
the gradient direction and the perturbation itself before L-infinity projection
and quantization. JPEG and resizing preferentially suppress fragile
high-frequency noise, so this biases the attack toward spatial structure more
likely to survive those operations while preserving the identical pixel budget.

The method is evaluated separately from uniform EOT. It is considered better
only if it raises defended ASR without an unacceptable LPIPS, SSIM, or runtime
cost. The repository does not claim that either adaptive method is guaranteed
to defeat every document.

## Defenses

All four defenses were run on every one of the 293 public clean-correct
documents at ε = 4, 8, and 16. Clean examples are evaluated as well so a defense
cannot appear effective merely by rejecting or corrupting inputs. The completed
study contains 4,688 records with zero logged failures:

- 293 documents × 4 clean-defense evaluations = 1,172;
- 293 documents × 3 ε values × 4 attack-defense evaluations = 3,516.

### JPEG Q85

The page is recompressed using real JPEG quality 85 with fixed chroma handling,
then decoded and sent to Qwen. This is inexpensive and particularly effective
against non-adaptive high-frequency perturbations, but may be vulnerable once
the attacker optimizes through JPEG.

### Randomized resize voting

Five deterministic, record-specific downsample/upsample views are generated
with scales between 0.70 and 1.00. Qwen predicts each view and the majority
wins. Randomization makes one fixed preprocessing path less predictable, while
the frozen seed keeps the experiment reproducible.

### Selective visual multi-view consistency

The model sees identity, JPEG, resize, blur, and brightness views under the same
prompt. At least three of five normalized answers must agree; otherwise the
system abstains. This tests whether the answer is stable under small ordinary
document transformations.

### Cross-view and cross-prompt semantic consistency

The project-specific ensemble pairs the same five visual transformations with
five semantically equivalent prompts. It releases an answer only with a
three-of-five majority and records agreement, normalized entropy, abstention,
runtime, and added VLM calls. This targets attacks that overfit one exact prompt
or one visual preprocessing path.

These are initially **non-adaptive defenses** because the standard attack did
not know which defense would be applied. Both full adaptive attack runs then
evaluate the same defenses under a stronger white-box threat model.

## Current results

### Clean Qwen performance

| Dataset | Correct | Total | Clean accuracy |
|---|---:|---:|---:|
| CORD v2 | 96 | 100 | 96.0% |
| SROIE | 123 | 126 | 97.6% |
| Resumes | 74 | 75 | 98.7% |

The high clean accuracy makes the robustness evaluation meaningful: nearly all
documents enter the clean-correct attack pool, especially for resumes.

### Standard targeted Qwen attack

| Dataset | ε=2 | ε=4 | ε=8 | ε=16 |
|---|---:|---:|---:|---:|
| CORD v2 | 0.0% | 62.5% | 87.5% | 89.6% |
| SROIE | 0.0% | 37.4% | 67.5% | 91.1% |
| Resumes | 0.0% | 1.4% | 5.4% | 23.0% |

These results show three important patterns.

1. **ε = 2 is a meaningful negative control.** It produced no targeted success
   on any of the 293 documents, so a nonzero perturbation budget alone does not
   guarantee an attack.
2. **The task matters.** Receipt totals are substantially easier to redirect
   than resume degree classification under the same model and budget.
3. **Effectiveness grows with ε.** This creates an explicit robustness–visibility
   curve that must be interpreted together with LPIPS, SSIM, and visual review.

### Existing defenses against the standard attack

| Condition | ε=4 | ε=8 | ε=16 |
|---|---:|---:|---:|
| No defense, saved attack outputs | 36.5% | 58.4% | 73.4% |
| JPEG Q85 | **4.4%** | **15.7%** | 41.0% |
| Resize vote | 11.6% | 22.5% | **39.9%** |
| Selective visual multi-view | 23.9% | 38.2% | 54.6% |
| Semantic consistency | 15.4% | 31.1% | 44.4% |

JPEG is strongest at ε = 4 and 8. Resize voting is slightly stronger at ε =
16. The more elaborate consistency ensembles are useful but do not dominate the
simple preprocessing baselines in this non-adaptive setting. That is a result,
not a failure: complexity alone does not make a defense better.

The table does not yet establish robustness against an informed attacker.
Uniform EOT and frequency-aware hard-view EOT are designed to measure how much
of each reduction survives after the attacker adapts.

## How the results guide the final defense

The project will not invent the final defense before observing the adaptive
failure modes. After both adaptive runs finish, we will compare, for each
dataset and ε:

- defended ASR for all four defenses;
- the strict fraction of PNGs that defeat all four;
- which visual or prompt views remain hardest;
- LPIPS/SSIM and visual quality;
- abstention, clean utility, latency, and model-call cost;
- uniform-EOT versus frequency-hard-view improvement.

The final defense will target the measured weakness. For example, if
low-frequency attacks survive JPEG but disagree across prompts, the design
should emphasize semantic verification. If they remain prompt-consistent but
fail under particular transformations, the defense should select or randomize
those transformations. Any proposed defense must then be evaluated on clean
documents, the original attacks, both adaptive attacks, and the external
receipts. This produces an evidence-driven attack–defense arms race rather than
a defense selected because it sounds sophisticated.

## Response to the proposal feedback

The project directly addresses the three concerns raised at proposal time.

**“The project may be trivial because perturbations should affect the model.”**
We require exact attacker-chosen outputs, bounded and reloaded PNGs, multiple ε
values, perceptual metrics, clean-correct denominators, two tasks, and defended
evaluation. The ε = 2 and resume results show that influence is not automatic.

**“The attack method needs a clearer and more difficult treatment.”** The
method progresses from exact-forward targeted PGD to visual and semantic
EOT/BPDA, exact randomized defense views, hard-view loss weighting, and
frequency-aware optimization. Each addition has a controlled baseline and
frozen contract.

**“Existing attacks and defenses should be examined.”** Standard projected
gradient optimization, BPDA, EOT, preprocessing, voting, abstention, and
consistency are evaluated under matched datasets and metrics. Adaptive attacks
prevent us from reporting a defense only against an attacker that did not know
the defense existed.

**“Open-source VLMs may memorize public data.”** The independently collected
114-receipt external set is never used to select public examples. Transfer to a
second model and the second Donut architecture provide additional evidence that
the findings concern model behavior rather than only one public benchmark.

## Remaining work

1. Run and strictly merge both 879-record adaptive Qwen studies.
2. Compare them using `midterm/10_adaptive_defense_attack.ipynb`.
3. Finish and freeze the 114-receipt external manifest.
4. Run Qwen and Donut clean baselines and attacks on the external set.
5. Complete Donut public attacks and the black-box transfer study.
6. Design one final defense from the observed adaptive failure pattern.
7. Evaluate that defense on clean, standard, adaptive, transfer, and external
   examples before making a final claim.

## Repository map

```text
annotations/   human labels and audit material
configs/       tracked experiment templates and frozen method settings
data/          local datasets and processed manifests; not stored in Git
docs/          operational runbook, project plans, and experiment contracts
midterm/       CPU-only report tables and high-quality figures
notebooks/     data auditing and qualitative result inspection
outputs/       checkpoints, records, images, and strict merges; ignored by Git
scripts/       preparation, validation, submission, status, and merge tools
src/           models, attacks, defenses, stages, and shared contracts
tests/         CPU contract and regression tests
```

For execution and recovery procedures, use
[`docs/CARC_END_TO_END.md`](docs/CARC_END_TO_END.md). For the report-facing
figures and tables, start with [`midterm/README.md`](midterm/README.md).
