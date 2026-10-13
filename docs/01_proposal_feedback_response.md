# Proposal feedback and project response

## Feedback received

The proposal received full credit, with three substantive cautions:

1. Simply showing that image perturbations can change a model decision may be
   too predictable to support a strong project by itself.
2. The attack method needs a clearer description because correct targeted
   optimization is the technically difficult part.
3. Open-source vision-language models may have memorized public benchmark
   documents, so results on SROIE or CORD alone may overstate generalization.

The reviewer also suggested studying existing attacks and defenses rather than
reporting only an unconstrained perturbation experiment.

## How the revised project addresses the feedback

### 1. Replace the trivial question with measurable robustness questions

The project no longer asks only whether a perturbation can change an answer.
It measures:

- targeted exact-answer success at pixel-space epsilon 2, 4, 8, and 16;
- conditional attack success among clean-correct documents;
- harmful-output rate over all usable documents;
- save/reload survival, LPIPS, SSIM, runtime, and optimization steps;
- transfer to a separately pinned InternVL model;
- robustness to JPEG, resize, blur, brightness, and translation;
- clean-accuracy versus robustness tradeoffs for three defenses; and
- adaptive attacks that know the complete defended pipeline.

These measurements can distinguish a weak implementation, a digitally fragile
attack, a transferable attack, and a defense that merely obscures gradients.

### 2. Make the attack method explicit and auditable

The threat model modifies the uploaded RGB pixels while model weights remain
frozen. The adversarial objective is teacher-forced target-token loss on a
predeclared harmful answer: ten times the true receipt total, or the next
higher resume degree. Projection and reported epsilon are defined in original
0--255 RGB space, not in normalized model space.

Donut is the smaller white-box reproduction and implementation check. Qwen is
the primary model. Every attack must demonstrate:

- finite nonzero image gradients and no model-parameter updates;
- parity between differentiable and official preprocessing;
- exact pixel-space projection after each step;
- atomic checkpoints and deterministic resume;
- lossless PNG save/reload verification using the official inference path; and
- immutable model, prompt, baseline, config, and Git provenance.

Success is counted from the reloaded PNG, never only from an in-memory tensor.

### 3. Evaluate existing attacks and real defenses

The Donut stage follows the targeted OCR-free document attack formulation from
*Counterfeit Answers*. The Qwen stage applies the same controlled threat model
to a larger instruction-following VLM. Later stages evaluate transfer,
expectation-over-transformation attacks, and three defenses:

1. lossless-to-lossy JPEG re-encoding;
2. randomized resize with majority voting; and
3. detector-guided routing to a stronger verification pipeline.

Each defense is tested against both existing attacks and new adaptive attacks.
The project will not call a defense robust merely because it defeats an attack
that was unaware of the defense.

### 4. Address benchmark memorization with prospective data

Five teammates will collect 20 physical receipts each, for 100 total. Images
are photographed and annotated after the model revisions were published. This
set is kept separate as `external_test` and is never used to select prompts,
step sizes, iteration counts, thresholds, or stopping rules.

The frozen Qwen and Donut baselines will run on all 100 receipts. Frozen attacks
will then run only on the clean-correct subset for the corresponding model.
Results are reported separately from SROIE and CORD.

This prospective set provides evidence about generalization beyond the public
benchmarks. It does **not** prove that a model has never seen a similar merchant
layout, so the final report will describe it as an external generalization test
rather than a complete proof against memorization.

## Experimental claims the project can support

If all gates pass, the project can claim how often bounded targeted attacks
succeed, how their success varies with epsilon and model, whether success
survives serialization and transformations, how much it transfers, and what
clean-accuracy/latency cost accompanies each defense. It can also compare
public-benchmark behavior with a prospectively collected external set.

It cannot claim universal document-model vulnerability, universal defense,
proof of training-data membership, or physical-world robustness without the
separate print-and-photograph protocol.

## Report wording

Use language such as:

> To reduce the risk that public-benchmark familiarity explains the observed
> results, we evaluate the frozen pipeline on a prospectively collected set of
> 100 physical receipts. We do not tune any model, prompt, attack, or defense on
> this set, and we report it separately from SROIE and CORD.

Avoid saying that collecting new receipts “eliminates memorization.”
