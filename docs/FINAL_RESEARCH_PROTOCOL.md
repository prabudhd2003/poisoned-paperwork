# Final research protocol, version 1

Status on 2026-11-09: implemented candidate and experiment machinery, not an
established robust defense. No new GPU outcomes have been observed. Existing
queued protocols and results are unchanged. Entry commit was `35dd6c1`.

## Research question and proposed contribution

Does requiring an extracted answer to agree with independently recognized,
localized document evidence improve the trade-off between accepted targeted
errors, correct answers, and abstention under adaptive attack?

Evidence-grounded selective inference uses Tesseract OCR, conservative task
rules, and the evaluated Qwen or Donut model. Receipt rules seek explicitly
labeled totals and reject subtotal, tax, tender, cash, change, savings, and
ambiguous amounts. Resume rules inspect every page, look for dated education
or explicit completion statements, and reject uncertain degree mentions.
Evidence coordinates come only from OCR of the input pixels. The model reads
the whole document and the evidence regions; the full method releases an
answer only when these readings and the OCR-derived answer agree.

The inference interface takes pixels, task identity, a frozen prompt and
settings. It cannot receive the ground truth, attacker target, source record,
or annotation coordinates. Evaluation attaches labels after inference.

The hypothesis is falsified if the method does not improve adaptive accepted
ASR at useful clean coverage relative to OCR-only and the strongest existing
baselines. Lower ASR achieved only by increased abstention is not recovery.
A useful contribution may be a negative result exposing correlated evidence
failures, not a claim of a new universally effective algorithm.

## Primary-source review

- [Counterfeit Answers](https://arxiv.org/html/2512.04554v1), Pintore et al.,
  studies targeted and untargeted adversarial forgery against OCR-free DocVQA,
  including Donut and Pix2Struct. Targeted document attacks are prior work;
  our contribution cannot be the existence of such attacks.
- [PDA](https://arxiv.org/html/2604.01010v1), Xu, Luo and Liu, is
  *PDA: Text-Augmented Defense Framework for Robust Vision-Language Models
  against Adversarial Image Attacks*. It combines paraphrasing, decomposition,
  and aggregation. The existing five-prompt ensemble is a baseline, not a novel
  defense claim. Our implementation is not a reproduction of PDA.
- [MirrorCheck, v5](https://arxiv.org/html/2406.09250v5) regenerates images from
  model outputs and compares image embeddings. Our field-level OCR evidence
  check differs from that regeneration mechanism. Neither mechanism's claims
  establish exact receipt extraction robustness here. We have not reproduced
  MirrorCheck or claimed superiority over it.
- [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) and earlier
  selective classification work establish the rejection/coverage formulation.
  Our rule-based rejection mechanism is not new selective-prediction theory,
  calibrated risk control, or a certified bound.
- [Obfuscated Gradients](https://proceedings.mlr.press/v80/athalye18a.html)
  motivates exact-forward preprocessing and explicit tests for unusable
  gradients. Failure to differentiate through OCR is not evidence of safety.
- [On Adaptive Attacks](https://papers.nips.cc/paper/2020/file/11f38f8ecd71867b42433548d1078e38-Paper.pdf)
  motivates attacking the actual decision rule, inspecting optimizer behavior,
  and avoiding claims from a single ineffective surrogate.
- [ASTRA](https://arxiv.org/abs/2411.16721) concerns activation steering against
  VLM jailbreaks. Harmful-response suppression is distinct from exact-field
  document extraction and is not a matched quantitative baseline here.
- The [InternVL model card](https://huggingface.co/OpenGVLab/InternVL2_5-4B)
  documents image tiling, thumbnails, multi-image patch counts and remote-code
  chat inference. We inspected the pinned remote `modeling_internvl_chat.py`.
  The card's transformers minimum does not establish compatibility with CARC's
  transformers 5.17.0. Actual loading and multi-page inference remain smoke gates.
- [Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage)
  defines TSV word confidence and coordinates. This is a separate OCR pathway,
  not an oracle or an independent guarantee of document semantics.

This focused review supports a testable contribution direction, not an
exhaustive novelty determination. Related OCR-plus-VQA pipelines and selective
systems must be acknowledged in the final report. No empirical ranking against
PDA, MirrorCheck, or ASTRA is claimed without matched reproductions.

## Frozen study design

Full public clean cohorts are 100 CORD, 126 SROIE and 75 resumes. Donut excludes
resumes. The external clean cohort is all 114 collected receipts, including
baseline errors. Defense replay and adaptive experiments use epsilon 4/8/16.
Transfer retains 2/4/8/16 for standard attacks and 4/8/16 for adaptive attacks.
Every provided source family must cover its entire baseline-clean-correct
cohort. A partial EOT snapshot cannot silently become a final full source.
Public and collected runs have distinct IDs and never share denominators.

The nine conditions are no defense, JPEG 85, resize vote, visual consistency,
semantic consistency, OCR-only, full+OCR, full+crop, and the full evidence method.
All run on identical documents within a source/dataset/epsilon stratum. OCR-only
is an independent baseline; the two partial agreement methods are ablations.
Full+crop still uses OCR localization but removes the OCR-answer agreement gate.
These are component comparisons, not equal-cost methods; cost is reported.

The confidence cutoff of 60 and crop padding of at least 16 pixels are declared
engineering choices, not tuned or calibrated thresholds. Do not choose them on
external outcomes. Any later tuning requires a separate public development
split, a new protocol version/run ID, and a documented locked choice before
external evaluation. Version 1 does not claim coverage guarantees.

Report full-cohort clean accuracy, coverage, selective accuracy/risk, abstention,
accepted targeted ASR over all eligible attempts, and targeted rate among
accepted outputs. Correct recovery means the clean answer is released on an
originally successful attack. Rejection and other wrong answers are separate.
Transfer reports both source-eligible and source-and-target-clean-correct
counts. Target clean correctness is measured on the original, untransformed
clean document, independently of each transformed attack evaluation. Each
transform is separate, including direct PNG transfer. Matched source-model
intersections must accompany comparisons between Qwen and Donut attacks.

Confidence intervals use document-paired bootstrap within a single dataset,
model, source family, epsilon and example-type stratum. Do not pool repeated
transforms or epsilon values as independent documents. Zero accepted outputs
have undefined selective accuracy/risk, not perfect accuracy.

## Adaptive evaluation

The targeted adaptive protocol uses 100 white-box PGD steps against full-page
and dynamically localized crop target losses. OCR is rerun every step; crop
coordinates are constant only during that gradient step. Official-forward
preprocessing parity is checked, and zero/nonfinite gradients fail the run.
The optimizer uses cross-entropy and a step of epsilon/4 with projection and
8-bit quantization. It does not differentiate OCR recognition.

A subsequent 2,000-query square-style random search evaluates the actual
pipeline, including new OCR, localization, cropping and abstention on every
candidate. Seeds include clean pixels, the new white-box endpoint and every
supplied standard/EOT/hard-view image at the same epsilon. Search ranks accepted
target success first, then the number of target-matching components. Restarts
occur every 200 queries; budget traces are saved at 100/500/1000/2000. It stops
on exact success. A separate availability objective searches for abstention;
report it only relative to clean accepted coverage, since initially rejected
inputs are not newly induced denials.

Final PNGs are reopened, checked for bounds, and evaluated again. LPIPS/SSIM
and per-page L-infinity values are recorded. Both phases checkpoint after each
step/query. CPU tests cover deterministic interruption/resume, not model
optimization efficacy. Checkpoint files are trusted local restart state and
must never be loaded from an untrusted downloaded artifact.

This is a hybrid surrogate-plus-query attack, not a complete white-box attack
on OCR, not the published Square Attack loss, and not a robustness certificate.
Discrete feedback may plateau. The white-box seed uses its final iterate, not
an exact-defense best-iterate oracle. Audit loss curves, budget saturation,
multiple seeds/restarts and per-component failures. Increase budgets in new
run IDs if ASR is still rising. A low measured ASR remains an empirical lower
bound on attacker capability. Replay the resulting PNGs through all nine
conditions using a separate defense run to compare them fairly.

## Scientific and engineering limitations

Receipt rules currently require the label and amount on one OCR line. Layouts
with split labels, multilingual labels beyond the rule vocabulary, OCR errors,
and unusual formatting may abstain. Resume rules are conservative and do not
solve arbitrary language, implicit credentials, or temporal reasoning. The
existing task says highest degree *stated*, while evidence rules require
support for completion. Measure the resulting utility loss; do not silently
change benchmark labels or claim a matched earned-degree task.

OCR weights and executable hashes are frozen. System libraries, OCR behavior,
remote model dependencies, large resumes, repeated region crops, and peak GPU
memory still need real CARC validation. Transfer predictions never enter
source attack generation or tuning. InternVL remains evaluation-only.

The actual GPU experiments, exact current Donut defense values, queued job
status and permission repair are unavailable on this Mac. No new ASR, efficacy,
or GPU compatibility result is claimed. A successful CPU preflight cannot
substitute for inference, gradient, restart and merge checks on an L40S.
