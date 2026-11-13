# CSEV receipt study, version 1

This is the bounded final study promised in the midterm report, not a result.
The tracked template is `configs/csev/protocol_v1.json`. Operational commands live
in `CARC_END_TO_END.md` section 10. Historical protocol stages are unchanged.

## Hypothesis and origin

Counterfactual spatial evidence verification asks whether a full-image answer
is supported by a model-localized total region. It is motivated by visual
contrastive decoding (Leng et al., CVPR 2024, https://arxiv.org/abs/2311.16922)
and selective prediction (Geifman and El-Yaniv, 2017,
https://arxiv.org/abs/1705.08500). The local masking/acceptance rule is the project's
hypothesis, not a reproduced result or a proof of novel/adversarial robustness.

The pinned Qwen model predicts the amount and separately generates one JSON box
in 0--1000 image coordinates. No annotated target, correct answer or attack target
enters inference or localization. The box must be valid, at least 28 pixels on
each dimension after expansion, and occupy at most 35% of the page. A deterministic
5x5 grid supplies the farthest nonoverlapping equal-size control box. A control
box is geometrically unrelated, not guaranteed to contain no relevant evidence.
Missing boxes, invalid answers and crop disagreement are rejection reasons;
model/OCR exceptions are job errors, never scientific abstentions.

The full and cropped predictions must agree after the original task normalization.
Masking uses RGB 128. The score is mean log probability of normalized answer tokens
(excluding EOS), under the same task prompt. D = score(control masked) minus
score(evidence masked). CSEV accepts when crop agreement and a valid control exist
and D >= tau. Repeated totals, localization errors, masking distribution shift,
and attacks within the crop are explicit limitations.

## Frozen cohorts and comparison

Seed 566, SHA256 ordering: 30 CORD + 30 SROIE development; 60 collected evaluation
receipts chosen before eligibility filtering; 20 Qwen-baseline-correct evaluation
receipts for the adaptive pilot. Public and collected domains remain separate.
Prior aggregate collected findings informed the design; it is not a wholly unseen
test set. Counts/targets/PNG bounds and hashes are frozen before GPU execution.

Development selects the largest nonnegative observed gap threshold meeting >=80%
coverage and selective accuracy >= full-image accuracy minus 0.02. If no threshold
meets both, record goal_met=false and use tau=0 without changing the criteria.
Do not tune on external attack outcomes. Curves across external scores are
post-hoc descriptions, not alternative selected deployment thresholds.

Nine conditions: none, JPEG, resize_vote, selective_multiview, semantic_consistency,
OCR_only, old full evidence, new-localizer crop_only, CSEV. Old baseline transforms
and OCR rules retain their existing configuration. Crop-only uses the *new* Qwen
region so it isolates the effect of masking; it is distinct from historical
OCR-localized full+crop. Full/crop computations are shared, so independent latency
for those two ablations is deliberately null rather than fabricated.

Evaluation inputs: 60 clean + 9M saved attacks for M originally correct selected
receipts. Three families (standard, uniform EOT, hardview), eps 4/8/16. Preserve
family/budget denominators. Paired bootstrap resamples documents within each
family/budget; it does not pool correlated perturbations as independent samples.

## Direct-adaptive study

20 documents x eps 4/8 x two restarts = 80 attempts and 40 final trials. Each restart
uses 100 gradient steps and 100 actual-verifier queries, including final reopened
PNG verification. Gradient objective minimizes target CE(full) + target CE(crop)
+ max(0,tau-D_target). Exact-forward processor BPDA is reused. Localization is
refreshed from current pixels at actual queries every ten steps; coordinates are
fixed inside a gradient step. Remaining queries use bounded square proposals.
Actual-pipeline verification, not surrogate loss, selects the candidate.

Random state, pixels, best candidate, iteration and query count are checkpointed
after each step/query. Two restarts are reduced by accepted target success then
full target/crop/gap/target score, never by ground truth recovery. The selected
image is reloaded, hash checked and measured for L-infinity, SSIM and LPIPS.
Replay compares all conditions on this same chosen image.

Report accepted-target ASR, clean utility, coverage, selective accuracy, paired
intervals, costs, traces and induced rejection among clean-accepted documents.
Encountered rejection is a diagnostic from targeted search, not an independently
optimized availability study. Discrete localization and finite search can limit
attack strength; low empirical ASR does not certify robustness.

## Provenance and completion

New immutable run IDs, frozen Git commit/package versions/OCR binaries, permanent
selected input copies, five deterministic shards, output locks, exact expected
record sets, and strict merged artifacts. No external GPU result exists from local
CPU tests. Production completion and method efficacy must be established on CARC.
