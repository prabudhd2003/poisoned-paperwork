# Results

Every number in the README and on the poster, with its denominator and where it comes from.
C / W / R = correct and accepted, wrong but accepted, rejected (sent to a person).

Two kinds of evidence back these numbers:

- **CARC pipeline runs** (`src/`, `scripts/`, `configs/`) on L40S GPUs: clean baselines, standard
  attacks and EOT attacks. Their merged tables live in the CARC `outputs/` tree, which is not in Git.
- **Colab studies** (`notebooks/04`–`09`) on A100 GPUs: CBVR, the attacks aimed at it,
  resumes and Donut. Their summary files are committed under [`results/`](../results).

## 1. Clean baselines (CARC)

| Model | Dataset | Correct / total | Accuracy |
|---|---|---:|---:|
| Qwen2.5-VL-3B | CORD | 96 / 100 | 96.0% |
| Qwen2.5-VL-3B | SROIE | 123 / 126 | 97.6% |
| Qwen2.5-VL-3B | Resumes | 74 / 75 | 98.7% |
| Qwen2.5-VL-3B | Self-collected receipts | 110 / 114 | 96.5% |
| Donut | CORD | 75 / 100 | 75.0% |
| Donut | SROIE | 46 / 126 | 36.5% |
| Donut | Self-collected receipts | 90 / 114 | 78.9% |

## 2. Standard targeted attacks (CARC)

White-box PGD; success means the saved, reloaded PNG returns the attacker's target
(10× the true total, or the next degree up). Rates are over clean-correct documents.

| Model | Dataset | ε = 2 | ε = 4 | ε = 8 | ε = 16 |
|---|---|---:|---:|---:|---:|
| Qwen | CORD | 0.0% | 62.5% | 87.5% | 89.6% |
| Qwen | SROIE | 0.0% | 37.4% | 67.5% | 91.1% |
| Qwen | Self-collected (110) | 0.0% | 70.9% | 88.2% | 99.1% |
| Qwen | Resumes | 0.0% | 1.4% | 5.4% | 23.0% |
| Donut | CORD | 8.0% | 16.0% | 50.7% | 93.3% |
| Donut | SROIE | 2.2% | 10.9% | 26.1% | 63.0% |

Median SSIM on self-collected receipts: 0.994, 0.939, 0.802, 0.539 at ε = 2, 4, 8, 16.

## 3. Image-cleaning defenses vs. a plain attack (CARC, Qwen, public receipts and resumes)

4,688 records over 293 documents. Attack success rate (lower is better):

| Defense | Clean accuracy | ε = 4 | ε = 8 | ε = 16 |
|---|---:|---:|---:|---:|
| No defense | 100.0% | 36.5% | 58.4% | 73.4% |
| JPEG quality 85 | 99.7% | 4.4% | 15.7% | 41.0% |
| Resize vote | 100.0% | 11.6% | 22.5% | 39.9% |
| Visual multi-view | 100.0% | 23.9% | 38.2% | 54.6% |
| Semantic consistency | 99.7% | 15.4% | 31.1% | 44.4% |

EOT attacks optimize through these transformations. On CORD at ε = 16, uniform
JPEG-aware EOT fools all four defenses at once on 87.5% of documents and frequency
hard-view EOT on 66.7%. Section 4 evaluates the same two attacks on self-collected receipts.

## 4. CBVR on self-collected receipts (Colab, Qwen) — main result

Files: [`results/receipts_qwen/`](../results/receipts_qwen). Study `receipt_bands_full_v1`,
contract `94243896…`, runner `scripts/colab_receipt_study.py`, code snapshot commit `64aa87b`.

Protocol: 75 development receipts (60 public + 15 collected) for design; frozen;
then evaluated once on 45 reserved collected receipts and their 396 saved attacks
(standard, uniform EOT, hard-view EOT × ε 4/8/16; 132 per family).

**396 attacked images**

| Method | C | W | R | Attacker's total |
|---|---:|---:|---:|---:|
| No defense | 64 | 332 | 0 | 313 |
| JPEG | 115 | 281 | 0 | 260 |
| Resize vote | 94 | 302 | 0 | 286 |
| Vote over views | 76 | 313 | 7 | 299 |
| Re-worded questions | 78 | 282 | 36 | 272 |
| OCR only | 106 | 3 | 287 | 0 |
| Model + OCR + crop agree | 22 | 0 | 374 | 0 |
| Fixed crop | 313 | 83 | 0 | 0 |
| One located band | 356 | 28 | 12 | 0 |
| Full page and band agree | 60 | 0 | 336 | 0 |
| Masked-band check | 49 | 0 | 347 | 0 |
| **CBVR** | **354** | **7** | **35** | **0** |

**45 clean receipts:** no defense 44/1/0, CBVR 39/1/5.

**By attack family (132 each), correct with no defense → CBVR:** standard 15 → 115,
uniform EOT 19 → 121, hard-view EOT 30 → 118.

**CBVR's 42 misses:** 35 rejected (21 with one band correct, 12 with no usable band,
2 with both bands wrong) and 7 accepted errors, none of them the attacker's total
(for example 71.33 read as 7133.00).

## 5. Attacks aimed at CBVR (Colab, Qwen)

Files: [`results/receipts_qwen_direct_attack/`](../results/receipts_qwen_direct_attack)
(one row per run in `direct_attack_runs.csv`; the three schedule files record which
receipts were run in each part).

Each run: 100 gradient steps on both bands plus 100 queries to the real CBVR
pipeline; success = CBVR accepts the exact 10× total on the saved PNG.
20 receipts in a fixed hash order (parts 1–2: the first ten; part 3: the next ten
that CBVR answers correctly when clean). One receipt (c00032) was rejected when
clean, so it is not eligible: **19 eligible receipts, 76 runs.**

| Budget | Receipts broken (any try) | 95% Wilson interval | Runs | No defense, ordinary attack |
|---|---:|---:|---:|---:|
| ε = 4 | 4 / 19 | 9–43% | 6 / 38 | 15 / 19 |
| ε = 8 | 8 / 19 | 23–64% | 12 / 38 | 17 / 19 |

Broken at least once: c00060, c00069, c00071, c00086, c00096, c00103, c00106, c00110.
Every other run returned the true total (no rejections). 11 receipts never broke.
The undefended comparison uses the saved standard attack, a different procedure,
so it is a side-by-side, not a controlled comparison.

## 6. Resumes (Colab, Qwen)

Policy: frozen `resume_development_retry_v4` (education-section bands, one retry on
malformed localization). Files: [`results/resumes_qwen/`](../results/resumes_qwen).

**Reserved evaluation** (`reserved/summary.csv`):

| | No defense | Contextual recovery |
|---|---|---|
| 60 clean resumes | 59 / 1 / 0 | 56 / 2 / 2 |
| 236 saved attacks (ε 2–16) | 214 / 22 / 0 | 224 / 1 / 11 |

**Direct attack on recovery** (`direct_attack/`): first ten reserved resumes with saved
attacks, ε 8 and 16, two tries, 100 steps + 40 queries. **0 of 40 runs** returned the
fake degree (39 correct, 1 rejected). On the same ten, the undefended standard attack
succeeded on 1 / 10 at each budget.

Resumes are hard to attack at all: the target must replace a whole word, and
ordinary attacks reach only 23% at ε = 16 even without a defense.

## 7. Donut as the reader (Colab)

Files: [`results/receipts_donut/`](../results/receipts_donut). Qwen locates the band,
Donut reads it. The 396 attacks were built against Qwen, so this measures transfer.

| | No defense | CBVR |
|---|---|---|
| 45 clean receipts | 38 / 6 / 1 | 31 / 3 / 11 |
| 396 transferred attacks | 344 / 43 / 9 | 276 / 16 / 104 |

The attacks did not transfer (0 attacker totals under any method), so CBVR could only
cost answers here. Direct attacks on the Donut pipeline were planned but not run (0 / 120).

## Not done

- Direct attacks on the comparison methods beyond 5 exploratory runs.
- Direct attacks on the Donut pipeline.
- A third reader model (InternVL): protocol only, no results.
