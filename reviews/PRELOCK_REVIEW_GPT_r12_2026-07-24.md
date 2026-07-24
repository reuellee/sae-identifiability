# Round-12 pre-lock review — GPT-5.x (codex exec), 2026-07-24 — verdict REVISE

Independent second reviewer (headless GPT via `codex exec`), pointed at the pushed
public repo, NOT primed with Gemini's findings. Returned 12 findings. Dispositions:

| # | Sev | Finding | Disposition |
|---|-----|---------|-------------|
| 1 | CRIT | P1 conflates absorption with feature LOSS — never checks the letter survives in the SAE reconstruction | **FIX**: absorbed = letter-present(x) ∧ letter-retained(x̂) ∧ main-L-latent misses; report `loss_rate` separately |
| 2 | CRIT | Frozen scorer doesn't enforce frozen params (θ, sel_min, k, n_carriers, model/layer, λ) | **FIX**: scorer asserts each _fl.json's config == registered; refuse CONFIRM otherwise; record MIN_WORDS/PROBE_C |
| 3 | CRIT | No enforcement of 5 unique seeds/arm | **FIX**: scorer checks exact registered seed set per arm |
| 4 | CRIT | matched-L0 gate tests each mean in a wide band, not that the arms MATCH (L1=28 vs TopK=36 passes) | **FIX**: gate on |ΔL0| small AND near target; drop auto-widening (widening is a lock-time amendment) |
| 5 | CRIT | matched-letter safeguard uses majority not intersection, different denominators, and never revises P1 | **FIX**: true intersection (clean in EVERY SAE both arms); gate P1 on the sign holding there |
| 6 | HIGH | main-latent selection + miss estimation double-dip the same words (winner's curse) | **FIX**: discovery/estimation word split — pick latent+eligibility on discovery half, estimate on holdout half |
| 7 | HIGH | Independent (not paired) bootstrap; 5 seeds → sign-test floor 2/32=0.0625 > 0.05 | **FIX**: paired per-seed bootstrap; **bump to 8 seeds/arm** (16 SAEs; init is shared per seed so pairing is valid) |
| 8 | HIGH | P3 omits opportunity baseline + precision (a detector touching most latents gets free recall) | **FIX**: report recall vs involved-fraction baseline (enrichment) + precision |
| 9 | HIGH | P2 probe direction not out-of-fold for the attributed word | **FIX**: per-word held-out-fold direction+intercept (same refactor as #1) |
| 10 | MED | P2 tail control structurally favorable; dominant components generically align | **FIX**: label-permuted (double-difference) null vs other letters |
| 11 | MED | P2 pooled across arches, prereg says by-arch | **FIX**: report P2 per arch |
| 12 | MED | "balanced positives/negatives" not literal (class_weight, not resampling) | **FIX**: reword prereg to match code |

All accepted. #1,#3,#4,#5,#7 are about the PRIMARY manufacturing confirmation — non-negotiable.
Rewrite: word-level OOF probe (per-word held-out dir+intercept), discovery/estimation split,
loss-vs-absorption, config+seed conformance gate, ΔL0 gate, true-intersection P1 gate, paired
bootstrap @8 seeds, P3 baseline+precision, P2 permuted null + by-arch. Then re-review + lock.

## Targeted re-verification (GPT via codex, 2026-07-24)

After fixing #1/#4/#5, a TIGHTLY-SCOPED re-check (verify these three only; no
open-ended new-findings pass, per advisor guidance to avoid the review treadmill):
- #1 loss-vs-absorption: YES — Xhat mapped to raw space, full-fit retention probe,
  absorbed = present ∧ retained ∧ latent-miss; loss_rate separate. "Feature loss
  cannot count as absorption."
- #4 matched-L0 gate: YES — gates on |ΔL0|≤3 AND both in band; L1=28/TopK=36 now
  FAILS (|Δ|=8); no auto-widening.
- #5 matched-letter: YES — true intersection across every SAE; P1 confirmation
  gated on the L1>TopK sign holding on common letters.
Verdict: **#1/#4/#5 CORRECTLY FIXED.** Round 12 LOCKED at commit 0722212.
