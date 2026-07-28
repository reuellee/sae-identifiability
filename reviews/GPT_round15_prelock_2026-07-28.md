# GPT (codex, gpt-5.5) pre-lock review — round 15 (2026-07-28)

Adversarial pre-lock review of the round-15 prereg + harness + evaluator +
driver, run headless via `codex exec` against the pre-lock working tree
(concurrently with `reviews/GEMINI_round15_prelock_2026-07-28.md`). The
original transcript's opening was clipped by the collection pipe; the P1/P2.1–6
findings below are the compact reprint recovered from the same codex session.
Adjudication: **[A]pplied / [R]ejected with reason** — all pre-lock, no
round-15 number existed at any point.

## P1

- **P1.1 — Outcome-dependent clean-letter intersection** (evaluator). Width can
  causally remove letters from the clean set (splitting dilutes selectivity),
  biasing the paired P1 contrast toward survivors. **[A]** Registered D4
  sensitivity: eligibility fixed by the 16k baseline cell, τ-waived rates at
  262k, letter-churn count reported. P1's registered rule stands; divergence is
  informative.
- **P1.2 — Evaluator does not enforce frozen configuration.** **[A]** Gate 0
  added: fail-closed on θ/τ/FAM_CAP mismatch or missing registered cells.
- **P1.3 — Full run can report success after evaluator failure** (driver).
  **[A]** `run_full.sh` now `set -euo pipefail` + verifies both result files
  non-empty before emitting `DONE_ROUND15`.

## P2

- **P2.1 — Layer series scored against layer-13 activations for every layer.**
  Real bug (driver built one words file). **[A]** Words builder extracts all
  four hook layers in one forward pass; scorer groups SAEs by layer with
  per-layer probes.
- **P2.2 — Bootstrap omits word/probe/family-estimation uncertainty.** **[A as
  scope]** Registered limitation added ("What this cannot do"); nested
  resampling is a different design, out of scope.
- **P2.3 — Family discovery and scoring reuse the same words.** **[R]**
  Inherited from the frozen 13a endpoint deliberately — port fidelity is the
  point of a transfer test. Noted in the prereg; cross-fitting is a successor
  design.
- **P2.4 — P3 sign criterion algebraically near-tautological.** Correct — the
  family contains the argmax, so rate_family ≤ rate_single always. **[A]** P3
  re-registered as MATERIAL relative inflation: bar CI-lower > 0.10, floor
  rate_family ≥ 0.005, three-width intersection, ≥ 8 letters.
- **P2.5 — P3 violated the common-letter restriction.** **[A]** folded into the
  P3 redesign (inter3, letter-mean, equal weight).
- **P2.6 — P2 used the cap-censored family size.** **[A]** `fam_size_uncapped`
  + `cap_hit` recorded; P2 tests the uncapped size; scoring family stays capped
  for 13a fidelity.
- **P2.7 — Conformance gate is domain-shifted** (config L0 measured on
  pretraining sequences vs our BOS+word tokens) and was the sole hook oracle.
  **[A]** Pilot adds a sae-lens official-loader oracle (tolerance 1e-3);
  registered caveat: oracle-pass + band-fail = domain shift → re-register the
  band, never silently widen.
- **P2.8 — 262k memory understated.** Partially stale (the `yL==0` complement
  copy was already removed via sum-based sel). **[A]** BATCH default halved to
  1024; revised live-set estimate ~13–15 GB on 32 GB.
- **P2.9 — No revision pinning / promised hashes absent.** **[A]** `dl` stage
  writes `PROVENANCE.json` (resolved HF repo commit SHAs, per-file sha256) and
  pip version freeze; collected with results.
- **P2.10 — Watchdog powers off before collection.** **[A]** Post-exit grace
  raised to 3600 s (poweroff ≠ delete; collect can restart, but should not have
  to).
- **P2.11 — Two unadjusted primaries.** **[A]** P1 sole primary; P2 demoted to
  key secondary; success language registered.

## P3

- **P3.1 — Spearman tie handling.** **[A]** (already fixed from the Gemini
  review): scipy tie-aware spearmanr.
- **P3.2 — THETA recorded but ignored.** **[A]** scorer asserts θ=0 for the
  locked run.
- **P3.3 — P3 emitted an unregistered FALSIFIED-DIRECTION branch.** **[A]**
  covered by the P3 redesign (CONFIRMED / NOT CONFIRMED only).
- **P3.4 — "LOCKED" while untracked.** **[R]** House convention: the lock IS
  the commit adding the file (identical wording to round 14); the commit
  follows this review immediately.
- **P3.5 — setsid PID tracking / watchdog log not collected.** **[A]** PID
  persisted to `~/r15/job.pid`, shown in `status`; watchdog log collected.

## Checks that passed (verbatim from the original run)

- The 262k−16k paired direction and P1/P2 verdict inequalities are implemented
  as registered.
- `boot_ci_letters` preserves all entries belonging to a resampled letter.
- Presence probes are out-of-fold, retention uses the full-fit raw-residual
  probe on each reconstruction, and the present/retained denominator matches
  round 13a.
- `sel >= τ`, argmax selection, `FAM_CAP=32`, and single/family missing-fire
  definitions faithfully port round13a:103–125.
- **The JumpReLU calculation is correct for released Gemma Scope 2 inference
  weights: the threshold is latent-specific, and the training-time pre-encoder
  decoder bias is folded into the released encoder parameters** (Gemma Scope 2
  technical paper). [This refutes the Gemini review's P1; the harness's blind
  encoder-variant rule remains as a safety net and should select `raw`.]
- `hidden_states[LAYER+1]` is the appropriate HF representation for
  `model.layers.LAYER.output`; the published layer-13 config declares that
  exact hook and L0=60.
- Weight orientation checks are consistent with the released `(d_model, width)`
  encoder / `(width, d_model)` decoder convention.
- The stratified sampling code is deterministic and reaches the requested cap.
- `bash -n` accepts the driver; the nested quoted heredocs are syntactically
  sound.
