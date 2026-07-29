# Pre-lock review, round 16 — GPT-5.6 (codex exec, 2026-07-29)

*Captured in two parts (the live stream clipped; head recovered via `codex exec resume`). Verdict: REVISE-BEFORE-LOCK.*

Not lock-ready. I found five P1 issues: the evaluator is not actually fail-closed, activation-cache identity is unchecked, P1 is mechanically favored without a sufficient registered control, the resumable driver violates the registered wipe guarantee, and L1 dose/calibration conformance is not enforced.

## P1 — Lock-blocking

### P1.1 — Gates 1, 2, and 4 do not gate any verdict

> [notes/prereg-round16-l0axis.md:155](/E:/Projects/sae-identifiability/notes/prereg-round16-l0axis.md:155): “**Conformance (fail-closed)**”

The evaluator computes `g1`, `g2`, and `g4`, but P1 depends only on MC and its CI:

> [analysis/analyze_round16.py:229](/E:/Projects/sae-identifiability/analysis/analyze_round16.py:229): `if not any_mc: ... elif lo1 > 0: p1 = "CONFIRMED..."`

Thus a wrong model/layer, missing seed, duplicate SHA, or off-pattern row can coexist with `P1 CONFIRMED`. Missing rows can also yield fewer than the registered eight `u_s` while still producing a verdict.

The interior gate is independently fail-open:

> [analysis/analyze_round16.py:109](/E:/Projects/sae-identifiability/analysis/analyze_round16.py:109): `if interior and int_ids != INTERIOR_PINS:`

An empty interior file passes `g1`; lines 116–118 explicitly make absence “not gating,” contrary to:

> [notes/prereg-round16-l0axis.md:159](/E:/Projects/sae-identifiability/notes/prereg-round16-l0axis.md:159): “Interior rows: exactly the 16 pinned … pairs.”

Fix before lock: define `global_ok = g1 and g2 and g4`; when false, make P1/P2/P3 uninterpretable and exit nonzero. Compare `int_ids != INTERIOR_PINS` unconditionally. Add self-tests proving every global-gate failure suppresses verdicts.

### P1.2 — The prior layer/activation failure class remains possible

The word cache contains `model` and `layer`, but the scorer ignores them:

> [analysis/round16_scorer.py:182](/E:/Projects/sae-identifiability/analysis/round16_scorer.py:182):  
> `W = safe_load(os.environ["WORDS"])`  
> `Xr = W["acts"]...`  
> `letters = np.array(W["letters"])`

The evaluator’s model/layer fields come from the SAE blob, not the scored word activations. A wrong-layer `WORDS` cache can therefore pass every current row gate.

The driver also trusts any pre-existing cache merely because the filename exists:

> [ops/l4_r16.sh:38](/E:/Projects/sae-identifiability/ops/l4_r16.sh:38): `[ -f "$dst" ] || gcloud storage cp ...`

Training and held-out evaluation caches are not cross-checked either. The scorer also discards the SAE blob’s `stats.eval_src`, so “held-out L0” is not verifiable.

Fix: assert for TRAIN, EVAL, and WORDS that embedded `model`, `layer`, and activation dimension agree and equal Pythia-1.4B/L12/2048. Record their full hashes or immutable GCS generations. Preserve and gate `eval_src` in scored rows.

### P1.3 — P1 is mechanically favored; the current D-control is insufficient

P1 is not mathematically guaranteed, because retention, family membership, and learned activations can change. But its registered direction is mechanically favored under an unchanged-representation opportunity model.

The endpoint is defined by the absence of any family firing:

> [analysis/round16_scorer.py:137](/E:/Projects/sae-identifiability/analysis/round16_scorer.py:137): `miss_fam = present & (~fires[..., fam].any(axis=1)) & retained`

Moving TopK from 16 to 64 gives four times as many opportunities to intersect `F_L`; growing families compound this. Moreover, `rate_family + rate_lost` is exactly the no-family-fire rate on probe-present words, so the combined P4 endpoint is definitionally a firing endpoint.

The prereg recognizes this:

> [notes/prereg-round16-l0axis.md:186](/E:/Projects/sae-identifiability/notes/prereg-round16-l0axis.md:186): “firing probability scales with L0 by construction.”

But D-control uses treatment-specific families selected on the same masks via `sel ≥ .30`, different clean-letter sets/family sizes, raw-letter denominators rather than P1’s probe-positive denominator, and no CI or quantitative meaning for “comparable” or “low/flat”:

> [notes/prereg-round16-l0axis.md:144](/E:/Projects/sae-identifiability/notes/prereg-round16-l0axis.md:144): “a **comparable rise** in `fam_fire_absent`”  
> [notes/prereg-round16-l0axis.md:148](/E:/Projects/sae-identifiability/notes/prereg-round16-l0axis.md:148): “stays **low/flat**”

Because `F_L` is selected for high present-minus-absent firing, `fam_fire_absent` can remain low by construction. Flat absent firing plus rising present firing rejects only indiscriminate global firing; it does not establish a “dedicated slot.”

Register something stronger now:

- A deterministic score-time matched-budget sensitivity: construct an exactly/top-at-most-16 firing mask for every SAE, recompute family selection and the no-family-fire endpoint under that common budget, then repeat the seed-clustered k16−k64 contrast.
- A quantitative seed-level specificity contrast, e.g.  
  `[(ff_present−ff_absent)@k64 − (ff_present−ff_absent)@k16]`, using common eligible letters and registered weighting/CI rules.
- State that a native-mask P1 confirmation establishes an L0 association only; representational-change language requires survival under the matched-budget analysis.

### P1.4 — The driver does not perform the registered clean start

The prereg says:

> [notes/prereg-round16-l0axis.md:62](/E:/Projects/sae-identifiability/notes/prereg-round16-l0axis.md:62): “All `sae_*.pt` files are **deleted after calibration and before training**”  
> [notes/prereg-round16-l0axis.md:163](/E:/Projects/sae-identifiability/notes/prereg-round16-l0axis.md:163): “results dir wiped before the round”

The driver instead preserves every on-pattern file and skips its cell:

> [ops/l4_r16.sh:46](/E:/Projects/sae-identifiability/ops/l4_r16.sh:46): “PURGE off-pattern files … keeps completed work”  
> [ops/l4_r16.sh:88](/E:/Projects/sae-identifiability/ops/l4_r16.sh:88): `if [ -f "$OUTF" ]; then ... continue`

The historical round-12 filenames are off-pattern and will be purged, which is good. But an on-pattern pre-lock, smoke, wrong-step, or prior-attempt round-16 file survives. Cached lambda files are likewise accepted without a lock/config association. The evaluator cannot detect wrong steps, caches, resampling recipe, or calibrated-lambda provenance.

The final name gate only prints the count:

> [ops/l4_r16.sh:115](/E:/Projects/sae-identifiability/ops/l4_r16.sh:115): `echo "  $N SAEs pass ... (expect 32)"`

It does not abort when `N != 32`.

Fix: use a first-run sentinel/manifest bound to the lock SHA, exact config, and cache hashes. Wipe on initial creation; preserve outputs only when resuming the same manifest. Abort unless `N == 32`. Validate an existing output before skipping it.

The immediate `rm "$RAWF"` → train → `mv "$RAWF" "$OUTF"` sequence at lines 89–97 is otherwise correct.

### P1.5 — Registered L1 calibration and dose bands are not enforced

The prereg requires:

> [notes/prereg-round16-l0axis.md:84](/E:/Projects/sae-identifiability/notes/prereg-round16-l0axis.md:84): “realized held-out L0 must land in the cell’s band … **fail-closed**”

But the evaluator only tests a 2.5× mean ratio:

> [analysis/analyze_round16.py:176](/E:/Projects/sae-identifiability/analysis/analyze_round16.py:176): `mc[a] = hi >= MC_RATIO * lo`


[transcript continues — recovered tail of P1.5 and the P2/P3 sections:]

The lambda gate also checks only within-cell constancy and `λ16 > λ64`; it never verifies equality to the calibration outputs, despite prereg line 158 requiring that.

Assessment of the 2.5× threshold: acceptable as a minimum “manipulation bit” check, but insufficient as dose conformance. With the registered bands enforced, the minimum intended ratio is already `56/18 ≈ 3.11`.

Fix: ingest the calibration artifacts, compare exact chosen lambdas, define whether the realized band applies per seed or to the cell mean, and fail that arch’s P1 when violated. Keep gate 3 restricted to P3; its role as an arch-dose-matching gate is otherwise correct.

## P2 — Important

### P2.1 — Single-architecture MC pooling is correct, but adjacent logic is not aligned

Under complete conforming data, lines 221–224 correctly implement `u_s`; if one arch fails MC, `u_s` becomes the surviving arch’s `d_s`.

However:

- The failed arch still receives a numerical per-arch CI, merely annotated as excluded.
- P4 averages loss over both arches at lines 310–313, so an MC-failing arch can determine the confound interpretation of a single-arch P1.
- Without global gate enforcement, `u_s` can have variable arch composition or fewer than eight seeds.

Restrict P4’s interpretive comparison to the same MC-passing arches/seeds and label failed-arch CIs explicitly uninterpretable.

### P2.2 — “SHA256” is only a 64-bit prefix

> [analysis/round16_scorer.py:69](/E:/Projects/sae-identifiability/analysis/round16_scorer.py:69): `return h.hexdigest()[:16]`

All 16 interior name/hash pairs correctly match the prefixes in `round13b_results.json`, but these are 16 hexadecimal characters, not full SHA-256 values. Gate 4 also checks only these prefixes.

Use full 64-character digests, or describe them accurately as truncated SHA-256 prefixes.

### P2.3 — Filename/blob identity is incomplete

The evaluator does not verify that:

- blob `arch` equals filename architecture;
- blob `seed` equals filename seed;
- L0/FVU came from the registered held-out cache;
- training used 15,000 steps.

Given the resumable skip behavior, these omissions matter. Add explicit row fields and checks.

### P2.4 — P3’s directional verdict is specified only in code

The prereg gives the P3 interaction formula and power caveat but does not state that positive is `CONFIRMED` and negative is `FALSIFIED-DIRECTION`. The evaluator invents that vocabulary at lines 290–295. Add the criteria to the prereg text and add P3 sign/null self-tests.

## P3 — Verified / minor

- The scorer diff against `round13b_scorer.py` contains only:

  - docstring changes;
  - the two counters at lines 140–155;
  - the harmless default output rename from `round13a_family.json` to `round16_results.json`.

  No pre-existing endpoint, probe, encoding, retention, loss, or aggregation logic changed.

- Both counters are correctly computed from raw `yL` masks and are independent of probe-present/retained calls.
- Gate 3 is correctly used only for P3, with MC additionally requiring both arches.
- The driver’s raw-name deletion and immediate rename sequence is sound.
- `bash -n ops/l4_r16.sh` passes.
- I could not execute the Python self-test because this workspace has no accessible Python executable; static inspection shows it does not test global-gate suppression, missing-interior failure, both-arch MC failure, or P3 verdict branches.

Verdict: REVISE-BEFORE-LOCK
