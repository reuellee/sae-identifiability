# Round 13a — is the first-letter absorption endpoint measuring absorption or feature splitting?

Registered: `notes/prereg-round13a-family-endpoint.md`, locked `d2d42fe`
(+ Amendment 1 and the evaluator committed pre-results, `8f033ab`).
Frozen scorer `analysis/round13a_family_endpoint.py`, evaluator
`analysis/analyze_round13a.py`. Full output: `results_round13a.txt`.

No new training. The 16 existing frozen round-12 SAEs (8 seeds ×
{L1 λ=4.5, TopK k=32}, m=16384) re-scored on the same held-out words, changing
**only** the endpoint. CPU-only, one throwaway e2-standard-8, ~40 min, ~$0.20.

## Registered verdicts

```
gates: conformance=True seeds=True provenance=True baseline_repro=True
P1 (primary): SURVIVES (CI lower 0.0494 > 0.01)
P2 (primary): R^2=0.381 -> PASS
```

**Gate 4 (the one that mattered).** The re-score harness reproduces the frozen
round-12 single-latent rate on all 16 SAEs, max |Δ| = 0.0017 against a 0.002
tolerance. The new endpoint is therefore being read off a harness that
demonstrably reproduces the registered result.

## P1 — absorption SURVIVES the family correction

| | single-latent | family |
|---|---|---|
| pooled | 0.0724 | **0.0542**  95% CI [0.0494, 0.0592] |
| L1 | 0.0737 | 0.0536 |
| TopK | 0.0711 | 0.0548 |

Scoring each letter against its whole split family instead of one designated
latent removes **25.2%** of the measured absorption. That quarter *was*
splitting. The remaining ~75% is not: it is instances where the letter is still
linearly present in the reconstruction and **no** sufficiently-selective latent
for that letter fires at all.

So the round-12 endpoint **overstates absorption by about a quarter** — a real
validity problem for a SAEBench-style single-latent metric in current use — but
absorption is not an artifact of splitting.

## P2 — the endpoint is meaningfully less a selectivity re-expression

`rate_family = −0.298·max_sel + 0.293`, **R² = 0.381**, against the single-latent
comparator **R² = 0.673**. Registered bar R² < 0.40: **PASS**.

**Honest caveat: this is a narrow pass.** 0.381 sits just under the 0.40 bar, and
38% of the corrected endpoint's variance is still explained by the family's max
selectivity. The family correction improves the metric substantially; it does not
make it clean.

## P3 — heterogeneity is NOT fixed

Top-3-letter share of all absorbed instances: L1 53%→**51%**, TopK 73%→**69%**.
Essentially unchanged. The endpoint is still carried by ~3 of 24 letters
(`s`, `c`, and `r`/`p`) under either definition. Concentration is a property of
the *task*, not of the single-latent rule, and it remains the weakest point of
any first-letter absorption measurement — including round 12's.

## P4 — the architecture null persists under the corrected endpoint

Paired L1−TopK on `rate_family`: **−0.0012, 95% CI [−0.0081, +0.0049]** (n=8).

Round 12's registered null was **not** a splitting artifact. Correcting the
endpoint for splitting leaves the L1-vs-TopK contrast null, and if anything moves
it slightly negative. **This refutes H2** as an explanation of the round-12 null.

(Secondary and non-confirmatory by registration: it reuses round-12 weights and
data. Its role is to size round 13b.)

## P5 — L1 splits about twice as much as TopK

Mean split-family size |F_L|: **L1 2.61 vs TopK 1.25** (medians 2 vs 1; cap 32
never binding, max 9). Paired by letter: **+1.36, CI [+0.94, +1.88]**, larger for
L1 in **22/24** letters.

This is the round-12 diagnosis confirmed on a direct measurement rather than
inferred from selectivity: under an L1 penalty the letter feature is carried by
~2.6 latents; under TopK by ~1.25. It is a large, robust, architecture-level
difference — and it is *not* absorption.

## What round 13a establishes

1. The single-latent first-letter endpoint **inflates absorption by ~25%** via
   feature splitting. Anyone using it — SAEBench included — is measuring a
   composite. Reporting it against a split family is a cheap correction.
2. **Genuine absorption exists** at m=16384 on Pythia-1.4B: 0.054, CI
   [0.049, 0.059], robust to the correction.
3. **H2 is refuted.** The round-12 P1 null is a statement about real absorption,
   not a metric artifact. Round 12's NOT CONFIRMED is *strengthened*.
4. L1 vs TopK differ sharply — but in **splitting** (2.61 vs 1.25 latents/letter),
   not in absorption (0.0536 vs 0.0548).

## What it does not establish

- Nothing confirmatory about architectures: same weights, same data as round 12.
- Nothing about H1, the capacity-regime hypothesis. Round 12 ran at **dead%
  46–57%** — a spare-capacity regime, the same trap that made the earlier m=1536
  GPT-2 POC a null. With H2 eliminated, **H1 is now the live explanation** for why
  no architecture gap appears, and it is untested.

## Next

Round 13b (registered condition met: 13a's P1 SURVIVED). Capacity sweep
m ∈ {2048, 4096, 8192, 16384} at fixed L0=32, both arches, fresh SAEs, per-SAE
dead% recorded so **live latents** — not `m` — index capacity. Primary endpoint
should be the **family** endpoint (13a shows the single-latent one is 25%
splitting). Pre-registered prediction: absorption rises as capacity falls, and any
L1-vs-TopK gap opens only in the scarce regime. Requires GPU (~15h L4, ~$6–8).
