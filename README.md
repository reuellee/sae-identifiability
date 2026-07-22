# A Solvable Model of Feature Absorption in L1 Sparse Autoencoders

Exact theory + GPU experiments on when sparse autoencoders (SAEs) can and cannot
recover true features — centered on **feature absorption** (a parent/child concept
pair merging into one latent) in an analytically solvable model.

**Status: actively developed technical report** (started 2026-07-21; revised
2026-07-22 after an external review — `reviews/EXTERNAL_REVIEW_GPT-5.6_2026-07-22.md`
and the point-by-point response beside it). Seven experiment rounds so far
(2,500+ trained SAEs, all confirmatory predictions pre-registered), theory
computationally verified (sympy — not a proof assistant), semi-synthetic
validation of the capacity-dependent regime structure on real GPT-2 activations
(report.md §14–15b), plus a fresh-context LLM referee pass (adversarial, not
external human peer review). Highlights beyond the list below: a no-go
theorem for coherence penalties in the undercomplete orthonormal setting, with
a √2 critical occurrence ratio (§7.1b, `theory/general_no_go.md`); an
event-weighted remedy that provably eliminates the toy-model transition and
rescues injected absorbed features on real GPT-2 activations in the
capacity-limited regime (§12, §15 — oracle-dependent: a diagnostic existence
result, not yet a practical method); two label-free estimators refuted with
understood mechanisms (§15b); and the Arm A discovery that trained absorption
is *gated* ("leaky"), inverting both halves of a pre-registered
identifiability prediction (§16).

## Key results

1. **Non-identifiability wall** (ε = 0: child never appears without parent):
   the absorbed configuration is the global optimum over dictionaries of any
   size — the *active* dictionary is unique up to permutation and unused
   atoms — and the faithful/absorbed ontologies are information-theoretically
   indistinguishable. Two-line proof via ‖f‖₁ ≥ ‖Df‖.
2. **Exact pure-strategy crossover** (computationally verified, sympy):
   among per-pair 2-latent dictionaries, the pure absorbed dictionary beats the
   pure faithful one iff
   **ε < ε\*(λ, q) = λq(8 − 4√2 − λ) / (2(1 − (2 − √2)λ)) ≈ 1.17 λq**,
   where ε = P(child solo), q = P(parent∧child), λ = L1 coefficient.
   The continuously optimized dictionary tilts smoothly through intermediate
   angles rather than jumping (report §5) — ε\* organizes, not equals, the
   practical transition. 135 GPU runs collapse onto a single sigmoid under
   ε/ε\* rescaling (empirical midpoint at 0.58–0.70·ε\*).
3. **Practitioner rule of thumb:** λ is a resolution limit — feature pairs with
   ε ≲ 1.17·λq get absorbed *at the objective's optimum*; more compute/data
   provably cannot help. Only changing the objective (or λ) can.
4. **Coherence-penalty remedy: pre-registered prediction REFUTED, theory
   corrected.** The predicted critical penalty β\* was derived by comparing only
   faithful-vs-absorbed pairs. The true global optimum at β ≥ β\*, small ε, is an
   **anti-rotated absorbed pair** {≈−40°, ≈+46°}: the composite keeps absorbing
   while the parent rotates to make the pair near-orthogonal, zeroing the penalty.
   Corrected boundary ε\*\*(β): in its domain p₀ ≲ √2·q (co-occurrence-dominated
   hierarchies; above that ratio penalties genuinely work), the penalty shrinks the
   absorption region at most ~4× (optimum near β ≈ β\*), **never eliminates it**, and larger β makes it
   *worse* (ε\*\* increases in β, saturating as both branches become orthogonal
   frames competing only on rotation angle). Caught by pre-registration +
   GPU falsification; see `theory/theory_merged.py`.
5. **Matryoshka SAEs:** exact mechanism analysis — prefix scarcity + parent
   reusability, not the naive account. Single-child hierarchies are unrescuable
   (GPU-confirmed); two-child rescue under rich metrics was a *partial* rescue in
   round 3 (child-dominant latents, not a clean fix).

Full write-up: [`report.md`](report.md).

## Layout

- `report.md` — the main report (theory + experiments; updated as rounds land)
- `theory/` — computationally verified derivations and exact scans
  (`verify_absorption_theory.py`, `verify_remedies.py`, `matryoshka_multichild.py`,
  `theory_curves.py`, `theory_merged.py` — the corrected variable-latent-count +
  penalty analysis; pure python, no deps)
- `experiments/` — GPU experiment code (PyTorch, runs on a single NVIDIA L4).
  `sae_round2.py`/`sae_round3.py` train hundreds of SAEs *simultaneously* by
  folding the run dimension into batched einsums (~40× wall-clock speedup vs
  serial runs)
- `analysis/` — result analysis and figures
- `results/` — raw CSVs per round
- `ops/` — GCP supervision script (crash-restart chain, result collection,
  auto-stop)
- `notes/` — follow-up theory notes on the §15b open problem (label-free frequency
  estimation): `label-free-frequency-identifiability.md` + its pre-registration
  (**Arm A run 2026-07-22: both hypotheses inverted** — trained absorption is
  gated/leaky; see report §16 and `results/prereg_armA/SUMMARY.md`), and
  `prereg-pair-identification.md` (the successor experiment: label-free
  detection of absorbed pairs)
- `reviews/` — adversarial review artifacts: fresh-context LLM referee reports,
  the external GPT-5.6 review (2026-07-22), and point-by-point responses

## Reproducing

Theory (laptop, `theory/theory_merged.py` needs nothing but python3;
`verify_*` need sympy/numpy):

```bash
python3 theory/theory_merged.py
```

Experiments (any CUDA box; each round ≈ minutes on an L4):

```bash
python3 experiments/sae_experiments.py   # round 1: 135-run absorption grid + recovery
python3 experiments/sae_remedies.py      # round 1: remedy tests C1-C3
python3 experiments/sae_round2.py        # batched fine transition + oracle-init controls
python3 experiments/sae_round3.py        # corrected-boundary validation + rich C3 metrics
```

`SMOKE=1` runs a tiny end-to-end version of the round-2/3 suites.

## Method notes

- All C-experiment predictions were **pre-registered** before results; scoring is
  reported honestly including the refuted C1 prediction (which produced result 4).
- Theory claims are computationally verified (symbolic KKT enumeration in sympy /
  exact numeric scans — verification scripts, not a formal proof assistant), and
  every experimental analysis script is in `analysis/`.
- Environment: theory scripts run on stock python3 (+ sympy/numpy where noted);
  GPU experiments used torch 2.x/CUDA on a single NVIDIA L4 (GCP g2), CPU-smoke
  verified on torch 1.13. Result tables cite their source CSVs in `results/`;
  per-round provenance is in the git history (each round lands as one commit).
