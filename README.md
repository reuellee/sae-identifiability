# SAE Identifiability: Feature Absorption as a Phase Transition

Exact theory + GPU experiments on when sparse autoencoders (SAEs) can and cannot
recover true features — centered on **feature absorption** (a parent/child concept
pair merging into one latent) as a solvable phase transition.

**Status: research program complete** (2026-07-21, single day). Six experiment
rounds (2,000+ trained SAEs, all predictions pre-registered), theory symbolically
verified, real-data validation of the capacity-dependent regime structure
(report.md §14–15b), and an independent adversarial referee review applied
(no blocking defects). Highlights beyond the list below: a general no-go
theorem for coherence penalties with a √2 critical occurrence ratio (§7.1b,
`theory/general_no_go.md`), an event-weighted remedy that provably eliminates
the transition and rescues absorbed features on real GPT-2 activations in the
capacity-limited regime (§12, §15), and two label-free estimators refuted with
understood mechanisms (§15b — the remaining open problem).

## Key results

1. **Non-identifiability wall** (ε = 0: child never appears without parent):
   the absorbed dictionary is the *unique global optimum over dictionaries of any
   size*, and the faithful/absorbed ontologies are information-theoretically
   indistinguishable. Two-line proof via ‖f‖₁ ≥ ‖Df‖.
2. **Exact phase boundary** (machine-checked, sympy):
   absorption is globally preferred iff
   **ε < ε\*(λ, q) = λq(8 − 4√2 − λ) / (2(1 − (2 − √2)λ)) ≈ 1.17 λq**,
   where ε = P(child solo), q = P(parent∧child), λ = L1 coefficient.
   135 GPU runs collapse onto a single sigmoid under ε/ε\* rescaling.
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
   (GPU-confirmed); two-child rescue under rich metrics is being tested in round 3.

Full write-up: [`report.md`](report.md).

## Layout

- `report.md` — the main report (theory + experiments; updated as rounds land)
- `theory/` — machine-checked proofs and exact scans
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
  estimation): `label-free-frequency-identifiability.md` (a no-go for binarized
  co-firing signatures + a within-composite bimodality estimator, with the
  topic-model/tensor identifiability backbone) and its `prereg-*` experiment spec
  (**theory + pre-registered predictions; not yet run**)

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
- Theory claims are machine-verified (symbolic KKT enumeration / exact numeric
  scans), and every experimental analysis script is in `analysis/`.
