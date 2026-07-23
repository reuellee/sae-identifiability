# A Solvable Model of Feature Absorption in L1 Sparse Autoencoders

Exact theory + GPU experiments on when sparse autoencoders (SAEs) can and cannot
recover true features — centered on **feature absorption** (a parent/child concept
pair merging into one latent) in an analytically solvable model.

**Status: actively developed technical report** (started 2026-07-21; revised
2026-07-22 after an external review — `reviews/EXTERNAL_REVIEW_GPT-5.6_2026-07-22.md`
and the point-by-point response beside it). Nine experiment rounds so far
(2,900+ trained SAEs, all confirmatory predictions pre-registered — round 9
additionally dual-reviewed *before* its lock by Gemini 2.5 Pro and GPT-5.6,
`reviews/ROUND9_PREREG_*`), theory
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

**Abstract-level claim (scoped, post-review):** We study feature absorption in
a controlled sparse-autoencoder model. A capacity-limited L1 objective can
prefer composite decoder directions over a planted child feature, while
additional dictionary headroom frequently permits redundant
parent–child–composite representations. In trained models, decoder-level
absorption can coexist with code-level separation through encoder gating —
dictionary identifiability and code identifiability are distinct properties.
On matched synthetic data, a detector combining decoder geometry with code
co-firing identifies many planted parent/composite pairs and recovers the
residual child direction; on semi-synthetic real activations its statistic
separates absorbed from faithful pairs cleanly but the toy-locked cutoff was
knife-edge (pass at m=256, fail at m=128, recorded as registered). The
detector remains a **synthetic proof of concept**: cutoff transfer, scaling
to large overcomplete dictionaries, robustness to nonorthogonality, and
practical false-positive control are open problems.

## Key results

1. **Non-identifiability wall** (ε = 0: child never appears without parent):
   the absorbed configuration is the global optimum over dictionaries of any
   size — the set of active *directions* is unique; dictionary columns and
   sparse codes remain non-unique under permutation, duplicate collinear atoms
   with coefficient splitting, and the presence of unused atoms — and the
   faithful/absorbed ontologies are information-theoretically
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
3. **Practitioner rule of thumb:** λ sets a resolution *scale* — in the
   capacity-limited two-latent model, 1.17·λq is the characteristic scale of
   the absorption transition (exact as the pure-strategy crossover; the global
   two-latent optimum crosses its midpoint at ≈ 0.88·ε\*, SGD at 0.58–0.70·ε\*).
   Deep inside it (ε ≪ 1.17·λq), absorption is the population optimum of the
   objective under the two-latent constraint — there, more compute/data cannot
   help; only changing the objective, λ, or capacity can.
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

Full write-up: [`PAPER.md`](PAPER.md) (formal paper draft, math notation) and
[`report.md`](report.md) (the round-by-round session record). Plan:
[`RESEARCH_PLAN.md`](RESEARCH_PLAN.md).

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
- Environment: theory scripts run on stock python3 (`theory_merged.py`: no deps;
  `verify_*`: sympy/numpy — CPU-verified on Debian 12, python 3.11.2,
  numpy 1.24.2, torch 1.13.1+cpu for SMOKE runs). GPU experiments ran on a
  single NVIDIA L4 (GCP g2, image `dev-gpu-img-tmp`, torch 2.x/CUDA; the exact
  version is printed at the top of each run log in `results/`).
- Result-table provenance (results CSV → commit that produced it):
  | results | commit |
  |---|---|
  | round 1 (`results/round1/`) | `bdcc552` |
  | rounds 2–3 (`results/round2/`, `round3/`) | `3b62c27` |
  | round 4 + OrtSAE-style (`results/round4/`) | `82d2b60` |
  | round 5 critical ratio (`results/round5/`) | `aecffd8` |
  | §14 POC/audit (`results/round6/`) | `ee9d1b8`, bounded `d47c5db` |
  | disambiguation (`results/round6/`) | `08ac574` |
  | §15 capacity-limited (`results/round6/`) | `196df8c` |
  | §15b bg-relative (`results/round6/`) | `05861b0` |
  | Arm A (`results/prereg_armA/`) | `53b7e01` (pre-results lock: `cfd3e09`) |
  | capacity m≥33 rerun (`results/capacity_m33/`) | results `465a139` (pre-results lock `0cba6b2`) |
  | pair-ID Arm 1 (`results/prereg_pairid/`) | results `465a139` (structure lock `e586f02`, threshold lock `1bbca24`) |
  | pair-ID Arm 2 held-out (`results/prereg_pairid/arm2_runs.csv`) | results `319fa1f` (v1.1 frozen pre-run in `465a139`) |
  | round 8 E1–E3 (`results/round8/`) | results `68f444c` (lock `a539c76`, pre-collection amendment `69ca642`) |
  | natural-feature adjudication of S1 (`results/round8/natfeat_*`) | this commit (pre-results lock `0603d38`) |
  | S1 stability, corrected+exclusion rerun (`results/round8/s1_*`) | v1 `841e8cd` (superseded); v2 with oracle-touch exclusion: this commit |
