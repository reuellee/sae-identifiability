# Environment & run commands

## CPU verification environment (independently rerun, clean box)

Debian 12 (bookworm), GCP e2-standard-4 "orchestrator":
- python 3.11.2 (`python3`, Debian)
- numpy 1.24.2 (`python3-numpy` 1:1.24.2-1+deb12u1)
- sympy 1.11.1 (`python3-sympy`, Debian bookworm)
- torch 1.13.1+cpu (`python3-torch` 1.13.1+dfsg-4) — SMOKE runs only

Independently rerun from this clean environment on 2026-07-22 (no GPU, no
project venv):
- `python3 theory/theory_merged.py` — reproduces the corrected penalty
  analysis and three-basin comparison (no deps).
- `python3 theory/symbolic_verify.py` — all six formula checks pass
  (ε₀, p₀\*, p₀\*/q → √2, ε\*, C(λ), λ_crit, k\*).
- `cd results/round1 && python3 ../../analysis/analyze_ab.py` — reproduces the
  round-1 transition table from committed CSVs (stdlib only).
- `SMOKE=1 python3 experiments/prereg_bimodality_armA.py` — end-to-end smoke
  of the Arm A suite on CPU torch.

## GPU environment

Single NVIDIA L4 (GCP g2-standard-8), image `dev-gpu-img-tmp`
(created 2026-07-21 from the round-1–4 research box). Exact python/torch/CUDA
versions are captured per session in the run logs collected into `results/`
(`env.txt` where present; earlier rounds: see the log headers). A full pip
lockfile for the GPU image is pending (tracked in
`reviews/RESPONSE_GPT-5.6.md` §Pending).

## Run commands

Theory (seconds, CPU): as above.

GPU rounds (each ≈ minutes–1h on an L4):
```bash
python3 experiments/sae_experiments.py        # round 1
python3 experiments/sae_remedies.py           # round 1 remedies
python3 experiments/sae_round2.py             # round 2 (batched)
python3 experiments/sae_round3.py             # round 3
python3 experiments/sae_round4.py             # round 4 weighted remedy
python3 experiments/sae_round5.py             # round 5 critical ratio
python3 experiments/prereg_bimodality_armA.py # Arm A (results/prereg_armA/)
python3 experiments/prereg_pairid_arm1.py     # pair-ID Arm 1 (results/prereg_pairid/)
python3 experiments/capacity_m33_rerun.py     # m>=33 capacity rerun (results/capacity_m33/)
python3 experiments/prereg_pairid_arm2.py     # pair-ID Arm 2 (gated; needs GPT-2 acts)
```
`SMOKE=1` runs tiny end-to-end versions of the batched suites.
Result-table → commit provenance: see README.
