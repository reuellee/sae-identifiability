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

## GPU environment (pinned; captured live from the session box 2026-07-22)

Single NVIDIA L4 (GCP g2-standard-8), image `dev-gpu-img-tmp`
(created 2026-07-21 from the round-1–4 research box), Ubuntu 22.04:
- python 3.10.12
- torch 2.5.1+cu121 (CUDA 12.1, cuDNN 9.1.0.70, triton 3.1.0, NCCL 2.21.5)
- numpy 1.26.4, sympy 1.13.1, networkx 3.4.2
- nvidia-* cu12 wheels: cublas 12.1.3.1, cufft 11.0.2.54, curand 10.3.2.106,
  cusolver 11.4.5.107, cusparse 12.1.0.106, nvjitlink 12.9.86, nvtx 12.1.105
  (pip user-site of `reuellee_gmail_com`)

All GPU rounds in `results/` after the image date ran on this stack; rounds
1–4 ran on the box this image was taken from (same install).

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
