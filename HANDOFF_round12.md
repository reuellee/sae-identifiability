# HANDOFF — round 12 completion (2026-07-24)

Ownership handed to the **orchestrator** so the laptop session could close.
Round 12 is the pre-registered, **LOCKED** (commit `c0eb337`) confirmatory
matched-seed L1-vs-TopK real-SAE first-letter-absorption experiment on
Pythia-1.4B L12. Dual pre-lock review complete (Gemini LOCK-READY; GPT 12
findings, 3 P1-critical fixed+verified — see `reviews/`).

## State at handoff
- **First 16-SAE run was KILLED** after 2 SAEs — two real-scale problems the
  pythia-70m SMOKE could not surface:
  1. **L0 mismatch (blocks P1):** calibrated λ=4.0 at 8k steps → L0=32.2, but
     training at 15k → **L0=37.7**, outside the scorer's `|ΔL0|≤3` matched gate.
  2. **Too slow:** fp32 ~46 min/SAE → ~12h.
  (Also dead% 46-57% — REPORTED, not re-engineered; both arches share the recipe.)
- **Caches preserved on the L4** `dev-gpu-2` (zone `us-east1-b`), in `~/r12/results/real/`:
  `acts_train.pt` (3.9G), `acts_eval.pt` (1.5G, held-out), `words_pythia-1.4b_L12.pt`.
- **Fixes applied + pushed** (do NOT touch the frozen metric/scorer/predictions):
  - `experiments/real_train_sae.py`: **TF32** enabled (infra; both arches identical).
  - `ops/l4_r12_resume.sh`: **recalibrates λ at 15k steps** (so calibrated L0 ==
    trained L0), grid {4.5,5,5.5,6,7} picking L0 closest to 32, then trains 16
    SAEs + TF32, scores, detector, frozen scorer → `results_round12.txt` → GCS.
  - `ops/drive_r12.sh`: orchestrator-side driver (below).

## INTEGRITY (must hold)
- **Every recipe choice is BLIND to the absorption result** — λ by L0, TF32 by
  speed, nothing tuned to move L1−TopK. Never inspect absorption and adjust a
  knob. The frozen metric/scorer/predictions (commit c0eb337) are UNCHANGED, so
  **no re-review is needed** (re-running Gemini/GPT here just respawns churn).
- The calibration is on **seed 0**; the gate is on the **8-seed mean**. Aim
  seed-0 L0 **near 32** (margin), because seeds scatter and the mean must clear ±3.
- Report the outcome the frozen scorer yields — CONFIRM / FALSIFIED /
  NOT-CONFIRMED — whichever it is. Failure is a valid, publishable result here.

## To complete (deterministic — already wired)
On the orchestrator: `cd ~/sae-identifiability && git pull && nohup bash
ops/drive_r12.sh > ~/drive_r12.out 2>&1 &`. It: pushes code to the L4, launches
`l4_r12_resume.sh`, polls to completion (~4h with TF32), pulls results from GCS,
commits `results/real/SUMMARY_round12.md` + JSONs, pushes, and **deletes the L4**
on clean completion (leaves it up if the run died, for inspection). Progress in
`~/drive_r12.log`.

## Judgment layer — do INTERACTIVELY (not an unsupervised autonomous agent)
`drive_r12.sh` already commits a FAITHFUL result (the frozen scorer decides
CONFIRM/FALSIFIED/NOT-CONFIRMED honestly and it is committed verbatim) and stops
the L4 — so the science + honest report + cleanup complete with no agent. A
cron-fired `claude -p --dangerously-skip-permissions` finalize routine was
considered and **removed**: a standing full-permission autonomous agent running
unsupervised (nobody watching) is an unacceptable risk, and it is not needed.
When someone next opens a session (orchestrator or laptop), do the judgment layer
interactively: (1) sanity-check the frozen-scorer verdict + that the matched-L0 /
conformance / seed / matched-letter gates all passed; (2) enrich
`results/real/SUMMARY_round12.md` (keep the verdict verbatim; add per-arch
absorption/loss/dead%/FVU, P2 concentration by arch, P3 recall-vs-baseline, and
the north-star meaning); (3) update `RESEARCH_PLAN.md` + `CLAIM_LEDGER.md`;
(4) confirm the L4 is gone (`gcloud compute instances list`).
The north star: geometry → identifiable codes → **causally valid features** →
reusable abstractions. If P1 holds, the registered next step is the deferred
**Chanin model-behavior causal test** (round-12 P2 was only reconstruction-space).

## Infra notes
- L4 `dev-gpu-2` us-east1-b; GCS `gs://sae-identifiability-artifacts-ebd5a273/round12`.
- On the orchestrator (Linux) `gcloud compute ssh` uses OpenSSH — clean. (On the
  laptop it used plink, which rejects `-o` flags — irrelevant here.)
- L4 acquisition needs `--maintenance-policy=TERMINATE`; L4 stockouts common in
  us-central/west (got one in us-east1-b). Always STOP the box when done.
