#!/bin/bash
# Runs ON the L4 (as reuellee_gmail_com) from a fresh clone at ~/r11.
# SMOKE-gate -> extract Pythia-1.4B activations -> train TopK + L1 SAEs ->
# upload big artifacts (acts, weights) to GCS -> leave small logs/stats for the
# orchestrator to collect. Self-contained (git-clones the public repo, no scp).
set -e
BUCKET=gs://sae-identifiability-artifacts-ebd5a273/round11
cd ~/r11
echo "=== pip transformers ==="
python3 -m pip install --user -q transformers 2>&1 | tail -1 || true

echo "=== SMOKE gate ==="
SMOKE=1 python3 experiments/real_extract.py > logs_smoke_ext.log 2>&1
SMOKE=1 ACTS=results/real/acts_pythia-70m_L3.pt python3 experiments/real_train_sae.py > logs_smoke_tr.log 2>&1
echo "smoke ok"

echo "=== EXTRACT Pythia-1.4B layer 12, 2M tokens ==="
python3 experiments/real_extract.py > logs_extract.log 2>&1
ACTS=results/real/acts_pythia-1.4b_L12.pt

echo "=== TRAIN TopK (K=32, x8) ==="
ACTS=$ACTS ARCH=topk K=32 EXPANSION=8 python3 experiments/real_train_sae.py > logs_train_topk.log 2>&1

echo "=== TRAIN L1 (lam=5, x8) ==="
ACTS=$ACTS ARCH=l1 LAM=5 EXPANSION=8 python3 experiments/real_train_sae.py > logs_train_l1.log 2>&1

# stats first (must survive even if the GCS upload has a scope issue)
grep -h '^STATS' logs_train_topk.log logs_train_l1.log > stats_summary.txt || true
echo "=== STATS ==="; cat stats_summary.txt

echo "=== UPLOAD artifacts to GCS (non-fatal) ==="
set +e
gcloud storage cp results/real/*.pt $BUCKET/ 2>&1 | tail -3
echo "upload exit: $?"

touch ~/r11_done
echo "L4 PIPELINE COMPLETE"
