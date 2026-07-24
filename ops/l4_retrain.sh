#!/bin/bash
# Runs ON the L4 box from ~/r11 (after `git pull`), reusing already-extracted
# activations. Preserve acts to GCS -> retrain TopK + L1 (OOM-fixed) -> upload
# SAEs -> analyze absorbed pairs for both -> upload -> done flag. Invoked simply
# (no inline quoting): nohup bash ops/l4_retrain.sh &
BUCKET=gs://sae-identifiability-artifacts-ebd5a273/round11
cd ~/r11
ACTS=results/real/acts_pythia-1.4b_L12.pt
echo "=== preserve acts to GCS ==="; gcloud storage cp $ACTS $BUCKET/ 2>&1 | tail -1
echo "=== train TopK (K=32 x8) ==="
ACTS=$ACTS ARCH=topk K=32 EXPANSION=8 python3 experiments/real_train_sae.py > logs_train_topk.log 2>&1
echo "=== train L1 (lam=5 x8) ==="
ACTS=$ACTS ARCH=l1 LAM=5 EXPANSION=8 python3 experiments/real_train_sae.py > logs_train_l1.log 2>&1
echo "=== upload SAEs ==="
gcloud storage cp results/real/sae_pythia-1.4b_L12_topk_x8.pt results/real/sae_pythia-1.4b_L12_l1_x8.pt $BUCKET/ 2>&1 | tail -2
echo "=== analyze absorbed pairs ==="
SAE=results/real/sae_pythia-1.4b_L12_topk_x8.pt ACTS=$ACTS python3 experiments/real_analyze.py > logs_analyze_topk.log 2>&1
SAE=results/real/sae_pythia-1.4b_L12_l1_x8.pt   ACTS=$ACTS python3 experiments/real_analyze.py > logs_analyze_l1.log 2>&1
gcloud storage cp results/real/sae_pythia-1.4b_L12_topk_x8_pairs.json results/real/sae_pythia-1.4b_L12_l1_x8_pairs.json $BUCKET/ 2>&1 | tail -2
grep -h '^STATS' logs_train_topk.log logs_train_l1.log > stats_summary.txt
echo "=== SUMMARY ==="; cat stats_summary.txt
grep -h 'wrote\|flagged' logs_analyze_topk.log logs_analyze_l1.log
touch ~/r11_done2
echo "RETRAIN COMPLETE"
