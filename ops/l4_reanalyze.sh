#!/bin/bash
# Re-run the absorbed-pair detector on both real SAEs (after the analyze OOM fix).
BUCKET=gs://sae-identifiability-artifacts-ebd5a273/round11
cd ~/r11
A=results/real/acts_pythia-1.4b_L12.pt
SAE=results/real/sae_pythia-1.4b_L12_topk_x8.pt ACTS=$A python3 experiments/real_analyze.py > logs_analyze_topk.log 2>&1
SAE=results/real/sae_pythia-1.4b_L12_l1_x8.pt   ACTS=$A python3 experiments/real_analyze.py > logs_analyze_l1.log 2>&1
gcloud storage cp results/real/sae_pythia-1.4b_L12_topk_x8_pairs.json results/real/sae_pythia-1.4b_L12_l1_x8_pairs.json $BUCKET/ 2>&1 | tail -2
echo "=== RESULTS ==="; grep -h 'rate-window\|flagged\|wrote' logs_analyze_topk.log logs_analyze_l1.log
touch ~/r11_reanalyze_done
echo REANALYZE COMPLETE
