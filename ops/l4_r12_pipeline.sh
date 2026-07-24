#!/bin/bash
# Round 12 — runs ON the L4 (as reuellee_gmail_com) from a fresh clone at ~/r12.
# SMOKE gate -> extract TRAIN + held-out EVAL acts -> build first-letter word set
# -> calibrate L1 lambda to match TopK L0 -> train 10 SAEs (5 seeds x {topk,l1})
# -> score first-letter absorption (non-circular causal) -> detector -> frozen
# scorer. Big artifacts -> GCS; small JSONs/logs stay for the orchestrator.
# Self-contained: git-clones the public repo, uploads to GCS itself.
set -e
BUCKET=gs://sae-identifiability-artifacts-ebd5a273/round12
cd ~/r12
RR=results/real
STEPS=${STEPS:-15000}
export GPU_ACTS=1 PYTHONIOENCODING=utf-8

echo "=== deps ==="
python3 -m pip install --user -q transformers scikit-learn 2>&1 | tail -1 || true

echo "=== SMOKE gate (pythia-70m, full pipeline incl. non-circular causal) ==="
SMOKE=1 python3 experiments/real_extract.py > logs_smoke.log 2>&1
SMOKE=1 MODE=words python3 experiments/real_firstletter.py >> logs_smoke.log 2>&1
SMOKE=1 ACTS=$RR/acts_pythia-70m_L3.pt ARCH=topk K=32 python3 experiments/real_train_sae.py >> logs_smoke.log 2>&1
SMOKE=1 MODE=score SAE=$RR/sae_pythia-70m_L3_topk_x8_s0.pt WORDS=$RR/words_pythia-70m_L3.pt python3 experiments/real_firstletter.py >> logs_smoke.log 2>&1
echo "smoke ok"; rm -f $RR/acts_pythia-70m_L3.pt $RR/words_pythia-70m_L3.pt $RR/sae_pythia-70m_L3_*.pt

echo "=== EXTRACT train acts (Pythia-1.4B L12, 2M tokens, books A) ==="
SPLIT=train N_TOKENS=2000000 python3 experiments/real_extract.py > logs_extract_train.log 2>&1
TRAIN=$RR/acts_pythia-1.4b_L12.pt
mv $TRAIN $RR/acts_train.pt; TRAIN=$RR/acts_train.pt

echo "=== EXTRACT held-out eval acts (disjoint books B, 400k tokens) ==="
SPLIT=eval N_TOKENS=400000 OUT=$RR/acts_eval.pt python3 experiments/real_extract.py > logs_extract_eval.log 2>&1
EVAL=$RR/acts_eval.pt

echo "=== BUILD first-letter word set (Pythia-1.4B L12, held out) ==="
MODE=words python3 experiments/real_firstletter.py > logs_words.log 2>&1
WORDS=$RR/words_pythia-1.4b_L12.pt

echo "=== upload caches to GCS (non-fatal) ==="
set +e; gcloud storage cp $TRAIN $EVAL $WORDS $BUCKET/ 2>&1 | tail -2; set -e

echo "=== CALIBRATE L1 lambda to TopK L0=32 (seed 0) ==="
ACTS=$TRAIN EVAL_ACTS=$EVAL python3 experiments/calibrate_lambda.py > logs_calib.log 2>&1
cat logs_calib.log
LAM=$(grep '^CHOSEN_LAM' logs_calib.log | awk '{print $2}')
echo "chosen lambda = $LAM"; echo "$LAM" > chosen_lambda.txt

echo "=== TRAIN 16 SAEs (8 seeds x {topk k=32, l1 lam=$LAM}), held-out eval ==="
for SEED in 0 1 2 3 4 5 6 7; do
  ACTS=$TRAIN EVAL_ACTS=$EVAL ARCH=topk K=32 SEED=$SEED STEPS=$STEPS python3 experiments/real_train_sae.py >> logs_train.log 2>&1
  ACTS=$TRAIN EVAL_ACTS=$EVAL ARCH=l1 LAM=$LAM SEED=$SEED STEPS=$STEPS python3 experiments/real_train_sae.py >> logs_train.log 2>&1
  # upload weights as each seed completes (spot-preemption safe)
  set +e; gcloud storage cp $RR/sae_pythia-1.4b_L12_*_s$SEED.pt $BUCKET/ 2>&1 | tail -1; set -e
done
grep -h '^STATS' logs_train.log > stats_summary.txt || true
echo "=== TRAIN STATS ==="; cat stats_summary.txt

echo "=== SCORE first-letter absorption (each SAE) ==="
for f in $RR/sae_pythia-1.4b_L12_*.pt; do
  MODE=score SAE=$f WORDS=$WORDS python3 experiments/real_firstletter.py >> logs_score.log 2>&1
done

echo "=== DETECTOR (blind, seeded, shared subsample) ==="
for f in $RR/sae_pythia-1.4b_L12_*.pt; do
  SEED=0 SAE=$f ACTS=$TRAIN python3 experiments/real_analyze.py >> logs_detector.log 2>&1
done

echo "=== FROZEN SCORER (P1/P2/P3) ==="
N_SEEDS=8 LOCK_LAM=$LAM python3 analysis/analyze_round12.py > results_round12.txt 2>&1
cat results_round12.txt

echo "=== upload result JSONs to GCS ==="
set +e; gcloud storage cp $RR/sae_pythia-1.4b_L12_*_fl.json $RR/sae_pythia-1.4b_L12_*_pairs.json $BUCKET/ 2>&1 | tail -1; set -e

touch ~/r12_done
echo "L4 R12 PIPELINE COMPLETE"
