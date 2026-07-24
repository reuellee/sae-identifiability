#!/bin/bash
# Round 12 RESUME — runs ON the L4 from the EXISTING ~/r12 (caches already present:
# acts_train.pt, acts_eval.pt, words_pythia-1.4b_L12.pt). Fixes the two real-scale
# problems the pythia-70m smoke could not surface:
#   (1) L0 mismatch: the first run calibrated at 8k steps (lam=4 -> L0=32.2) but
#       trained at 15k (-> L0=37.7, outside the |dL0|<=3 gate). FIX: recalibrate at
#       the SAME 15k step budget so the calibrated L0 == the trained L0.
#   (2) speed: fp32 ~46min/SAE -> ~12h. FIX: TF32 (in real_train_sae.py; both
#       arches identical, negligible precision). dead% (46-57%) is REPORTED, not
#       re-engineered (both arches share the recipe -> controlled).
# INTEGRITY: lambda is chosen BLIND to absorption (by L0 only). Metric/scorer/
# predictions are the LOCKED artifacts (commit 0722212) and are NOT touched.
set -e
BUCKET=gs://sae-identifiability-artifacts-ebd5a273/round12
cd ~/r12
RR=results/real
STEPS=${STEPS:-15000}
export GPU_ACTS=1 PYTHONIOENCODING=utf-8
TRAIN=$RR/acts_train.pt; EVAL=$RR/acts_eval.pt; WORDS=$RR/words_pythia-1.4b_L12.pt
for f in $TRAIN $EVAL $WORDS; do [ -f $f ] || { echo "MISSING cache $f -- run l4_r12_pipeline.sh from scratch"; exit 1; }; done
rm -f ~/r12_done $RR/sae_pythia-1.4b_L12_*.pt $RR/sae_pythia-1.4b_L12_*.json   # clear the doomed run's outputs

echo "=== deps ==="; python3 -m pip install --user -q transformers scikit-learn 2>&1 | tail -1 || true

echo "=== RECALIBRATE L1 lambda at the TRAINING step budget ($STEPS), TF32 ==="
# lam=4 gave L0=37.7 at 15k; need slightly higher lam for L0~=32. Grid brackets it.
# calibrate_lambda picks the lam whose 15k-step L0 is closest to 32 (blind to absorption).
ACTS=$TRAIN EVAL_ACTS=$EVAL LAM_GRID="4.5,5,5.5,6,7" CALIB_STEPS=$STEPS TARGET=32 \
  python3 experiments/calibrate_lambda.py > logs_calib.log 2>&1
cat logs_calib.log
LAM=$(grep '^CHOSEN_LAM' logs_calib.log | awk '{print $2}')
BAND=$(grep '^CHOSEN_LAM' logs_calib.log | awk '{print $4}')
echo "$LAM" > chosen_lambda.txt
echo ">>> chosen lambda = $LAM ($BAND) at $STEPS steps"

echo "=== TRAIN 16 SAEs (8 seeds x {topk k=32, l1 lam=$LAM}), TF32, held-out eval ==="
for SEED in 0 1 2 3 4 5 6 7; do
  ACTS=$TRAIN EVAL_ACTS=$EVAL ARCH=topk K=32 SEED=$SEED STEPS=$STEPS python3 experiments/real_train_sae.py >> logs_train.log 2>&1
  ACTS=$TRAIN EVAL_ACTS=$EVAL ARCH=l1 LAM=$LAM SEED=$SEED STEPS=$STEPS python3 experiments/real_train_sae.py >> logs_train.log 2>&1
  set +e; gcloud storage cp $RR/sae_pythia-1.4b_L12_*_s$SEED.pt $BUCKET/ 2>&1 | tail -1; set -e
  echo "seed $SEED done ($(grep -c '^STATS' logs_train.log) SAEs)"
done
grep -h '^STATS' logs_train.log > stats_summary.txt || true
echo "=== TRAIN STATS ==="; cat stats_summary.txt

echo "=== SCORE first-letter absorption (each SAE) ==="
for f in $RR/sae_pythia-1.4b_L12_*.pt; do
  MODE=score SAE=$f WORDS=$WORDS python3 experiments/real_firstletter.py >> logs_score.log 2>&1
done
echo "=== DETECTOR (blind, seeded) ==="
for f in $RR/sae_pythia-1.4b_L12_*.pt; do
  SEED=0 SAE=$f ACTS=$TRAIN python3 experiments/real_analyze.py >> logs_detector.log 2>&1
done
echo "=== FROZEN SCORER (P1/P2/P3), LOCK_LAM=$LAM ==="
N_SEEDS=8 LOCK_LAM=$LAM python3 analysis/analyze_round12.py > results_round12.txt 2>&1
cat results_round12.txt

echo "=== upload results to GCS ==="
set +e
gcloud storage cp $RR/sae_pythia-1.4b_L12_*_fl.json $RR/sae_pythia-1.4b_L12_*_pairs.json $BUCKET/ 2>&1 | tail -1
gcloud storage cp results_round12.txt stats_summary.txt chosen_lambda.txt logs_calib.log $BUCKET/ 2>&1 | tail -1
tar czf r12_results.tgz $RR/sae_pythia-1.4b_L12_*_fl.json $RR/sae_pythia-1.4b_L12_*_pairs.json results_round12.txt stats_summary.txt chosen_lambda.txt 2>/dev/null
gcloud storage cp r12_results.tgz $BUCKET/ 2>&1 | tail -1
set -e
touch ~/r12_done
echo "L4 R12 RESUME COMPLETE (lambda=$LAM)"
