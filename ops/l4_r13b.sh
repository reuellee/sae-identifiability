#!/usr/bin/env bash
# Round 13b on the L4: capacity sweep m in {2048,4096,8192,16384} x {L1,TopK} x 8 seeds.
# Prereg: notes/prereg-round13b-capacity.md (LOCKED before this ran).
#
# Anti-contamination (round 12's failure mode): the results dir is wiped at the
# start, every trained file is name-checked against the registered pattern, and
# any file not matching is DELETED before scoring. Round 12 was corrupted by a
# stale out-of-config SAE carrying a duplicate seed.
set -euo pipefail

R=${R:-$HOME/r13b}
REPO=$R/sae-identifiability
RR=$REPO/results/real
BUCKET=${BUCKET:-gs://sae-identifiability-artifacts-ebd5a273}
STEPS=${STEPS:-15000}
LAM_GRID=${LAM_GRID:-"2,3,4,4.5,5,6"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
EXPANSIONS=${EXPANSIONS:-"1 2 4 8"}
TRAIN=$R/acts_train.pt
EVAL=$R/acts_eval.pt
WORDS=$R/words_pythia-1.4b_L12.pt
NAME_RE='^sae_pythia-1\.4b_L12_(l1|topk)_x(1|2|4|8)_s[0-7]\.pt$'

cd "$REPO"
mkdir -p "$RR"
echo "=== WIPE results dir (anti-contamination) ==="
rm -f "$RR"/sae_*.pt "$RR"/*_fl.json "$RR"/round13b_*.json
export PYTHONUNBUFFERED=1 GPU_ACTS=1

# ---------------------------------------------------------- 1. calibrate lambda
declare -A LAMS
for X in $EXPANSIONS; do
  echo "=== CALIBRATE lambda, EXPANSION=$X (m=$((X*2048))), $STEPS steps, grid $LAM_GRID ==="
  ACTS=$TRAIN EVAL_ACTS=$EVAL EXPANSION=$X LAM_GRID="$LAM_GRID" \
    CALIB_STEPS=$STEPS TARGET=32 \
    python3 experiments/calibrate_lambda.py > "$R/calib_x$X.log" 2>&1
  cat "$R/calib_x$X.log"
  LAMS[$X]=$(grep '^CHOSEN_LAM' "$R/calib_x$X.log" | awk '{print $2}')
  echo ">>> EXPANSION=$X lambda=${LAMS[$X]}"
  echo "$X ${LAMS[$X]}" >> "$R/chosen_lambda_by_width.txt"
done

# ------------------------------------------------------------------- 2. train 64
echo "=== TRAIN 64 SAEs (4 widths x 2 arches x 8 seeds), TF32, held-out eval ==="
for X in $EXPANSIONS; do
  for SEED in $SEEDS; do
    ACTS=$TRAIN EVAL_ACTS=$EVAL EXPANSION=$X ARCH=topk K=32 SEED=$SEED STEPS=$STEPS \
      python3 experiments/real_train_sae.py >> "$R/train.log" 2>&1
    ACTS=$TRAIN EVAL_ACTS=$EVAL EXPANSION=$X ARCH=l1 LAM=${LAMS[$X]} SEED=$SEED STEPS=$STEPS \
      python3 experiments/real_train_sae.py >> "$R/train.log" 2>&1
  done
  echo "width x$X done ($(ls "$RR"/sae_*.pt 2>/dev/null | wc -l) SAEs so far)"
  set +e; gcloud storage cp "$RR"/sae_pythia-1.4b_L12_*_x${X}_s*.pt "$BUCKET/round13b/" 2>&1 | tail -1; set -e
done

# ------------------------------------------------- 3. enforce the naming gate
echo "=== NAME GATE (delete anything off-pattern BEFORE scoring) ==="
for f in "$RR"/sae_*.pt; do
  b=$(basename "$f")
  if ! [[ $b =~ $NAME_RE ]]; then echo "  DELETING off-pattern: $b"; rm -f "$f"; fi
done
N=$(ls "$RR"/sae_*.pt | wc -l)
echo "  $N SAEs pass the name gate (expect 64)"

# --------------------------------------------------------------- 4. score them
echo "=== SCORE (frozen 13b scorer: single + family + lost) ==="
SAES=$(ls "$RR"/sae_*.pt | tr '\n' ',' | sed 's/,$//')
WORDS=$WORDS SAES="$SAES" OUT=$R/round13b_results.json THETA=0.0 TAU=0.30 \
  python3 analysis/round13b_scorer.py > "$R/score.log" 2>&1
tail -5 "$R/score.log"

# ------------------------------------------------------------ 5. frozen verdict
echo "=== FROZEN EVALUATOR ==="
R13B=$R/round13b_results.json python3 analysis/analyze_round13b.py \
  > "$R/results_round13b.txt" 2>&1
cat "$R/results_round13b.txt"

# --------------------------------------------------------------- 6. ship it out
set +e
gcloud storage cp "$R/round13b_results.json" "$R/results_round13b.txt" \
  "$R/chosen_lambda_by_width.txt" "$R"/calib_x*.log "$R/train.log" \
  "$BUCKET/round13b/" 2>&1 | tail -2
set -e
echo "L4 R13B COMPLETE"
