#!/usr/bin/env bash
# Round 16 on the L4: L0-axis sweep at fixed m=16384 (EXPANSION=8).
# Prereg: notes/prereg-round16-l0axis.md (LOCKED before this runs).
#
# Cells: {l1,topk} x L0-target {16,64} x seeds 0-7 = 32 fresh SAEs, plus the
# 13b x8 interior cell (16 frozen SAEs) RE-SCORED only. The trainer's output
# name does not encode k/lambda, so every file is renamed to the registered
# k-tagged pattern IMMEDIATELY after its run; anything left in the raw name
# is debris and is deleted (round-12's stale-file failure class).
#
# Resumable: a (cell, seed) whose renamed output exists is skipped, and chosen
# lambdas are cached to disk -- a spot preemption costs at most the in-flight
# SAE. Rerun the script to continue.
set -euo pipefail

R=${R:-$HOME/r16}
REPO=$R/sae-identifiability
RR=$REPO/results/real
INT=$R/interior
BUCKET=${BUCKET:-gs://sae-identifiability-artifacts-ebd5a273}
STEPS=${STEPS:-15000}
MAX_EVALS=${MAX_EVALS:-6}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
KTARGETS=${KTARGETS:-"16 64"}
TRAIN=$R/acts_train.pt
EVAL=$R/acts_eval.pt
WORDS=$R/words_pythia-1.4b_L12.pt
NAME_RE='^sae_pythia-1\.4b_L12_(l1|topk)_x8_k(16|64)_s[0-7]\.pt$'

cd "$REPO"
mkdir -p "$RR" "$INT"
LOCK=$(git rev-parse HEAD)
echo "=== round 16 @ $LOCK ==="

# ---------------- run manifest (GPT pre-lock P1.4): the registered clean
# start happens exactly once per lock; resume is only valid against the SAME
# lock. Any manifest mismatch aborts rather than silently mixing runs.
if [ -f "$R/RUN_MANIFEST" ]; then
  OLD=$(awk '/^lock/{print $2}' "$R/RUN_MANIFEST")
  if [ "$OLD" != "$LOCK" ]; then
    echo "!!! RUN_MANIFEST lock $OLD != checked-out $LOCK -- refusing to resume."
    echo "!!! Delete $R/RUN_MANIFEST only if you intend a FRESH round (wipes all outputs)."
    exit 3
  fi
  echo "=== resuming run for lock $LOCK ==="
else
  echo "=== FIRST RUN for lock $LOCK: registered clean start (wipe outputs) ==="
  rm -f "$RR"/sae_*.pt "$R"/chosen_lambda_k*.txt "$R"/chosen_lambdas.txt \
        "$R"/round16_*.json "$R"/results_round16.txt
  { echo "lock $LOCK"; date -u; } > "$R/RUN_MANIFEST"
fi

# ------------------------------------------------------------ 0. inputs
# Pinned md5s (base64, recorded from GCS object metadata at lock time —
# GPT pre-lock P1.2: cache identity must be verified, not trusted by name).
declare -A MD5
MD5[$TRAIN]="D48OnIJPLIGqKgsCMur+JQ=="
MD5[$EVAL]="hNwKlaK4hG1H7SMchaglDw=="
MD5[$WORDS]="VX/fNafoHdj8D37yWgfSrw=="
for f in "$TRAIN:round12/acts_train.pt" "$EVAL:round12/acts_eval.pt" \
         "$WORDS:round12/words_pythia-1.4b_L12.pt"; do
  dst=${f%%:*}; src=${f##*:}
  [ -f "$dst" ] || gcloud storage cp "$BUCKET/$src" "$dst"
  got=$(gcloud storage hash "$dst" --skip-crc32c 2>/dev/null | awk '/md5/{print $NF}')
  if [ "$got" != "${MD5[$dst]}" ]; then
    echo "!!! CACHE IDENTITY FAILURE: $dst md5=$got expected ${MD5[$dst]}"
    exit 4
  fi
  echo "  cache ok: $(basename "$dst")"
done
if [ "$(ls "$INT"/sae_*_x8_s*.pt 2>/dev/null | wc -l)" -ne 16 ]; then
  gcloud storage cp "$BUCKET/round13b/sae_pythia-1.4b_L12_l1_x8_s"{0..7}".pt" \
                    "$BUCKET/round13b/sae_pythia-1.4b_L12_topk_x8_s"{0..7}".pt" "$INT/"
fi

# ------------------------- 1. purge anything not in the registered pattern
echo "=== PURGE off-pattern files (anti-contamination; keeps completed work) ==="
for f in "$RR"/sae_*.pt; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  if ! [[ $b =~ $NAME_RE ]]; then echo "  deleting: $b"; rm -f "$f"; fi
done
export PYTHONUNBUFFERED=1 GPU_ACTS=1

# ---------------------------------------------------- 2. calibrate lambda
declare -A LAMS BANDLO BANDHI
BANDLO[16]=14; BANDHI[16]=18; BANDLO[64]=56; BANDHI[64]=72
for KT in $KTARGETS; do
  if [ -f "$R/chosen_lambda_k$KT.txt" ]; then
    LAMS[$KT]=$(awk '{print $1}' "$R/chosen_lambda_k$KT.txt")
    echo "=== lambda for L0-target $KT cached: ${LAMS[$KT]} ==="
    continue
  fi
  echo "=== CALIBRATE lambda (adaptive), TARGET=$KT, m=16384, $STEPS steps ==="
  ACTS=$TRAIN EVAL_ACTS=$EVAL EXPANSION=8 MAX_EVALS=$MAX_EVALS \
    CALIB_STEPS=$STEPS TARGET=$KT LAM0=4.5 \
    BAND_LO=${BANDLO[$KT]} BAND_HI=${BANDHI[$KT]} \
    python3 experiments/calibrate_lambda_adaptive.py > "$R/calib_k$KT.log" 2>&1
  cat "$R/calib_k$KT.log"
  grep '^CHOSEN_LAM' "$R/calib_k$KT.log" | awk '{print $2}' > "$R/chosen_lambda_k$KT.txt"
  LAMS[$KT]=$(awk '{print $1}' "$R/chosen_lambda_k$KT.txt")
  echo ">>> TARGET=$KT lambda=${LAMS[$KT]}"
  # calibration writes legitimately-named raw files with discarded lambdas
  echo "=== DELETE calibration artifacts (raw names) ==="
  for f in "$RR"/sae_*.pt; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    if ! [[ $b =~ $NAME_RE ]]; then rm -f "$f"; fi
  done
done

# --------------------------------------------------------- 3. train 32
echo "=== TRAIN 32 SAEs (2 targets x 2 arches x 8 seeds) at m=16384 ==="
for KT in $KTARGETS; do
  for SEED in $SEEDS; do
    for A in topk l1; do
      OUTF=$RR/sae_pythia-1.4b_L12_${A}_x8_k${KT}_s${SEED}.pt
      RAWF=$RR/sae_pythia-1.4b_L12_${A}_x8_s${SEED}.pt
      if [ -f "$OUTF" ]; then echo "  skip existing $(basename "$OUTF")"; continue; fi
      rm -f "$RAWF"
      if [ "$A" = topk ]; then
        ACTS=$TRAIN EVAL_ACTS=$EVAL EXPANSION=8 ARCH=topk K=$KT SEED=$SEED \
          STEPS=$STEPS python3 experiments/real_train_sae.py >> "$R/train.log" 2>&1
      else
        ACTS=$TRAIN EVAL_ACTS=$EVAL EXPANSION=8 ARCH=l1 LAM=${LAMS[$KT]} SEED=$SEED \
          STEPS=$STEPS python3 experiments/real_train_sae.py >> "$R/train.log" 2>&1
      fi
      mv "$RAWF" "$OUTF"
      echo "  done $(basename "$OUTF") ($(date -u +%H:%M:%S))"
    done
  done
  set +e
  if ! gcloud storage cp "$RR"/sae_*_x8_k${KT}_s*.pt "$BUCKET/round16/" 2>&1 | tail -1; then
    echo "!!! GCS UPLOAD FAILED for k$KT -- results stay on local disk."
  fi
  set -e
done

# ------------------------------------------------------ 4. the name gate
echo "=== NAME GATE (delete anything off-pattern BEFORE scoring) ==="
for f in "$RR"/sae_*.pt; do
  b=$(basename "$f")
  if ! [[ $b =~ $NAME_RE ]]; then echo "  DELETING off-pattern: $b"; rm -f "$f"; fi
done
N=$(ls "$RR"/sae_*.pt | wc -l)
echo "  $N SAEs pass the name gate (expect 32)"
if [ "$N" -ne 32 ]; then
  echo "!!! ABORT: $N != 32 SAEs after the name gate (GPT pre-lock P1.4)."
  exit 5
fi

# ------------------------------------------------------------- 5. score
echo "=== SCORE fresh cells (frozen round-16 scorer) ==="
SAES=$(ls "$RR"/sae_*.pt | tr '\n' ',' | sed 's/,$//')
WORDS=$WORDS SAES="$SAES" OUT=$R/round16_results.json THETA=0.0 TAU=0.30 \
  python3 analysis/round16_scorer.py > "$R/score.log" 2>&1
tail -3 "$R/score.log"
echo "=== SCORE interior 13b x8 cell (same scorer, separate pass/dir) ==="
ISAES=$(ls "$INT"/sae_*.pt | tr '\n' ',' | sed 's/,$//')
WORDS=$WORDS SAES="$ISAES" OUT=$R/round16_interior.json THETA=0.0 TAU=0.30 \
  python3 analysis/round16_scorer.py > "$R/score_interior.log" 2>&1
tail -3 "$R/score_interior.log"

# ----------------------------------------------------- 6. frozen verdict
echo "=== FROZEN EVALUATOR ==="
for KT in $KTARGETS; do echo "$KT ${LAMS[$KT]}"; done > "$R/chosen_lambdas.txt"
R16=$R/round16_results.json R16_INT=$R/round16_interior.json \
  R16_LAMBDAS=$R/chosen_lambdas.txt \
  python3 analysis/analyze_round16.py > "$R/results_round16.txt" 2>&1
cat "$R/results_round16.txt"

# -------------------------------------------------------- 7. provenance + ship
{ echo "lock_commit $(git rev-parse HEAD)"; date -u
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null
  python3 -m pip freeze 2>/dev/null | grep -iE '^(torch|numpy|scikit-learn)=='
} > "$R/PROVENANCE_round16.txt"
set +e
gcloud storage cp "$R/round16_results.json" "$R/round16_interior.json" \
  "$R/results_round16.txt" "$R"/chosen_lambda_k*.txt "$R/chosen_lambdas.txt" \
  "$R"/calib_k*.log "$R/train.log" "$R"/score*.log "$R/RUN_MANIFEST" \
  "$R/PROVENANCE_round16.txt" "$BUCKET/round16/" 2>&1 | tail -1
if [ $? -ne 0 ]; then
  echo "!!! FINAL GCS UPLOAD FAILED -- artifacts are ONLY on this box's disk."
  echo "!!! Do not delete this instance until they are pulled."
fi
set -e
echo "L4 R16 COMPLETE"
