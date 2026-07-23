#!/bin/bash
# Patient GPU retry for round 9 (gating-corrected rho estimators).
# Runs ON THE ORCHESTRATOR. Tries: start dev-gpu (us-west1-a); else create
# dev-gpu-2 from dev-gpu-img-tmp in fallback zones. On success: push the
# round-9 files + the round-8 activation bank (reused for protocol
# fidelity), SMOKE-gate, run full suite, collect CSV+weights+logs back to
# ~/sae-identifiability/results/round9/, stop/delete the box.
# LAUNCH ONLY AFTER THE PREREG LOCK COMMIT (prereg pre-lock phase).
REPO=~/sae-identifiability
OUT=$REPO/results/round9
log(){ echo "$(date -u +%F' '%H:%M:%S) $*" >> ~/retry_round9.log; }

run_on(){ # $1 inst, $2 zone
  sleep 50
  for i in $(seq 1 12); do
    timeout 60 gcloud compute ssh $1 --zone=$2 --quiet --command="echo ok" >/dev/null 2>&1 && break; sleep 20
  done
  timeout 90 gcloud compute scp --zone=$2 --quiet \
    $REPO/experiments/round9_rho_estimator.py \
    $REPO/experiments/round8_synthetic.py \
    $REPO/experiments/prereg_bimodality_armA.py \
    $REPO/experiments/extract_activations.py \
    $REPO/analysis/rho_estimators_lib.py \
    $1:/tmp/ >>~/retry_round9.log 2>&1
  timeout 600 gcloud compute scp --zone=$2 --quiet \
    $REPO/experiments/activations_l6.pt $1:/tmp/ >>~/retry_round9.log 2>&1
  timeout 120 gcloud compute ssh $1 --zone=$2 --quiet --command="sudo mkdir -p /home/reuellee_gmail_com/round9/experiments /home/reuellee_gmail_com/round9/analysis /home/reuellee_gmail_com/round9/results && sudo cp /tmp/round9_rho_estimator.py /tmp/round8_synthetic.py /tmp/prereg_bimodality_armA.py /tmp/extract_activations.py /tmp/activations_l6.pt /home/reuellee_gmail_com/round9/experiments/ && sudo cp /tmp/rho_estimators_lib.py /home/reuellee_gmail_com/round9/analysis/ && sudo chown -R reuellee_gmail_com:reuellee_gmail_com /home/reuellee_gmail_com/round9 && sudo -u reuellee_gmail_com bash -c 'cd /home/reuellee_gmail_com/round9/experiments && rm -f done_r9.flag fail_r9.flag && nohup bash -c \"if SMOKE=1 python3 round9_rho_estimator.py > smoke_r9.log 2>&1; then python3 round9_rho_estimator.py > r9.log 2>&1 && touch done_r9.flag || touch fail_r9.flag; else touch fail_r9.flag; fi\" >/dev/null 2>&1 < /dev/null &'" >>~/retry_round9.log 2>&1
  log "round 9 launched on $1/$2 (SMOKE-gated)"
  D=$(($(date +%s) + 18000))
  mkdir -p $OUT
  while [ $(date +%s) -lt $D ]; do
    f=$(timeout 60 gcloud compute ssh $1 --zone=$2 --quiet --command="ls /home/reuellee_gmail_com/round9/experiments/done_r9.flag /home/reuellee_gmail_com/round9/experiments/fail_r9.flag 2>/dev/null" 2>/dev/null)
    [ -n "$f" ] && { log "flag: $f"; break; }
    # preemption safety: pull the incremental CSV each cycle; detect vanish
    timeout 120 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/round9/results/round9/r9_runs.csv" $OUT/r9_runs.partial.csv 2>/dev/null
    inst=$(gcloud compute instances describe $1 --zone=$2 --format='value(status)' 2>/dev/null)
    [ -z "$inst" ] && { log "$1 vanished (spot preemption?) - resuming acquisition"; return 1; }
    sleep 180
  done
  mkdir -p $OUT
  for f in r9_runs.csv weights_r9_syn.pt weights_r9_real_q04.pt weights_r9_real_q12.pt weights_r9_real_q20.pt weights_r9_real_q28.pt; do
    timeout 240 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/round9/results/round9/$f" $OUT/ 2>>~/retry_round9.log
  done
  timeout 60 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/round9/experiments/r9.log" $OUT/ 2>>~/retry_round9.log
  timeout 60 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/round9/experiments/smoke_r9.log" $OUT/ 2>>~/retry_round9.log
  log "collected: $(ls $OUT | tr '\n' ' ')"
  if [ "$1" = "dev-gpu" ]; then gcloud compute instances stop $1 --zone=$2 --quiet >>~/retry_round9.log 2>&1; log "dev-gpu stopped"
  else gcloud compute instances delete $1 --zone=$2 --quiet >>~/retry_round9.log 2>&1; log "$1 deleted"; fi
  log "ROUND 9 PIPELINE COMPLETE"
  exit 0
}

try_create(){ # $1 zone, $2... extra create flags
  local z=$1; shift
  gcloud compute instances create dev-gpu-2 --zone=$z --image=dev-gpu-img-tmp \
    --boot-disk-size=150GB --quiet "$@" >>~/retry_round9.log 2>&1
  st2=$(gcloud compute instances describe dev-gpu-2 --zone=$z --format='value(status)' 2>/dev/null)
  if [ "$st2" = "RUNNING" ] || [ "$st2" = "STAGING" ]; then
    log "dev-gpu-2 created in $z ($*)"; run_on dev-gpu-2 $z
  fi
  gcloud compute instances delete dev-gpu-2 --zone=$z --quiet >/dev/null 2>&1
}

log "=== round 9 patient GPU retry (6h cap; spot-first per user authorization) ==="
DEADLINE=$(($(date +%s) + 21600))
while [ $(date +%s) -lt $DEADLINE ]; do
  # 1. Spot L4 (ephemeral, cheapest ~\$0.30/hr, separate capacity pool)
  for z in us-west1-a us-west1-b us-central1-a us-east1-c us-east1-d us-east4-a europe-west4-a asia-southeast1-b; do
    try_create $z --machine-type=g2-standard-8 --provisioning-model=SPOT --instance-termination-action=DELETE
  done
  # 2. On-demand L4: named box, then fallback zones
  gcloud compute instances start dev-gpu --zone=us-west1-a --quiet >>~/retry_round9.log 2>&1
  st=$(gcloud compute instances describe dev-gpu --zone=us-west1-a --format='value(status)' 2>/dev/null)
  if [ "$st" = "RUNNING" ]; then log "dev-gpu started (stockout cleared)"; run_on dev-gpu us-west1-a; fi
  for z in europe-west4-a us-east1-c asia-southeast1-b; do
    try_create $z --machine-type=g2-standard-8 --maintenance-policy=TERMINATE
  done
  # 3. Spot A100 (fast; only if quota permits - failure is logged and skipped)
  for z in us-central1-a us-central1-c us-east1-b; do
    try_create $z --machine-type=a2-highgpu-1g --provisioning-model=SPOT --instance-termination-action=DELETE
  done
  # 4. T4 (widely available, ~2-3x slower; last resort)
  for z in us-west1-b us-central1-a us-east1-c; do
    try_create $z --machine-type=n1-standard-8 --accelerator=type=nvidia-tesla-t4,count=1 --maintenance-policy=TERMINATE
  done
  log "all options exhausted this cycle; sleeping 600s"
  sleep 600
done
log "DEADLINE reached without GPU"
