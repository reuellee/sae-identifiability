#!/bin/bash
# Patient spot-first GPU retry for round 10 (TopK absorption). Runs ON THE
# ORCHESTRATOR. round10_topk.py is self-contained (torch+numpy only; no
# activation bank). SMOKE-gates, runs the full grid, collects CSV + all
# weights_r10_*.pt, stops/deletes the box. LAUNCH ONLY AFTER THE PREREG LOCK.
REPO=~/sae-identifiability
OUT=$REPO/results/round10
log(){ echo "$(date -u +%F' '%H:%M:%S) $*" >> ~/retry_round10.log; }

run_on(){ # $1 inst, $2 zone
  sleep 50
  for i in $(seq 1 12); do
    timeout 60 gcloud compute ssh $1 --zone=$2 --quiet --command="echo ok" >/dev/null 2>&1 && break; sleep 20
  done
  timeout 90 gcloud compute scp --zone=$2 --quiet \
    $REPO/experiments/round10_topk.py $1:/tmp/ >>~/retry_round10.log 2>&1
  timeout 120 gcloud compute ssh $1 --zone=$2 --quiet --command="sudo mkdir -p /home/reuellee_gmail_com/round10/experiments /home/reuellee_gmail_com/round10/results && sudo cp /tmp/round10_topk.py /home/reuellee_gmail_com/round10/experiments/ && sudo chown -R reuellee_gmail_com:reuellee_gmail_com /home/reuellee_gmail_com/round10 && sudo -u reuellee_gmail_com bash -c 'cd /home/reuellee_gmail_com/round10/experiments && rm -f done_r10.flag fail_r10.flag && nohup bash -c \"if SMOKE=1 python3 round10_topk.py > smoke_r10.log 2>&1; then python3 round10_topk.py > r10.log 2>&1 && touch done_r10.flag || touch fail_r10.flag; else touch fail_r10.flag; fi\" >/dev/null 2>&1 < /dev/null &'" >>~/retry_round10.log 2>&1
  log "round 10 launched on $1/$2 (SMOKE-gated)"
  D=$(($(date +%s) + 14400))
  mkdir -p $OUT
  while [ $(date +%s) -lt $D ]; do
    f=$(timeout 60 gcloud compute ssh $1 --zone=$2 --quiet --command="ls /home/reuellee_gmail_com/round10/experiments/done_r10.flag /home/reuellee_gmail_com/round10/experiments/fail_r10.flag 2>/dev/null" 2>/dev/null)
    [ -n "$f" ] && { log "flag: $f"; break; }
    timeout 120 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/round10/results/round10/r10_runs.csv" $OUT/r10_runs.partial.csv 2>/dev/null
    inst=$(gcloud compute instances describe $1 --zone=$2 --format='value(status)' 2>/dev/null)
    [ -z "$inst" ] && { log "$1 vanished (spot preemption?) - resuming acquisition"; return 1; }
    sleep 150
  done
  timeout 120 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/round10/results/round10/r10_runs.csv" $OUT/ 2>>~/retry_round10.log
  timeout 300 gcloud compute scp --zone=$2 --recurse --quiet "$1:/home/reuellee_gmail_com/round10/results/round10/*.pt" $OUT/ 2>>~/retry_round10.log
  timeout 60 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/round10/experiments/r10.log" $OUT/ 2>>~/retry_round10.log
  timeout 60 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/round10/experiments/smoke_r10.log" $OUT/ 2>>~/retry_round10.log
  log "collected: $(ls $OUT | tr '\n' ' ')"
  if [ "$1" = "dev-gpu" ]; then gcloud compute instances stop $1 --zone=$2 --quiet >>~/retry_round10.log 2>&1; log "dev-gpu stopped"
  else gcloud compute instances delete $1 --zone=$2 --quiet >>~/retry_round10.log 2>&1; log "$1 deleted"; fi
  log "ROUND 10 PIPELINE COMPLETE"
  exit 0
}

try_create(){ # $1 zone, $2... extra create flags
  local z=$1; shift
  gcloud compute instances create dev-gpu-2 --zone=$z --image=dev-gpu-img-tmp \
    --boot-disk-size=150GB --quiet "$@" >>~/retry_round10.log 2>&1
  st2=$(gcloud compute instances describe dev-gpu-2 --zone=$z --format='value(status)' 2>/dev/null)
  if [ "$st2" = "RUNNING" ] || [ "$st2" = "STAGING" ]; then
    log "dev-gpu-2 created in $z ($*)"; run_on dev-gpu-2 $z
  fi
  gcloud compute instances delete dev-gpu-2 --zone=$z --quiet >/dev/null 2>&1
}

log "=== round 10 patient GPU retry (6h cap; spot-first) ==="
DEADLINE=$(($(date +%s) + 21600))
while [ $(date +%s) -lt $DEADLINE ]; do
  for z in us-west1-a us-west1-b us-central1-a us-east1-c us-east1-d us-east4-a europe-west4-a asia-southeast1-b; do
    try_create $z --machine-type=g2-standard-8 --provisioning-model=SPOT --instance-termination-action=DELETE
  done
  gcloud compute instances start dev-gpu --zone=us-west1-a --quiet >>~/retry_round10.log 2>&1
  st=$(gcloud compute instances describe dev-gpu --zone=us-west1-a --format='value(status)' 2>/dev/null)
  if [ "$st" = "RUNNING" ]; then log "dev-gpu started (stockout cleared)"; run_on dev-gpu us-west1-a; fi
  for z in europe-west4-a us-east1-c asia-southeast1-b; do
    try_create $z --machine-type=g2-standard-8 --maintenance-policy=TERMINATE
  done
  for z in us-west1-b us-central1-a us-east1-c; do
    try_create $z --machine-type=n1-standard-8 --accelerator=type=nvidia-tesla-t4,count=1 --maintenance-policy=TERMINATE
  done
  log "all options exhausted this cycle; sleeping 600s"
  sleep 600
done
log "DEADLINE reached without GPU"
