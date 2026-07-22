#!/bin/bash
# Patient GPU retry for Arm A (prereg bimodality). Adapted from retry_until_gpu.sh.
# Tries: start dev-gpu (us-west1-a); else create dev-gpu-2 from dev-gpu-img-tmp
# in fallback zones. On success: push experiment, run, collect CSV+log, stop/delete.
SRC=~/sae-identifiability/experiments/prereg_bimodality_armA.py
OUT=~/sae_results/prereg_armA
log(){ echo "$(date -u +%F' '%H:%M:%S) $*" >> ~/retry_armA.log; }

run_on(){ # $1 inst, $2 zone
  sleep 50
  for i in $(seq 1 10); do
    timeout 60 gcloud compute ssh $1 --zone=$2 --quiet --command="echo ok" >/dev/null 2>&1 && break; sleep 20
  done
  timeout 60 gcloud compute scp --zone=$2 --quiet $SRC $1:/tmp/ >>~/retry_armA.log 2>&1
  timeout 90 gcloud compute ssh $1 --zone=$2 --quiet --command="sudo mkdir -p /home/reuellee_gmail_com/sae_identifiability/experiments /home/reuellee_gmail_com/sae_identifiability/results && sudo cp /tmp/prereg_bimodality_armA.py /home/reuellee_gmail_com/sae_identifiability/experiments/ && sudo chown -R reuellee_gmail_com:reuellee_gmail_com /home/reuellee_gmail_com/sae_identifiability && sudo -u reuellee_gmail_com bash -c 'cd /home/reuellee_gmail_com/sae_identifiability/experiments && rm -f done_armA.flag && nohup bash -c \"python3 prereg_bimodality_armA.py > armA.log 2>&1; touch done_armA.flag\" >/dev/null 2>&1 < /dev/null &'" >>~/retry_armA.log 2>&1
  log "Arm A launched on $1/$2"
  D=$(($(date +%s) + 7200))
  while [ $(date +%s) -lt $D ]; do
    f=$(timeout 60 gcloud compute ssh $1 --zone=$2 --quiet --command="ls /home/reuellee_gmail_com/sae_identifiability/experiments/done_armA.flag 2>/dev/null" 2>/dev/null)
    [ -n "$f" ] && break
    sleep 120
  done
  mkdir -p $OUT
  timeout 60 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/sae_identifiability/results/prereg_armA/armA_runs.csv" $OUT/ 2>>~/retry_armA.log
  timeout 60 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/sae_identifiability/experiments/armA.log" $OUT/ 2>>~/retry_armA.log
  log "collected: $(ls $OUT | tr '\n' ' ')"
  if [ "$1" = "dev-gpu" ]; then gcloud compute instances stop $1 --zone=$2 --quiet >>~/retry_armA.log 2>&1; log "dev-gpu stopped"
  else gcloud compute instances delete $1 --zone=$2 --quiet >>~/retry_armA.log 2>&1; log "$1 deleted"; fi
  log "ARM A PIPELINE COMPLETE"
  exit 0
}

log "=== Arm A patient GPU retry (4h cap, 10min cycle) ==="
DEADLINE=$(($(date +%s) + 14400))
while [ $(date +%s) -lt $DEADLINE ]; do
  gcloud compute instances start dev-gpu --zone=us-west1-a --quiet >>~/retry_armA.log 2>&1
  st=$(gcloud compute instances describe dev-gpu --zone=us-west1-a --format='value(status)' 2>/dev/null)
  if [ "$st" = "RUNNING" ]; then log "dev-gpu started (stockout cleared)"; run_on dev-gpu us-west1-a; fi
  for z in europe-west4-a us-east1-c asia-southeast1-b; do
    gcloud compute instances create dev-gpu-2 --zone=$z --machine-type=g2-standard-8 --image=dev-gpu-img-tmp --boot-disk-size=150GB --maintenance-policy=TERMINATE --quiet >>~/retry_armA.log 2>&1
    st2=$(gcloud compute instances describe dev-gpu-2 --zone=$z --format='value(status)' 2>/dev/null)
    if [ "$st2" = "RUNNING" ] || [ "$st2" = "STAGING" ]; then
      for i in $(seq 1 20); do
        st2=$(gcloud compute instances describe dev-gpu-2 --zone=$z --format='value(status)' 2>/dev/null)
        [ "$st2" = "RUNNING" ] && run_on dev-gpu-2 $z
        [ -z "$st2" ] && break
        sleep 15
      done
    fi
  done
  log "cycle done, all unavailable; sleeping 10min"
  sleep 600
done
log "RETRY DEADLINE REACHED without capacity"
