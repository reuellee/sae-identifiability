#!/bin/bash
# Patient GPU retry for PairID Arm1 (prereg bimodality). Adapted from retry_until_gpu.sh.
# Tries: start dev-gpu (us-west1-a); else create dev-gpu-2 from dev-gpu-img-tmp
# in fallback zones. On success: push experiment, run, collect CSV+log, stop/delete.
SRC=~/sae-identifiability/experiments/round8_realdata.py
SRC2=~/sae-identifiability/experiments/round8_synthetic.py
DEP=~/sae-identifiability/experiments/prereg_bimodality_armA.py
DEP2=~/sae-identifiability/experiments/extract_activations.py
OUT=~/sae_results/round8
log(){ echo "$(date -u +%F' '%H:%M:%S) $*" >> ~/retry_round8.log; }

run_on(){ # $1 inst, $2 zone
  sleep 50
  for i in $(seq 1 10); do
    timeout 60 gcloud compute ssh $1 --zone=$2 --quiet --command="echo ok" >/dev/null 2>&1 && break; sleep 20
  done
  timeout 60 gcloud compute scp --zone=$2 --quiet $SRC $SRC2 $DEP $DEP2 $1:/tmp/ >>~/retry_round8.log 2>&1
  timeout 90 gcloud compute ssh $1 --zone=$2 --quiet --command="sudo mkdir -p /home/reuellee_gmail_com/sae_identifiability/experiments /home/reuellee_gmail_com/sae_identifiability/results && sudo cp /tmp/round8_realdata.py /tmp/round8_synthetic.py /tmp/prereg_bimodality_armA.py /tmp/extract_activations.py /home/reuellee_gmail_com/sae_identifiability/experiments/ && sudo chown -R reuellee_gmail_com:reuellee_gmail_com /home/reuellee_gmail_com/sae_identifiability && sudo -u reuellee_gmail_com bash -c 'cd /home/reuellee_gmail_com/sae_identifiability/experiments && rm -f done_r8.flag && nohup bash -c \"python3 round8_realdata.py > r8e1.log 2>&1; python3 round8_synthetic.py > r8syn.log 2>&1; touch done_r8.flag\" >/dev/null 2>&1 < /dev/null &'" >>~/retry_round8.log 2>&1
  log "PairID Arm1 launched on $1/$2"
  D=$(($(date +%s) + 10800))
  while [ $(date +%s) -lt $D ]; do
    f=$(timeout 60 gcloud compute ssh $1 --zone=$2 --quiet --command="ls /home/reuellee_gmail_com/sae_identifiability/experiments/done_r8.flag 2>/dev/null" 2>/dev/null)
    [ -n "$f" ] && break
    sleep 120
  done
  mkdir -p $OUT
  timeout 60 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/sae_identifiability/results/round8/r8e1_runs.csv" $OUT/
  timeout 120 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/sae_identifiability/results/round8/r8syn_runs.csv" $OUT/
  timeout 120 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/sae_identifiability/results/round8/weights_r8e1_m128.pt" $OUT/ 2>>~/retry_round8.log
  timeout 120 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/sae_identifiability/results/round8/weights_r8e1_m256.pt" $OUT/ 2>>~/retry_round8.log
  timeout 60 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/sae_identifiability/experiments/r8e1.log" $OUT/ 2>>~/retry_round8.log
  timeout 60 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/sae_identifiability/experiments/r8syn.log" $OUT/ 2>>~/retry_round8.log
  timeout 240 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/sae_identifiability/experiments/activations_l6.pt" ~/sae-identifiability/experiments/ 2>>~/retry_round8.log 2>>~/retry_round8.log
  log "collected: $(ls $OUT | tr '\n' ' ')"
  if [ "$1" = "dev-gpu" ]; then gcloud compute instances stop $1 --zone=$2 --quiet >>~/retry_round8.log 2>&1; log "dev-gpu stopped"
  else gcloud compute instances delete $1 --zone=$2 --quiet >>~/retry_round8.log 2>&1; log "$1 deleted"; fi
  log "ARM A PIPELINE COMPLETE"
  exit 0
}

log "=== PairID Arm1 patient GPU retry (4h cap, 10min cycle) ==="
DEADLINE=$(($(date +%s) + 14400))
while [ $(date +%s) -lt $DEADLINE ]; do
  gcloud compute instances start dev-gpu --zone=us-west1-a --quiet >>~/retry_round8.log 2>&1
  st=$(gcloud compute instances describe dev-gpu --zone=us-west1-a --format='value(status)' 2>/dev/null)
  if [ "$st" = "RUNNING" ]; then log "dev-gpu started (stockout cleared)"; run_on dev-gpu us-west1-a; fi
  for z in europe-west4-a us-east1-c asia-southeast1-b; do
    gcloud compute instances create dev-gpu-2 --zone=$z --machine-type=g2-standard-8 --image=dev-gpu-img-tmp --boot-disk-size=150GB --maintenance-policy=TERMINATE --quiet >>~/retry_round8.log 2>&1
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
