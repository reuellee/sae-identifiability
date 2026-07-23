#!/bin/bash
# Runs ON the orchestrator. Spot-first-acquire an L4, have it git-clone the
# public repo and run ops/l4_real_pipeline.sh (SMOKE -> extract -> train ->
# upload big artifacts to GCS), collect the small logs/stats, delete the box.
# Big artifacts (acts, SAE weights) go to gs://.../round11/ (too big for git).
OUT=~/sae-identifiability/results/real
REPO=https://github.com/reuellee/sae-identifiability
log(){ echo "$(date -u +%F' '%H:%M:%S) $*" >> ~/run_real_l4.log; }
mkdir -p $OUT

run_on(){ # $1 inst, $2 zone
  sleep 50
  for i in $(seq 1 15); do
    timeout 60 gcloud compute ssh $1 --zone=$2 --quiet --command="echo ok" >/dev/null 2>&1 && break; sleep 20
  done
  # clone + launch as reuellee_gmail_com (the account with torch/CUDA)
  timeout 180 gcloud compute ssh $1 --zone=$2 --quiet --command="sudo -u reuellee_gmail_com bash -lc 'cd ~ && rm -rf r11 r11_done && git clone --depth 1 -q $REPO r11 && chmod +x r11/ops/l4_real_pipeline.sh && nohup bash r11/ops/l4_real_pipeline.sh > ~/r11_run.log 2>&1 & echo launched'" >>~/run_real_l4.log 2>&1
  log "pipeline launched on $1/$2"
  D=$(($(date +%s) + 14400))
  while [ $(date +%s) -lt $D ]; do
    f=$(timeout 60 gcloud compute ssh $1 --zone=$2 --quiet --command="sudo -u reuellee_gmail_com bash -lc 'ls ~/r11_done 2>/dev/null'" 2>/dev/null)
    [ -n "$f" ] && { log "done flag seen"; break; }
    inst=$(gcloud compute instances describe $1 --zone=$2 --format='value(status)' 2>/dev/null)
    [ -z "$inst" ] && { log "$1 vanished (spot preemption?)"; return 1; }
    # pull the in-progress run log for visibility
    timeout 60 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/r11_run.log" $OUT/ 2>/dev/null
    sleep 150
  done
  # collect small logs + stats (big artifacts already in GCS)
  for f in r11_run.log r11/logs_extract.log r11/logs_train_topk.log r11/logs_train_l1.log r11/stats_summary.txt r11/logs_smoke_ext.log r11/logs_smoke_tr.log; do
    timeout 60 gcloud compute scp --zone=$2 --quiet "$1:/home/reuellee_gmail_com/$f" $OUT/ 2>>~/run_real_l4.log
  done
  log "collected: $(ls $OUT | tr '\n' ' ')"
  gcloud compute instances delete $1 --zone=$2 --quiet >>~/run_real_l4.log 2>&1
  log "$1 deleted; REAL PIPELINE COMPLETE"
  touch ~/run_real_l4_done
  exit 0
}

try_create(){ # $1 zone
  gcloud compute instances create dev-gpu-2 --zone=$1 --image=dev-gpu-img-tmp \
    --boot-disk-size=200GB --machine-type=g2-standard-8 --provisioning-model=SPOT \
    --instance-termination-action=DELETE --scopes=cloud-platform --quiet \
    >>~/run_real_l4.log 2>&1
  st=$(gcloud compute instances describe dev-gpu-2 --zone=$1 --format='value(status)' 2>/dev/null)
  if [ "$st" = "RUNNING" ] || [ "$st" = "STAGING" ]; then
    log "dev-gpu-2 spot L4 created in $1"; run_on dev-gpu-2 $1
  fi
  gcloud compute instances delete dev-gpu-2 --zone=$1 --quiet >/dev/null 2>&1
}

log "=== real-model pipeline: spot L4 acquire (6h cap) ==="
DEADLINE=$(($(date +%s) + 21600))
while [ $(date +%s) -lt $DEADLINE ]; do
  for z in us-central1-a us-central1-b us-west1-a us-west1-b us-east1-c us-east1-d us-east4-a europe-west4-a; do
    try_create $z
  done
  # on-demand fallback (named dev-gpu box)
  gcloud compute instances start dev-gpu --zone=us-west1-a --quiet >>~/run_real_l4.log 2>&1
  st=$(gcloud compute instances describe dev-gpu --zone=us-west1-a --format='value(status)' 2>/dev/null)
  if [ "$st" = "RUNNING" ]; then
    log "dev-gpu on-demand started"
    gcloud compute instances add-metadata dev-gpu --zone=us-west1-a --metadata=x=1 --quiet >/dev/null 2>&1
    run_on dev-gpu us-west1-a
  fi
  log "no L4 this cycle; sleep 600"
  sleep 600
done
log "DEADLINE without L4"
