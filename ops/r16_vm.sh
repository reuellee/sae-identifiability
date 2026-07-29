#!/usr/bin/env bash
# Round 16 VM lifecycle (laptop-driven, Git Bash + plink quirks as in r15).
# GPU box: g2-standard-8 (1x L4). Created WITH the default SA + storage-rw
# scope AT CREATE TIME (13b lesson: a snapshot-rebuilt box cannot get scopes),
# so the box pulls caches from GCS and pushes artifacts back itself.
# Spot-first with on-demand fallback, zone hunt over the L4 pools.
# Staged: setup | deps | push | run | status | collect | clean
set -uo pipefail

NAME=${NAME:-r16}
ZONE_FILE=/tmp/r16_zone
PROJECT=${PROJECT:-project-ebd5a273-53ea-4c8b-81a}
REPO=${REPO:-/e/Projects/sae-identifiability}
LOCK_SHA=${LOCK_SHA:-}
ZONES=${ZONES:-"us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-c us-west1-a us-west1-b us-west4-a europe-west4-a europe-west4-b"}
ZONE=$(cat "$ZONE_FILE" 2>/dev/null || echo "")

log() { echo "$(date -u +%H:%M:%S) $*"; }
vssh() { gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --command="$1" 2>&1; }
vscp() { gcloud compute scp --project="$PROJECT" --zone="$ZONE" "$@" 2>&1 | tail -1; }

create_in() { # $1 zone, $2 spot|ondemand
  local extra=""
  [ "$2" = spot ] && extra="--provisioning-model=SPOT --instance-termination-action=STOP"
  gcloud compute instances create "$NAME" --project="$PROJECT" --zone="$1" \
    --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
    --image-family=common-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --metadata=install-nvidia-driver=True \
    --boot-disk-size=150GB --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE --scopes=storage-rw $extra 2>&1
}

case "${1:-}" in
setup)
  for MODE in spot ondemand; do
    for Z in $ZONES; do
      log "trying $MODE in $Z ..."
      if OUT=$(create_in "$Z" "$MODE"); then
        echo "$Z" > "$ZONE_FILE"
        log "CREATED $NAME ($MODE) in $Z"
        echo "$OUT" | tail -2
        exit 0
      fi
      echo "$OUT" | grep -qiE "quota|not have enough resources|ZONE_RESOURCE_POOL_EXHAUSTED|stockout" \
        || { echo "$OUT" | tail -3; log "non-capacity error; aborting zone hunt"; exit 1; }
    done
    log "$MODE exhausted in all zones"
  done
  log "NO L4 CAPACITY ANYWHERE (registered fallback: the round pauses)"
  exit 1
  ;;
deps)
  echo y | vssh 'echo ok' >/dev/null   # plink first-contact host key
  vssh 'nvidia-smi --query-gpu=name --format=csv,noheader
        python3 --version
        pip3 -q install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1
        pip3 -q install scikit-learn numpy 2>&1 | tail -1
        python3 -c "import torch;print(torch.__version__, torch.cuda.is_available())"'
  ;;
push)
  [ -n "$LOCK_SHA" ] || { log "set LOCK_SHA=<lock commit>"; exit 2; }
  vssh "rm -rf ~/r16 && mkdir -p ~/r16 && cd ~/r16 \
        && git clone -q https://github.com/reuellee/sae-identifiability.git \
        && cd sae-identifiability && git checkout -q $LOCK_SHA \
        && chmod +x ops/l4_r16.sh ops/vm_watchdog.sh && git rev-parse HEAD"
  ;;
run)
  vssh 'cd ~/r16/sae-identifiability
        nohup setsid env R=$HOME/r16 bash ops/l4_r16.sh > ~/r16/drive.log 2>&1 < /dev/null &
        PID=$!
        echo $PID > ~/r16/job.pid
        nohup setsid bash ops/vm_watchdog.sh $PID 26 3600 > /dev/null 2>&1 < /dev/null &
        echo "launched pid $PID (watchdog: 26h cap, 1h post-exit grace)"'
  ;;
status)
  vssh 'echo "job pid: $(cat ~/r16/job.pid 2>/dev/null) alive: $(kill -0 $(cat ~/r16/job.pid 2>/dev/null) 2>/dev/null && echo yes || echo no)"
        tail -6 ~/r16/drive.log 2>/dev/null
        echo "--- trained: $(ls ~/r16/sae-identifiability/results/real/sae_*_k*_s*.pt 2>/dev/null | wc -l)/32"
        tail -2 ~/r16/train.log 2>/dev/null'
  ;;
collect)
  mkdir -p "$REPO/results/real"
  for f in round16_results.json round16_interior.json results_round16.txt chosen_lambdas.txt PROVENANCE_round16.txt; do
    vscp "$NAME:r16/$f" "$REPO/results/real/" || log "missing: $f"
  done
  vscp "$NAME:r16/drive.log" "$REPO/results/real/round16_drive.log" || true
  log "collected (artifacts also in GCS round16/ if the box's uploads succeeded)."
  ;;
clean)
  gcloud compute instances delete "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet
  ;;
*)
  echo "usage: $0 setup|deps|push|run|status|collect|clean"; exit 2
  ;;
esac
