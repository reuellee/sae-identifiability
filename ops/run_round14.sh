#!/usr/bin/env bash
# Round 14 driver: carrier analysis on frozen weights, CPU-only ephemeral VM.
# Prereg: notes/prereg-round14-carrier.md (LOCKED at 2a81a98, before any r14 number).
#
# NOTE vs the prereg's Ops paragraph: that paragraph says activations come from
# acts_eval.pt. They do not -- analysis/round14_carrier.py reads only the WORDS file
# (X + letters), exactly as the 13a/13b scorers do. So staging is ~150MB, not 1.5GB.
# The prereg is left unedited (it is locked); this is an ops correction only and
# changes no registered quantity.
#
# Ops recipe (from the 13a re-score): the orchestrator has no pip/torch, so this runs
# on an `ephemeral` VM. Those VMs are created --no-service-account --no-scopes and
# therefore CANNOT reach GCS, so inputs are fetched HERE with user ADC and pushed in
# with `gcloud compute scp`. Do NOT put a user ADC token on the VM.
set -uo pipefail

NAME=${NAME:-r14}
TYPE=${TYPE:-e2-standard-8}
REPO=${REPO:-$HOME/sae-identifiability}
STAGE=${STAGE:-$HOME/r14_stage}
WEIGHTS=${WEIGHTS:-$HOME/r13b_pull}
ZONE=${ZONE:-us-west1-a}
PROJECT=${PROJECT:-project-ebd5a273-53ea-4c8b-81a}
# Registered cells: m=16384 (primary) and m=2048 (capacity contrast). NOT m=4096.
GLOB=${GLOB:-'sae_pythia-1.4b_L12_*_x8_s*.pt sae_pythia-1.4b_L12_*_x1_s*.pt'}

log() { echo "$(date -u +%H:%M:%S) $*"; }
vssh() { timeout 300 gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" \
           -- -o StrictHostKeyChecking=no "$1" 2>&1; }

mkdir -p "$STAGE"
cd "$REPO" || exit 2

# ---------------------------------------------------------------- 1. stage inputs
if [ ! -s "$STAGE/words_pythia-1.4b_L12.pt" ]; then
  log "fetching words file from GCS (user ADC)"
  ./ops/gcs_adc.sh get round12/words_pythia-1.4b_L12.pt \
    "$STAGE/words_pythia-1.4b_L12.pt" || exit 1
fi
N=0
for g in $GLOB; do
  for f in $WEIGHTS/$g; do [ -s "$f" ] && N=$((N+1)); done
done
log "staging $N weight files (expect 32: 16 at m=16384 + 16 at m=2048)"
[ "$N" -eq 32 ] || { log "WRONG WEIGHT COUNT ($N) -- aborting before spending VM time"; exit 1; }

# ---------------------------------------------------------------- 2. make the VM
if ! gcloud compute instances describe "$NAME" --zone="$ZONE" --project="$PROJECT" \
       --format='value(status)' >/dev/null 2>&1; then
  log "creating ephemeral VM $NAME ($TYPE)"
  ephemeral new "$NAME" "$TYPE" || exit 1
fi
for i in $(seq 1 30); do
  [ -n "$(vssh 'echo up' | grep -o up)" ] && break
  sleep 10
done
log "VM reachable"

vssh 'sudo apt-get update -qq >/dev/null 2>&1; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip >/dev/null 2>&1; pip3 -q install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1; pip3 -q install --break-system-packages scikit-learn numpy 2>&1 | tail -1; python3 -c "import torch,sklearn;print(torch.__version__, sklearn.__version__)"'

# ---------------------------------------------------------------- 3. push inputs
vssh "mkdir -p ~/r14/w"
log "copying words file"
gcloud compute scp --project="$PROJECT" --zone="$ZONE" --internal-ip \
  "$STAGE/words_pythia-1.4b_L12.pt" "$NAME:~/r14/" >/dev/null 2>&1 || exit 1
log "copying analysis code"
gcloud compute scp --project="$PROJECT" --zone="$ZONE" --internal-ip \
  analysis/round14_carrier.py analysis/analyze_round14.py "$NAME:~/r14/" >/dev/null 2>&1
log "copying weights (~4.8GB, this is the slow part)"
for g in $GLOB; do
  for f in $WEIGHTS/$g; do
    b=$(basename "$f")
    vssh "test -s ~/r14/w/$b" | grep -q . && continue
    gcloud compute scp --project="$PROJECT" --zone="$ZONE" --internal-ip \
      "$f" "$NAME:~/r14/w/" >/dev/null 2>&1 || log "  scp failed: $b"
  done
done
log "on VM: $(vssh 'ls ~/r14/w/*.pt | wc -l') weights"

# ---------------------------------------------------------------- 4. run detached
log "launching scorer (detached; ssh will return immediately)"
vssh 'cd ~/r14 && SAES=$(ls ~/r14/w/*.pt | tr "\n" "," | sed "s/,$//") \
      WORDS=$HOME/r14/words_pythia-1.4b_L12.pt OUT=$HOME/r14/round14_results.json \
      nohup setsid python3 round14_carrier.py > $HOME/r14/score.log 2>&1 < /dev/null & sleep 2; echo started'
log "done. poll:  gcloud compute ssh $NAME --zone $ZONE -- 'tail -5 ~/r14/score.log'"
log "when finished, collect and evaluate with ops/collect_round14.sh"
