#!/usr/bin/env bash
# Collect round-13b artifacts from the L4 -> orchestrator -> GCS, incrementally.
#
# WHY THIS EXISTS (ops bug found 2026-07-25, mid-run):
# dev-gpu was rebuilt from a snapshot and now runs as the DEFAULT COMPUTE SA
# (159398774377-compute@developer.gserviceaccount.com) with insufficient scopes:
#   "Provided scope(s) are not authorized" on gs://sae-identifiability-artifacts-*
# So every `gcloud storage cp` inside ops/l4_r13b.sh fails -- and because those
# lines are wrapped in `set +e` with `| tail -1`, the failure is INVISIBLE in the
# driver log. After ~9h of training, gs://.../round13b/ was still empty.
# Scopes cannot be changed while the instance is running, so the fix is to PULL:
#   - `gcloud compute scp` works (the orchestrator SA does have compute access)
#   - GCS writes then go through USER ADC via ops/gcs_adc.sh
#     (never `gcloud storage`/gsutil here -- those pick up the orchestrator SA,
#      which has zero bucket access; that was round 12's collection bug)
# Deliberately does NOT copy a user ADC token onto the VM.
#
# Runs incrementally so a stockout / poweroff can never strand a finished SAE:
# every file is shipped as soon as it appears, not in one batch at the end.
set -uo pipefail

PROJECT=${PROJECT:-project-ebd5a273-53ea-4c8b-81a}
ZONE=${ZONE:-us-west1-a}
VM=${VM:-dev-gpu}
RHOME=${RHOME:-/home/sa_110201476474221697918}
RR=$RHOME/r13b/sae-identifiability/results/real
RDIR=$RHOME/r13b
LOCAL=${LOCAL:-$HOME/r13b_pull}
REPO=${REPO:-$HOME/sae-identifiability}
INTERVAL=${INTERVAL:-300}
EXPECTED=${EXPECTED:-48}

mkdir -p "$LOCAL"
log() { echo "$(date -u +%H:%M:%S) $*"; }

rssh() { timeout 120 gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" \
           -- -o StrictHostKeyChecking=no "$1" 2>/dev/null; }

pull() { # pull REMOTE_ABS -> $LOCAL, then push to GCS under round13b/
  local rpath="$1" base
  base=$(basename "$rpath")
  if [ ! -s "$LOCAL/$base" ]; then
    timeout 600 gcloud compute scp --project="$PROJECT" --zone="$ZONE" --internal-ip \
      "$VM:$rpath" "$LOCAL/" >/dev/null 2>&1 || { log "  SCP FAILED $base"; return 1; }
  fi
  if ! grep -qxF "$base" "$LOCAL/.uploaded" 2>/dev/null; then
    if "$REPO/ops/gcs_adc.sh" put "$LOCAL/$base" "round13b/$base" >/dev/null 2>&1; then
      echo "$base" >> "$LOCAL/.uploaded"; log "  shipped $base"
    else
      log "  GCS PUT FAILED $base"; return 1
    fi
  fi
  return 0
}

log "collector start (expect $EXPECTED SAEs)"
while :; do
  # --- weights that exist on the box right now
  for f in $(rssh "ls $RR/sae_*.pt 2>/dev/null"); do pull "$f"; done
  n=$(ls "$LOCAL"/sae_*.pt 2>/dev/null | wc -l)

  # --- has the driver finished?
  if [ -n "$(rssh "ls $RDIR/DRIVER_EXITED 2>/dev/null")" ]; then
    log "DRIVER_EXITED seen; final sweep ($n/$EXPECTED weights local)"
    for f in $(rssh "ls $RR/sae_*.pt 2>/dev/null"); do pull "$f"; done
    for b in round13b_results.json results_round13b.txt chosen_lambda_by_width.txt \
             train.log drive.log score.log calib_x1.log calib_x2.log calib_x8.log \
             watchdog.log; do
      [ -n "$(rssh "ls $RDIR/$b 2>/dev/null")" ] && pull "$RDIR/$b"
    done
    n=$(ls "$LOCAL"/sae_*.pt 2>/dev/null | wc -l)
    log "final: $n/$EXPECTED weights, results present: $(ls "$LOCAL"/results_round13b.txt 2>/dev/null | wc -l)"
    # Only release the box once every expected weight is safely in GCS.
    if [ "$n" -ge "$EXPECTED" ] && [ -s "$LOCAL/round13b_results.json" ]; then
      rssh "touch $RDIR/RETRIEVED"; log "RETRIEVED marker set - box may power off"
    else
      log "INCOMPLETE - leaving box up for manual inspection"
    fi
    break
  fi
  log "in flight: $n/$EXPECTED weights collected"
  sleep "$INTERVAL"
done
log "collector done"
