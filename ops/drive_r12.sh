#!/bin/bash
# Runs ON the orchestrator (nohup). Deterministic backstop that carries round 12
# to completion after the laptop session closes: launch the corrected RESUME on
# the L4 (dev-gpu-2), wait, collect results from GCS, commit a faithful SUMMARY
# (the frozen scorer's verdict verbatim), push, and STOP the L4.
# The frozen scorer decides CONFIRM/FALSIFIED/NOT-CONFIRMED honestly, so the
# committed result is faithful without further judgment.
set -u
Z=us-east1-b; L4=dev-gpu-2
BUCKET=gs://sae-identifiability-artifacts-ebd5a273/round12
cd ~/sae-identifiability
log(){ echo "$(date -u +%F' '%H:%M:%S) $*" | tee -a ~/drive_r12.log; }
SSHL4(){ timeout 120 gcloud compute ssh $L4 --zone=$Z --quiet --command="$1" 2>/dev/null; }

log "=== drive_r12 start ==="
git pull -q || log "repo pull failed"

# 1. update the L4's ~/r12 code (keep caches) and launch the resume
log "launching corrected resume on $L4"
SSHL4 "cd ~/r12 && git pull -q && chmod +x ops/l4_r12_resume.sh && nohup bash ops/l4_r12_resume.sh > ~/r12_run.log 2>&1 & echo launched"

# 2. wait for completion (r12_done) or death
log "waiting for L4 (poll 5min, cap ~10h)"
DONE=0
for i in $(seq 1 120); do
  sleep 300
  S=$(SSHL4 "ls ~/r12_done 2>/dev/null && echo DONE; pgrep -f l4_r12_resume >/dev/null && echo ALIVE || echo DEAD")
  N=$(SSHL4 "grep -c '^STATS' ~/r12/logs_train.log 2>/dev/null")
  log "poll $i: $(echo $S | tr '\n' ' ') SAEs=$N"
  echo "$S" | grep -q DONE && { DONE=1; log "L4 reports DONE"; break; }
  echo "$S" | grep -q DEAD && { log "L4 resume DIED (no done flag) -- leaving box up for inspection"; break; }
done

# 3. collect results from GCS
log "collecting results from GCS"
mkdir -p results/real
gcloud storage cp "$BUCKET/results_round12.txt" "$BUCKET/stats_summary.txt" "$BUCKET/chosen_lambda.txt" . 2>&1 | tail -1
gcloud storage cp "$BUCKET/sae_pythia-1.4b_L12_*_fl.json" results/real/ 2>&1 | tail -1
gcloud storage cp "$BUCKET/sae_pythia-1.4b_L12_*_pairs.json" results/real/ 2>&1 | tail -1
LAM=$(cat chosen_lambda.txt 2>/dev/null || echo "?")

# 4. faithful SUMMARY (verbatim frozen-scorer verdict)
if [ -f results_round12.txt ]; then
  { echo "# Round 12 — real-model first-letter absorption (L1 vs TopK): RESULTS"; echo
    echo "Pre-registered and LOCKED (commit 0722212). L1 λ=$LAM, calibrated at the 15k"
    echo "training-step budget (fixing the 8k-vs-15k L0 drift from the first attempt) and"
    echo "matched to TopK's L0=32. TF32 enabled (infra; both arches identical). SAE dead%"
    echo "is reported, not re-engineered. Scored by the FROZEN analysis/analyze_round12.py;"
    echo "the verdict below is verbatim — CONFIRM / FALSIFIED / NOT-CONFIRMED as the gates yield."
    echo; echo '```'; cat results_round12.txt; echo '```'
    echo; echo "_Committed by the orchestrator drive after the laptop session closed._"
  } > results/real/SUMMARY_round12.md
  cp results_round12.txt results/real/results_round12.txt
  git add results/real/SUMMARY_round12.md results/real/results_round12.txt results/real/sae_pythia-1.4b_L12_*_fl.json results/real/sae_pythia-1.4b_L12_*_pairs.json 2>/dev/null
  git -c user.email=reuellee@gmail.com -c user.name=reuellee commit -q -m "round 12 RESULTS: real-model L1-vs-TopK first-letter absorption (lambda=$LAM, frozen-scorer verdict)" && log "committed" || log "nothing to commit"
  git push -q origin master && log "pushed" || log "push failed"
else
  log "no results_round12.txt collected -- run may have failed; box left up"
fi

# 5. stop the L4 only on clean completion
if [ "$DONE" = "1" ]; then
  log "deleting $L4"
  gcloud compute instances delete $L4 --zone=$Z --quiet && log "L4 deleted" || log "L4 delete failed -- STOP IT MANUALLY"
else
  log "NOT deleting $L4 (run incomplete) -- inspect ~/r12_run.log on the L4"
fi
log "=== drive_r12 done ==="
touch ~/drive_r12_done
