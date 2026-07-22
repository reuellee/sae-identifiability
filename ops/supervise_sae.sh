#!/bin/bash
# Supervises the SAE-identifiability runs on dev-gpu, then stops dev-gpu.
# Rev 4: chain is now experiments -> remedies -> round2 -> round3 (sae_round3.py,
# validates the corrected coherence-penalty boundary eps**); finish = done_r3.flag.
# Laptop session retired its supervisor 07:15 UTC and handed lifecycle to this
# session; this script is its instrument. Does NOT stop the orchestrator.
ZONE=us-west1-a
GPU=dev-gpu
RUSER="${USER}"
DIR=/home/$RUSER/sae_identifiability
DEADLINE=$(date -d '2026-07-21 09:45:00 UTC' +%s)
log(){ echo "$(date -u +%F' '%H:%M:%S) $*" >> ~/supervise.log; }
gssh(){ timeout 90 gcloud compute ssh $GPU --zone=$ZONE --quiet --command="$1" 2>/dev/null; }
collect(){
  mkdir -p ~/sae_results
  for f in results_absorption.csv results_recovery.csv results_remedies.csv \
           results_b_fine.csv results_a_oracle.csv results_c1_boundary.csv results_c3_rich.csv run.log run2.log run3.log run4.log; do
    timeout 120 gcloud compute scp --zone=$ZONE --quiet \
      "$GPU:$DIR/$f" ~/sae_results/ 2>>~/supervise.log
  done
  log "collected: $(ls ~/sae_results 2>/dev/null | tr '\n' ' ')"
}
arm_r2_chain(){
  gssh "sudo -u $RUSER bash -c 'cd $DIR && nohup bash -c \"while [ ! -f done2.flag ]; do sleep 20; done; [ -f done_r2.flag ] || python3 sae_round2.py > run3.log 2>&1\" >/dev/null 2>&1 < /dev/null &'"
}
log "=== supervision rev4 (chain through round3; finish=done_r3.flag) ==="
while true; do
  st=$(gcloud compute instances describe $GPU --zone=$ZONE --format='value(status)' 2>/dev/null)
  if [ "$st" != "RUNNING" ]; then log "dev-gpu status=$st; exiting loop"; break; fi
  if [ $(date +%s) -gt $DEADLINE ]; then
    log "DEADLINE — force collect + stop dev-gpu"
    collect
    gcloud compute instances stop $GPU --zone=$ZONE --quiet
    break
  fi
  if [ -n "$(gssh "ls $DIR/done_r3.flag 2>/dev/null")" ]; then
    log "done_r3.flag present — final collect + stop dev-gpu"
    collect
    gcloud compute instances stop $GPU --zone=$ZONE --quiet
    log "dev-gpu stopped; supervision complete"
    break
  fi
  alive=$(gssh "pgrep -f 'sae_experiment[s]|sae_remedie[s]|sae_round[23]|done.fla[g]' | head -1")
  d1=$(gssh "ls $DIR/done.flag 2>/dev/null")
  d2=$(gssh "ls $DIR/done2.flag 2>/dev/null")
  if [ -z "$alive" ]; then
    if [ -z "$d1" ]; then
      log "main run dead without done.flag — restarting (fresh) + re-arming chains"
      gssh "sudo -u $RUSER bash -c 'cd $DIR && nohup python3 sae_experiments.py >> run.log 2>&1 < /dev/null &'"
      gssh "sudo -u $RUSER bash -c 'cd $DIR && nohup bash -c \"while [ ! -f done.flag ]; do sleep 20; done; [ -f done2.flag ] || python3 sae_remedies.py > run2.log 2>&1\" >/dev/null 2>&1 < /dev/null &'"
      arm_r2_chain
    elif [ -z "$d2" ]; then
      log "remedies dead without done2.flag — restarting sae_remedies + r2 chain"
      gssh "sudo -u $RUSER bash -c 'cd $DIR && nohup python3 sae_remedies.py >> run2.log 2>&1 < /dev/null &'"
      arm_r2_chain
    elif [ -z "$(gssh "ls $DIR/done_r2.flag 2>/dev/null")" ]; then
      log "round2 dead without done_r2.flag — restarting sae_round2 + r3 chain"
      gssh "sudo -u $RUSER bash -c 'cd $DIR && nohup python3 sae_round2.py >> run3.log 2>&1 < /dev/null &'"
      gssh "sudo -u $RUSER bash -c 'cd $DIR && nohup bash -c \"while [ ! -f done_r2.flag ]; do sleep 20; done; [ -f done_r3.flag ] || python3 sae_round3.py > run4.log 2>&1\" >/dev/null 2>&1 < /dev/null &'"
    else
      log "round3 dead without done_r3.flag — restarting sae_round3"
      gssh "sudo -u $RUSER bash -c 'cd $DIR && nohup python3 sae_round3.py >> run4.log 2>&1 < /dev/null &'"
    fi
  else
    log "alive: $(gssh "tail -1 $DIR/run4.log 2>/dev/null || tail -1 $DIR/run3.log 2>/dev/null || tail -1 $DIR/run2.log 2>/dev/null || tail -1 $DIR/run.log 2>/dev/null")"
  fi
  sleep 120
done
log "supervision rev4 done (orchestrator left running for analysis)"
