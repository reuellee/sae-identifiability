#!/usr/bin/env bash
# Generic self-poweroff watchdog to install ON a compute VM running a long job.
#
# WHY: `ephemeral` VMs are created --no-service-account --no-scopes, so they cannot
# call the GCE API and CANNOT DELETE THEMSELVES. If the orchestrator session ends,
# or the job hangs, nothing stops the meter. Powering off is the one cost control
# the VM can apply unilaterally: a stopped e2-standard-8 bills ~$0 of CPU instead of
# ~$0.27/h, leaving only ~30GB of disk (~$1.20/month) until someone runs
# `ephemeral del <name>`.
#
# Round 13b's L4 had this; the round-14 ephemeral did not, which is the gap this
# closes. Watches a PID, never a pgrep pattern -- a pattern that appears in this
# script's own argv matches itself and the watchdog can then never terminate
# (that bug cost real time on 2026-07-25; see ops/idle_hold.sh).
#
# Usage, on the VM:
#   nohup setsid ops/vm_watchdog.sh <pid-to-watch> [max-hours] [grace-seconds] &
set -uo pipefail
PID=${1:?usage: vm_watchdog.sh <pid> [max-hours] [grace-seconds]}
MAXH=${2:-8}
GRACE=${3:-300}
LOG=${LOG:-$HOME/vm_watchdog.log}
HARD=$(( $(date +%s) + MAXH*3600 ))

echo "$(date -u) watching pid $PID; hard cap $(date -u -d "@$HARD"); grace ${GRACE}s" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$HARD" ]; then
    echo "$(date -u) HARD CAP with job still running - powering off" >> "$LOG"
    sudo poweroff; exit 0
  fi
  sleep 60
done
echo "$(date -u) pid $PID exited; ${GRACE}s grace then poweroff" >> "$LOG"
sleep "$GRACE"
echo "$(date -u) powering off" >> "$LOG"
sudo poweroff
