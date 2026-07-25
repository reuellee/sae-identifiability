#!/usr/bin/env bash
# Hold off the orchestrator's 2h idle auto-stop *while a specific process runs*.
#
# WHY: /usr/local/bin/orch-idle-check.sh counts the box "active" if there is a
# logged-in user, OR load1 > 0.40, OR a Claude transcript touched in the last
# 10 min. A nohup'd collector that sleeps 5 min between short scp bursts matches
# none of those reliably, so the orchestrator can stop itself in the middle of
# retrieving a ~$9 GPU run whose only copy is on an L4 that may not restart
# (recurring us-west1-a L4 stockouts).
#
# This does NOT disable the idle-stop. It refreshes the same last-active stamp the
# checker already uses -- i.e. it adds "a long detached job is running" as a fourth
# activity signal, matching the mechanism's existing intent. It is strictly bounded:
# it stops when the watched PID exits, and unconditionally at DEADLINE, after which
# normal idle-stop behaviour resumes on its own.
#
# WATCHES A PID, NOT A PATTERN. The first version took a pgrep pattern and hung
# forever: the pattern is one of this script's own arguments, so `pgrep -f` matched
# this process (and `timeout`, and the invoking shell) and the loop could never end.
# Excluding $$ and $PPID is not sufficient -- ANY ancestor whose argv contains the
# pattern also matches. The same self-match class bit twice in one session; the
# other time, `pkill -f "r13b/watchdog.sh"` killed the ssh session issuing it.
# `kill -0 <pid>` has no such ambiguity.
#
# Usage:
#   nohup ./long_job.sh & ops/idle_hold.sh $! [deadline-epoch] &
set -uo pipefail
PID=${1:?usage: idle_hold.sh <pid-to-watch> [deadline-epoch]}
DEADLINE=${2:-$(( $(date +%s) + 9*3600 ))}
STATE=/var/lib/orch-idle/last-active
INTERVAL=${INTERVAL:-240}          # poll period; release latency is bounded by this

kill -0 "$PID" 2>/dev/null || { echo "pid $PID is not running; nothing to hold"; exit 0; }
echo "$(date -u) holding idle-stop while pid $PID runs (deadline $(date -u -d "@$DEADLINE"))"

while kill -0 "$PID" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "$(date -u) deadline reached; releasing idle hold with pid $PID still running"
    exit 0
  fi
  date +%s | sudo tee "$STATE" >/dev/null 2>&1
  sleep "$INTERVAL"
done
echo "$(date -u) pid $PID exited; idle hold released"
