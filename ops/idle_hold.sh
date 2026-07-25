#!/usr/bin/env bash
# Hold off the orchestrator's 2h idle auto-stop *while a detached collector runs*.
#
# WHY: /usr/local/bin/orch-idle-check.sh counts the box "active" if there is a
# logged-in user, OR load1 > 0.40, OR a Claude transcript touched in the last
# 10 min. A nohup'd collector that sleeps 5 min between short scp bursts matches
# none of those reliably, so the orchestrator can stop itself in the middle of
# retrieving a ~$8 GPU run whose only copy is on an L4 that may not restart
# (recurring us-west1-a L4 stockouts).
#
# This does NOT disable the idle-stop. It refreshes the same last-active stamp the
# checker already uses -- i.e. it adds "a long detached job is running" as a fourth
# activity signal, matching the mechanism's existing intent. It is strictly bounded:
#   - stops as soon as the watched process exits, and
#   - stops unconditionally at DEADLINE (default: past the L4's own hard cap),
# after which normal idle-stop behaviour resumes on its own.
#
# Usage: ops/idle_hold.sh <pattern-to-watch> [deadline-epoch]
set -uo pipefail
PATTERN=${1:?usage: idle_hold.sh <pgrep-pattern> [deadline-epoch]}
DEADLINE=${2:-$(( $(date +%s) + 9*3600 ))}
STATE=/var/lib/orch-idle/last-active

while pgrep -f "$PATTERN" >/dev/null 2>&1; do
  now=$(date +%s)
  [ "$now" -ge "$DEADLINE" ] && { echo "$(date -u) deadline reached; releasing idle hold"; break; }
  date +%s | sudo tee "$STATE" >/dev/null 2>&1
  sleep 240
done
echo "$(date -u) idle hold released (watched process gone or deadline)"
