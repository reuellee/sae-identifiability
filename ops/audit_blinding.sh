#!/usr/bin/env bash
# Audit whether blinded agents actually stayed blind.
#
# WHY THIS EXISTS
# Round 13b produced two theory notes whose commit messages claim their predictions
# were "BOXED PRE-UNBLINDING". The commit timestamps do NOT establish that: the
# results were committed at 18:54:56 and the theory notes at 19:04:39 / 19:05:22,
# i.e. ten minutes LATER. On timestamps alone the claim is unsupported, and an
# external reviewer checking `git log` will see exactly that and stop reading.
#
# The real evidence is not in git, it is in the agent transcripts: were the agents
# spawned before the results existed, were they given an explicit blinding
# instruction, and did they in fact never open a results file? That is checkable,
# which is what the thesis defense (Q4, prereg auditability) asked for -- "say
# concretely what additional artifact would make the lock externally auditable".
# This is that artifact.
#
# It reports, per sub-agent: start/end time, tool-call count, and every tool call
# whose input matches the forbidden pattern. A compliant blinded agent shows zero
# matches -- or a match that is an EXCLUSION (e.g. `grep -v round13b`), which this
# prints in full so a human can tell the two apart rather than trusting a count.
#
# Usage:
#   ops/audit_blinding.sh <session-transcript-dir> <forbidden-regex> [start HH:MM:SS] [end HH:MM:SS]
# Example (round 13b):
#   ops/audit_blinding.sh ~/.claude/projects/-home-reuellee-gmail-com/55132645-*/ \
#       'round13b|r13b|13b_results' 18:40:00 19:10:00
set -uo pipefail
DIR=${1:?usage: audit_blinding.sh <session-dir> <regex> [start] [end]}
PAT=${2:?need a forbidden-path regex}
T0=${3:-00:00:00}
T1=${4:-23:59:59}
SUB="$DIR/subagents"
[ -d "$SUB" ] || { echo "no subagents/ under $DIR"; exit 1; }

python3 - "$SUB" "$PAT" "$T0" "$T1" <<'PY'
import sys, json, re, glob, os, datetime
sub, pat, t0, t1 = sys.argv[1:5]
rx = re.compile(pat, re.I)
rows = []
for f in glob.glob(os.path.join(sub, "agent-*.jsonl")):
    first = last = None; calls = 0; hits = []
    for line in open(f, errors="replace"):
        try: d = json.loads(line)
        except Exception: continue
        ts = (d.get("timestamp") or "")[11:19]
        if ts:
            first = first or ts
            last = ts
        c = (d.get("message") or {}).get("content")
        if not isinstance(c, list): continue
        for b in c:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                calls += 1
                inp = json.dumps(b.get("input", {}))
                if rx.search(inp):
                    hits.append((ts, b.get("name"), inp))
    if first and t0 <= first <= t1:
        rows.append((first, last, os.path.basename(f), calls, hits))

rows.sort()
print(f"agents starting in [{t0}, {t1}]: {len(rows)}\n")
clean = 0
for first, last, name, calls, hits in rows:
    print(f"{name[:26]}  {first} -> {last}  {calls} tool calls  {len(hits)} match(es)")
    for ts, tool, inp in hits:
        # An exclusion (grep -v / --exclude / "do not read") is compliance, not a breach.
        excl = re.search(r"grep\s+-v|--exclude|\bnot\b.{0,20}read", inp, re.I)
        tag = "EXCLUSION (compliant)" if excl else "*** ACCESS ***"
        print(f"    {ts} {tool}: {tag}")
        print(f"      {inp[:240]}")
    if not any(not re.search(r"grep\s+-v|--exclude|\bnot\b.{0,20}read", i, re.I)
               for _, _, i in hits):
        clean += 1
    print()
print(f"{clean}/{len(rows)} agents made no substantive access to the forbidden paths.")
print("A count alone is not proof -- read the printed inputs; an exclusion and an")
print("access both 'match' the pattern and only the text distinguishes them.")
PY
