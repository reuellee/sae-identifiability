#!/bin/bash
cd "$(dirname "$0")"
while IFS='|' read -r qid qtext; do
  out="${qid}.md"
  [ -s "$out" ] && { echo "$qid cached"; continue; }
  for attempt in 1 2 3; do
    result=$(timeout 200 gemx -p "$qtext

List up to 5 genuinely existing, most relevant items. For each: title, authors, venue/site, date, arXiv ID or URL, and 1-2 lines on the claim and how it relates. Do NOT invent items; only report what your web search actually returns. If nothing relevant exists, write exactly: NONE FOUND. Output plain markdown."  < /dev/null 2>/dev/null | grep -v "Ripgrep\|256-color\|Listener\|trace-warnings")
    if [ -n "$result" ] && ! echo "$result" | grep -q "Invalid stream\|ERROR"; then
      printf '## %s\n\n%s\n' "$qid" "$result" > "$out"
      echo "$qid done (attempt $attempt)"
      break
    fi
    echo "$qid attempt $attempt failed; retrying"
    sleep 5
  done
  [ -s "$out" ] || echo "FAILED: $qid" >> failures.log
done < queries.txt
echo "SWEEP COMPLETE"
