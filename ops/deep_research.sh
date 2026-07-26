#!/usr/bin/env bash
# Grounded literature search via gemx, driven so that its known failure mode cannot
# eat the run.
#
# WHY THIS SHAPE
# gemx (Vertex Gemini, see the notes in ops/defense.sh) has working google_web_search
# grounding but reliably dies in MULTI-STEP agentic loops -- "Invalid stream:
# malformed tool call", or it simply never returns and gets killed by the timeout.
# On 2026-07-26 a 5-question sweep lost question 2 to three consecutive timeouts. The
# question that died was the compound one: "what is the CURRENT definition, HAS it
# been revised, cite papers, versions, ids and dates" -- four asks in one, which is
# what pushes it into iterative search. The questions that succeeded were single-ask.
#
# So: one question per call, an explicit single-pass instruction, `< /dev/null`
# (gemx eats stdin inside while-read loops and silently consumes the rest of the
# query file), per-question checkpoint files, and a shorter timeout with more
# retries -- a hung call is not going to un-hang, so failing fast and retrying beats
# waiting ten minutes.
#
# IMPORTANT: treat every answer as a LEAD, not a fact. In the same sweep gemx
# attributed arXiv:2607.12166 to four named researchers; it is a single-author paper
# and gemx could not produce the id at all. Verify ids and author lists against the
# arXiv API (ops/mkbib.py, ops/check_citations.sh) before anything reaches the paper.
#
# RUN FROM AN EMPTY DIRECTORY -- this is the big one.
# gemx sets GEMINI_CLI_TRUST_WORKSPACE=true, so the CLI treats its CWD as a workspace
# and will happily spend its whole tool budget grepping and reading local files
# instead of searching the web. Launched from the repo root it emits "Ripgrep is not
# available. Falling back to GrepTool" and then dies on the timeout having never run
# a search. That is almost certainly a large part of the "gemx is flaky in batch"
# history: it was not flaky, it was exploring a hundred-file research repo. Every
# call below runs in a scratch directory containing nothing.
#
# Usage: ops/deep_research.sh <queries.txt> [outdir]
#   queries.txt: one question per line, "tag|question". Blank lines and # ignored.
set -uo pipefail
QF=$(readlink -f "${1:?usage: ops/deep_research.sh <queries.txt> [outdir]}")
OUT=${2:-$(dirname "$QF")/answers}
ATTEMPTS=${ATTEMPTS:-4}
PER_TRY=${PER_TRY:-240}

command -v gemx >/dev/null || { echo "gemx not on PATH"; exit 1; }
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
SCRATCH=$(mktemp -d); trap 'rm -rf "$SCRATCH"' EXIT
cd "$SCRATCH"

PREFIX='Answer in ONE pass. Run google_web_search if you need it, then write the
answer immediately -- do NOT iterate, do NOT re-plan, do NOT call more than a few
searches. Cite exact arXiv ids, exact titles, author lists and dates. State plainly
which claims you VERIFIED via search and which are recall. If you find nothing, the
correct answer is the single phrase NONE FOUND plus a short note on what you searched.

QUESTION: '

ok=0; fail=0
while IFS='|' read -r tag q; do
  case "$tag" in ''|'#'*) continue;; esac
  dst="$OUT/${tag}.md"
  if [ -s "$dst" ]; then echo "[$tag] cached, skipping"; ok=$((ok+1)); continue; fi
  got=0
  for i in $(seq 1 "$ATTEMPTS"); do
    printf '[%s] attempt %d/%d %s\n' "$tag" "$i" "$ATTEMPTS" "$(date -u +%H:%M:%S)"
    if timeout "$PER_TRY" gemx -p "${PREFIX}${q}" < /dev/null > "$dst.part" 2> "$OUT/${tag}.err"; then
      if [ -s "$dst.part" ]; then
        mv "$dst.part" "$dst"
        printf '[%s] OK (%s bytes)\n' "$tag" "$(wc -c < "$dst")"
        got=1; break
      fi
    fi
    sleep 5
  done
  rm -f "$dst.part"
  if [ "$got" = 1 ]; then ok=$((ok+1)); else
    fail=$((fail+1))
    printf '[%s] GAVE UP after %d attempts -- ask this one a different way, or split it.\n' "$tag" "$ATTEMPTS"
  fi
done < "$QF"

echo "--- $ok answered, $fail unanswered; answers in $OUT ---"
[ "$fail" -gt 0 ] && echo "Unanswered questions are usually COMPOUND. Split them into single asks and rerun."
exit 0
