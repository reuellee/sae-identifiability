#!/usr/bin/env bash
# Build PAPER.md -> PAPER.pdf (pandoc + pdflatex).
#
# SINGLE SOURCE OF TRUTH: PAPER.md.
# There is deliberately no hand-maintained .tex. This paper has been revised
# every round for thirteen rounds; a parallel .tex would drift from the markdown
# within a week, and the markdown is also what a LessWrong/Alignment Forum post
# needs. So the PDF is generated, never edited.
#
# The markdown NUMBERS ITS OWN SECTIONS ("## 4. The exact pure-strategy
# crossover"), so --number-sections is deliberately NOT passed -- passing it
# yields "4 4. The exact ...".
#
# Title and abstract are lifted out of the markdown into LaTeX metadata so the
# PDF gets a real title block and \begin{abstract}, rather than an "Abstract"
# section heading. PAPER.md itself is not modified.
#
# Usage: ops/build_paper.sh [output.pdf]
set -euo pipefail
REPO=${REPO:-$HOME/sae-identifiability}
cd "$REPO"
SRC=${SRC:-PAPER.md}
OUT=${1:-PAPER.pdf}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

command -v pandoc  >/dev/null || { echo "need pandoc: sudo apt-get install -y pandoc"; exit 1; }
# xelatex, not pdflatex: PAPER.md contains bare Unicode in PROSE (rho, sigma,
# section sign, en/em dashes, arrow) outside math mode, which pdflatex rejects
# with "Unicode character not set up for use with LaTeX". xelatex handles it
# natively, so the markdown stays readable instead of being littered with
# \(\rho\) escapes for the benefit of the PDF build.
command -v xelatex >/dev/null || { echo "need xelatex: sudo apt-get install -y texlive-xetex fonts-lmodern"; exit 1; }

DATE=$(git log -1 --format=%cs -- "$SRC" 2>/dev/null || date -u +%F)
REV=$(git rev-parse --short HEAD 2>/dev/null || echo "uncommitted")

python3 - "$SRC" "$TMP/body.md" "$TMP/meta.yaml" "$DATE" "$REV" <<'PY'
import sys, re, pathlib
src, body_out, meta_out, date, rev = sys.argv[1:6]
lines = pathlib.Path(src).read_text().splitlines()

title = lines[0].lstrip('#').strip() if lines and lines[0].startswith('# ') else 'Untitled'

# Slice out the Abstract section: everything between "## Abstract" and the next
# "## " heading. The preamble before it (the living-draft notice) is kept, but
# demoted to a small italic note under the title rather than body text.
abs_start = abs_end = None
for i, l in enumerate(lines):
    if re.match(r'^##\s+Abstract\s*$', l):
        abs_start = i
    elif abs_start is not None and l.startswith('## ') and i > abs_start:
        abs_end = i
        break
if abs_start is None:
    abstract, notice, body = '', '', lines[1:]
else:
    abstract = '\n'.join(lines[abs_start+1:abs_end]).strip().strip('-').strip()
    notice   = '\n'.join(lines[1:abs_start]).strip().strip('-').strip()
    body     = lines[abs_end:]

def esc(s):
    return s.replace('\\', '\\\\').replace('"', '\\"')

# YAML block scalars keep the markdown/math in the abstract intact.
def block(key, text):
    if not text:
        return ''
    ind = '\n'.join('  ' + l for l in text.splitlines())
    return f'{key}: |\n{ind}\n'

meta = (f'title: "{esc(title)}"\n'
        f'author: "Reuel Lee"\n'
        f'date: "{date} \\\\quad\\\\small rev \\\\texttt{{{rev}}}"\n'
        + block('abstract', abstract))
pathlib.Path(meta_out).write_text('---\n' + meta + '---\n')

out = []
if notice:
    # the living-draft caveat belongs on the title page, small and italic
    out += ['\\begin{center}\\begin{minipage}{0.88\\textwidth}\\footnotesize\\itshape',
            notice, '\\end{minipage}\\end{center}', '']
out += body
pathlib.Path(body_out).write_text('\n'.join(out))
PY

cat "$TMP/meta.yaml" "$TMP/body.md" > "$TMP/paper.md"

pandoc "$TMP/paper.md" \
  --from=markdown+tex_math_dollars+raw_tex+pipe_tables \
  --pdf-engine=xelatex \
  --citeproc --bibliography=refs.bib --bibliography=refs_manual.bib \
  --include-in-header=ops/paper_header.tex \
  --resource-path=.:figs:figures \
  -V documentclass=article -V papersize=a4 -V fontsize=11pt \
  -V geometry:margin=1in -V linkcolor=black \
  -V mainfont="TeX Gyre Pagella" -V mathfont="TeX Gyre Pagella Math" \
  --toc --toc-depth=2 \
  -o "$OUT" 2> "$TMP/pandoc.err" || { echo "BUILD FAILED:"; tail -25 "$TMP/pandoc.err"; exit 1; }

# A "Missing character" warning means xelatex SILENTLY DROPPED that glyph from the
# PDF -- e.g. Latin Modern has no Greek, so every bare rho/sigma in prose vanished
# and sentences read as "counting recovers  to <= 0.02". A research PDF that quietly
# loses characters is worse than one that fails to build, so this is fatal.
if grep -q "Missing character" "$TMP/pandoc.err" 2>/dev/null; then
  echo "BUILD FAILED: font is missing glyphs that would be dropped from the PDF:"
  grep "Missing character" "$TMP/pandoc.err" | sed 's/^/  /' | sort -u | head -10
  echo "  -> pick a mainfont/mathfont covering these (TeX Gyre Pagella and DejaVu Serif have Greek)."
  rm -f "$OUT"; exit 1
fi
# Pandoc warns rather than fails on missing images/refs -- surface those.
if [ -s "$TMP/pandoc.err" ]; then
  echo "--- pandoc warnings ---"; sed 's/^/  /' "$TMP/pandoc.err" | head -15
fi
echo "built $OUT ($(du -h "$OUT" | cut -f1), $(pdfinfo "$OUT" 2>/dev/null | awk '/^Pages/{print $2}') pages) from $SRC @ $REV"
