#!/usr/bin/env bash
# Verify every arXiv id cited in the repo resolves, and print title/authors so a
# human (or reviewer) can confirm the citation says what the text claims it says.
#
# WHY THIS EXISTS
# Much of the related-work and novelty apparatus here was assembled from
# LLM-assisted novelty sweeps (gemx grounded search + Claude search). Those
# produce *plausible* arXiv ids. Checking on 2026-07-25 found all 27 ids resolved
# -- but arXiv:2408.02622, cited in PAPER.md §9 as "O'Neill et al." for feature-
# family hierarchies, is actually "Language Model Can Listen While Speaking", a
# full-duplex speech paper. The intended work was arXiv:2408.00657 (O'Neill, Ye,
# Iyer, Wu, "Disentangling Dense Embeddings with Sparse Autoencoders"). The digits
# were corrupted into another real paper, which is the failure mode that a
# does-it-resolve check alone will never catch.
#
# So this prints the resolved title next to the surrounding citation text: an id
# that resolves is necessary, not sufficient. Topical mismatch is the real risk.
#
# Usage: ops/check_citations.sh [file ...]     (default: PAPER.md)
set -uo pipefail
REPO=${REPO:-$HOME/sae-identifiability}
cd "$REPO"
FILES=${*:-PAPER.md}
IDS=$(grep -ohE "arXiv:[0-9]{4}\.[0-9]{4,5}" $FILES 2>/dev/null | sed 's/arXiv://' | sort -u)
[ -z "$IDS" ] && { echo "no arXiv ids found in: $FILES"; exit 0; }
N=$(echo "$IDS" | wc -l)
echo "checking $N arXiv ids from: $FILES"

LIST=$(echo "$IDS" | tr '\n' ',' | sed 's/,$//')
XML=$(mktemp); trap 'rm -f "$XML"' EXIT
# NOTE: https, not http -- plain http returns an empty body from this network.
if ! curl -sS --max-time 120 "https://export.arxiv.org/api/query?id_list=${LIST}&max_results=100" -o "$XML" || [ ! -s "$XML" ]; then
  echo "FAIL: could not reach the arXiv API"; exit 1
fi

python3 - "$XML" $FILES <<'PY'
import sys, re, pathlib, xml.etree.ElementTree as ET
ns = {'a': 'http://www.w3.org/2005/Atom'}
xml, files = sys.argv[1], sys.argv[2:]
meta = {}
for e in ET.parse(xml).getroot().findall('a:entry', ns):
    aid = e.find('a:id', ns).text.rsplit('/', 1)[-1]
    base = aid.split('v')[0]
    title = ' '.join((e.find('a:title', ns).text or '').split())
    authors = [x.find('a:name', ns).text for x in e.findall('a:author', ns)]
    meta[base] = (title, authors)

text = '\n'.join(pathlib.Path(f).read_text() for f in files)
bad = 0
for m in sorted(set(re.findall(r'arXiv:([0-9]{4}\.[0-9]{4,5})', text))):
    if m not in meta:
        print(f'  UNRESOLVED  arXiv:{m}  <- does not exist on arXiv'); bad += 1; continue
    title, authors = meta[m]
    first = authors[0].split()[-1] if authors else '?'
    # pull the ~90 chars before the citation so the claimed attribution is visible
    ctx = win = ''
    for cm in re.finditer(r'arXiv:' + re.escape(m), text):
        ctx = ' '.join(text[max(0, cm.start()-90):cm.start()].split())[-80:]
        # wider window for matching: works are often named after the citation too
        win = ' '.join(text[max(0, cm.start()-160):cm.start()+120].split()).lower()
        break
    # A citation is "attributed" if either an author surname OR a distinctive word
    # from the real title appears nearby. Title tokens matter because papers here
    # are routinely cited by work name (OrtSAE, Tree SAE, C^2R, SynthSAEBench)
    # rather than by author, which is normal style and must not be flagged.
    surnames = {a.split()[-1].strip(",.").lower() for a in authors[:6]}
    STOP = {'the','and','for','with','from','into','sparse','autoencoders','autoencoder',
            'features','feature','learning','model','models','a','of','in','on','via'}
    toks = {w for w in re.findall(r"[a-z0-9^{}$']{4,}", title.lower()) if w not in STOP}
    hit = any(s and s in win for s in surnames) or any(t in win for t in toks)
    flag = ' ' if hit else '?'
    print(f'{flag} arXiv:{m}  {title[:64]}')
    print(f'    authors: {", ".join(authors[:3])}{" et al." if len(authors)>3 else ""}')
    print(f'    cited as: ...{ctx}')
    if not hit:
        print(f'    ^ no author surname from this entry appears near the citation -- CHECK ATTRIBUTION')
print()
print(f'{len(meta)}/{len(set(re.findall(r"arXiv:([0-9]{4}[.][0-9]{4,5})", text)))} ids resolved.')
print('Lines marked "?" need a human read: resolving is necessary, not sufficient.')
sys.exit(1 if bad else 0)
PY
