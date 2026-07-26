#!/usr/bin/env bash
# Verify the paper's citations: every [@key] resolves to a bib entry, every arXiv
# entry resolves on arXiv, and the attribution near each citation matches the real
# authors/title.
#
# WHY THIS EXISTS
# Much of the related-work and novelty apparatus here was assembled from
# LLM-assisted novelty sweeps (gemx grounded search + Claude search). Those produce
# *plausible* arXiv ids and *plausible* author lists. Two real errors so far, both
# of which a does-it-exist check passes:
#   - arXiv:2408.02622 cited as "O'Neill et al." for feature families is actually
#     "Language Model Can Listen While Speaking", a speech paper (correct: 2408.00657).
#   - arXiv:2606.12138 cited as "Paulo & Belrose" is Gerasimov et al.; Paulo &
#     Belrose is 2501.16615. Both real, both relevant, wrong pairing.
# The id RESOLVED in both cases. Topical/authorial mismatch is the real risk, so
# this prints the resolved title and authors beside the citing text.
#
# Since PAPER.md moved to pandoc [@key] citations, scanning for "arXiv:" in the
# prose finds nothing -- the ids now live in refs.bib. This checks the citation
# GRAPH instead: prose keys -> bib entries -> arXiv.
#
# Usage: ops/check_citations.sh [file ...]     (default: PAPER.md)
set -uo pipefail
REPO=${REPO:-$HOME/sae-identifiability}
cd "$REPO"
FILES=${*:-PAPER.md}

command -v curl >/dev/null || { echo "need curl"; exit 1; }
IDS=$(grep -ohE "eprint = \{[0-9]{4}\.[0-9]{4,5}\}" refs.bib 2>/dev/null |
      grep -oE "[0-9]{4}\.[0-9]{4,5}" | sort -u)
[ -z "$IDS" ] && { echo "no arXiv ids in refs.bib"; exit 1; }
LIST=$(echo "$IDS" | tr '\n' ',' | sed 's/,$//')
XML=$(mktemp); trap 'rm -f "$XML"' EXIT
# NOTE: https, not http -- plain http returns an empty body from this network.
if ! curl -sS --max-time 150 "https://export.arxiv.org/api/query?id_list=${LIST}&max_results=200" -o "$XML" || [ ! -s "$XML" ]; then
  echo "FAIL: could not reach the arXiv API"; exit 1
fi

python3 - "$XML" $FILES <<'PY'
import sys, re, pathlib, xml.etree.ElementTree as ET

ns = {'a': 'http://www.w3.org/2005/Atom'}
xml, files = sys.argv[1], sys.argv[2:]

live = {}
for e in ET.parse(xml).getroot().findall('a:entry', ns):
    base = e.find('a:id', ns).text.rsplit('/', 1)[-1].split('v')[0]
    live[base] = (' '.join((e.find('a:title', ns).text or '').split()),
                  [x.find('a:name', ns).text for x in e.findall('a:author', ns)])

# --- parse both bibliographies -------------------------------------------------
entries = {}
for bib in ('refs.bib', 'refs_manual.bib'):
    try:
        raw = pathlib.Path(bib).read_text()
    except FileNotFoundError:
        continue
    for blk in re.findall(r'@\w+\{([^,]+),(.*?)\n\}', raw, re.S):
        key, body = blk[0].strip(), blk[1]
        def field(n):
            m = re.search(n + r'\s*=\s*\{(.*?)\}\s*(?:,|\n)', body, re.S)
            return ' '.join(m.group(1).split()) if m else ''
        entries[key] = dict(eprint=field('eprint'), bib=bib,
                            title=field('title').strip('{}'),
                            author=field('author'))

text = '\n'.join(pathlib.Path(f).read_text() for f in files)
cited = sorted(set(re.findall(r'@([A-Za-z][A-Za-z0-9_:-]*)', text)))
bad = flagged = 0

print(f"{len(cited)} distinct keys cited in {' '.join(files)}; "
      f"{len(entries)} bib entries; {len(live)}/{len(set(e['eprint'] for e in entries.values() if e['eprint']))} arXiv ids resolved\n")

for key in cited:
    if key not in entries:
        print(f'  DANGLING   [@{key}]  <- no entry in refs.bib or refs_manual.bib'); bad += 1
        continue
    ent = entries[key]
    eid = ent['eprint']
    if eid and eid not in live:
        print(f'  UNRESOLVED [@{key}] arXiv:{eid} <- does not exist on arXiv'); bad += 1
        continue
    title, authors = live.get(eid, (ent['title'], [a.strip() for a in ent['author'].split(' and ')]))

    m = re.search(r'\[@[^]]*\b' + re.escape(key) + r'\b', text)
    ctx = ' '.join(text[max(0, m.start()-95):m.start()].split())[-78:] if m else ''
    win = ' '.join(text[max(0, m.start()-170):m.start()+130].split()).lower() if m else ''

    # Attributed if an author surname OR a distinctive title word appears nearby.
    # Title tokens matter because works here are routinely cited by name (OrtSAE,
    # Tree SAE, SAEBench) rather than by author -- normal style, must not flag.
    surnames = {a.split()[-1].strip(',.').lower() for a in authors[:6] if a.strip()}
    STOP = {'the','and','for','with','from','into','sparse','autoencoders','autoencoder',
            'features','feature','learning','model','models','via','are','different',
            'same','data','trained','analysis','study','studying'}
    toks = {w for w in re.findall(r"[a-z0-9^{}$']{4,}", title.lower()) if w not in STOP}
    hit = any(s and s in win for s in surnames) or any(t in win for t in toks)
    if not hit:
        flagged += 1
    print(f"{'  ' if hit else '? '}[@{key}]  {title[:62]}")
    print(f"      {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}"
          f"{'  arXiv:' + eid if eid else '  (' + ent['bib'] + ')'}")
    print(f"      cited as: ...{ctx}")
    if not hit:
        print('      ^ neither an author surname nor a title word appears nearby -- CHECK ATTRIBUTION')

unused = [k for k in entries if k not in cited]
if unused:
    print(f"\nnote: {len(unused)} bib entries are never cited (harmless with --citeproc, "
          f"they are simply omitted): {', '.join(sorted(unused)[:8])}")
print(f"\n{bad} broken, {flagged} need a human read.")
print('Resolving is necessary, not sufficient: check the "?" lines and any '
      'attribution that names a person.')
sys.exit(1 if bad else 0)
PY
