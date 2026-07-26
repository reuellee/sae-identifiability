import urllib.request, xml.etree.ElementTree as ET, re, sys, time

ids = """2408.00657 2409.14507 2410.19750 2503.17547 2505.11756 2506.15963
2509.22033 2602.11881 2602.14687 2605.07922 2606.12138 2606.30609
2502.04878 2506.01197 2607.12166 2412.06410
2605.18229 2503.09532 2501.16615""".split()

NS = {'a':'http://www.w3.org/2005/Atom'}
url = "https://export.arxiv.org/api/query?id_list=%s&max_results=100" % ",".join(ids)
raw = urllib.request.urlopen(url, timeout=90).read()
root = ET.fromstring(raw)

def key(first_author, year, title):
    ln = first_author.split()[-1].lower()
    ln = re.sub(r'[^a-z]', '', ln)
    w = [x for x in re.sub(r'[^a-zA-Z ]',' ',title).split() if len(x) > 3]
    return "%s%s%s" % (ln, year, (w[0].lower() if w else "x"))

out = []
found = set()
for e in root.findall('a:entry', NS):
    aid = e.find('a:id', NS).text.strip()
    m = re.search(r'abs/([0-9]+\.[0-9]+)', aid)
    if not m: continue
    num = m.group(1); found.add(num)
    title = " ".join(e.find('a:title', NS).text.split())
    pub = e.find('a:published', NS).text[:4]
    authors = [a.find('a:name', NS).text for a in e.findall('a:author', NS)]
    k = key(authors[0], pub, title)
    astr = " and ".join(authors)
    out.append((num, k, title, astr, pub, len(authors)))

out.sort()
with open('refs.bib','w') as f:
    f.write("%% Auto-generated from the arXiv API. Titles/authors/years are as returned by arXiv.\n")
    f.write("%% Regenerate: python3 ops/mkbib.py ; validate: ops/check_citations.sh\n\n")
    for num,k,title,astr,pub,na in out:
        f.write("@misc{%s,\n  title  = {{%s}},\n  author = {%s},\n  year   = {%s},\n  eprint = {%s},\n  archivePrefix = {arXiv},\n  primaryClass = {cs.LG},\n  url    = {https://arxiv.org/abs/%s}\n}\n\n" % (k,title,astr,pub,num,num))

print("RESOLVED %d / %d" % (len(found), len(ids)))
for num,k,title,astr,pub,na in out:
    print("  %-12s -> %-22s %s (%s, %d authors)" % (num, k, title[:72], pub, na))
missing = [i for i in ids if i not in found]
if missing: print("!! UNRESOLVED (likely fabricated or wrong id): %s" % missing)
