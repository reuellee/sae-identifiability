#!/usr/bin/env bash
# PREFLIGHT — run this BEFORE starting a GPU round. Exits non-zero if the round
# would be unable to deliver its results, or would violate prereg discipline.
#
# WHY THIS EXISTS
# Every expensive failure in this program so far has been OPS, not science, and
# every one was mechanically detectable in seconds before the GPU meter started:
#
#   round 12  collection read GCS as the orchestrator SA (zero bucket access).
#             Results sat safe in GCS while the driver reported "no results
#             collected", the box was deleted, and no SUMMARY was committed.
#   round 12  a stale out-of-config pythia-70m smoke SAE with a duplicate seed
#             leaked into scoring and poisoned every TopK mean until re-scored.
#   round 13b dev-gpu, rebuilt from a snapshot, ran as the DEFAULT COMPUTE SA with
#             no storage scopes. All uploads failed silently under `set +e` for
#             ~9h of training. Found by accident, mid-run, with 10 SAEs to go.
#   round 13b ops/gcs_adc.sh `ls` crashed on py3.11 (f-string backslash). `get`
#             and `put` were fine, so "verified working" had never touched `ls` --
#             which is exactly the verb that would have shown the empty bucket.
#
# The lesson is not "be careful". It is that a checklist gets ticked while a
# ROUND-TRIP PROBE gets executed. Everything below actually performs the
# operation the round will depend on, against the real identity, on the real box.
#
# Usage:
#   ops/preflight.sh <round-tag> [expected-sae-count] [expected-gb]
#   VM=dev-gpu ZONE=us-west1-a ops/preflight.sh round14 48 6
set -uo pipefail

TAG=${1:?usage: preflight.sh <round-tag> [expected-sae-count] [expected-gb]}
NSAE=${2:-48}
NEEDGB=${3:-8}
PROJECT=${PROJECT:-project-ebd5a273-53ea-4c8b-81a}
ZONE=${ZONE:-us-west1-a}
VM=${VM:-dev-gpu}
BUCKET=${BUCKET:-sae-identifiability-artifacts-ebd5a273}
REPO=${REPO:-$HOME/sae-identifiability}
PASS=0; FAIL=0; WARN=0
ok()   { echo "  PASS  $*"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $*"; FAIL=$((FAIL+1)); }
warn() { echo "  WARN  $*"; WARN=$((WARN+1)); }
sec()  { echo; echo "== $* =="; }

rssh() { timeout 120 gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" \
           -- -o StrictHostKeyChecking=no "$1" 2>/dev/null; }

echo "PREFLIGHT $TAG  (vm=$VM zone=$ZONE bucket=$BUCKET expect=${NSAE} SAEs)"

# --------------------------------------------------------------------------
sec "1. orchestrator -> GCS, all three verbs (round-12 + the ls bug)"
# `ls` is checked explicitly: it was broken while get/put worked, and it is the
# verb that reveals an empty prefix. A read-only check is not enough -- the
# collector must be able to WRITE, so probe a real round trip and clean up.
PROBE="_preflight/${TAG}.probe"
echo "preflight $TAG $(date -u +%FT%TZ)" > /tmp/pf_probe.txt
if "$REPO/ops/gcs_adc.sh" put /tmp/pf_probe.txt "$PROBE" >/dev/null 2>&1; then
  ok "gcs_adc.sh put (user ADC)"
  if "$REPO/ops/gcs_adc.sh" ls "_preflight/" 2>/dev/null | grep -q "$TAG"; then
    ok "gcs_adc.sh ls lists the object it just wrote"
  else
    bad "gcs_adc.sh ls did not show the probe -- listing path is broken (py3.11 f-string class of bug)"
  fi
  if "$REPO/ops/gcs_adc.sh" get "$PROBE" /tmp/pf_probe_back.txt >/dev/null 2>&1 \
     && cmp -s /tmp/pf_probe.txt /tmp/pf_probe_back.txt; then
    ok "gcs_adc.sh get round-trips byte-identically"
  else
    bad "gcs_adc.sh get failed or content differs"
  fi
else
  bad "gcs_adc.sh put failed -- orchestrator cannot write to gs://$BUCKET (check user ADC)"
fi
# gcloud storage from the orchestrator SHOULD fail (it uses the SA). If it ever
# starts working, fine -- but never rely on it: that was round 12's bug.
if timeout 60 gcloud storage ls "gs://$BUCKET/" >/dev/null 2>&1; then
  warn "gcloud storage works here now, but collectors must still use gcs_adc.sh (SA access is not guaranteed)"
else
  ok "gcloud storage is (as expected) unusable here -- collectors must use gcs_adc.sh"
fi

# --------------------------------------------------------------------------
sec "2. the BOX's own identity and bucket writability (round-13b)"
if [ -z "$(rssh 'echo up')" ]; then
  bad "cannot ssh $VM -- is it running?"
else
  ok "ssh to $VM"
  ACCT=$(rssh 'gcloud auth list --filter=status:ACTIVE --format="value(account)"' | tr -d '\r')
  echo "        active account on box: ${ACCT:-<none>}"
  # The real test: can the box actually WRITE to the bucket? Scopes cannot be
  # changed on a running instance, so discovering this after training is fatal.
  BOXPUT=$(rssh "echo probe > /tmp/pf.txt; gcloud storage cp /tmp/pf.txt gs://$BUCKET/_preflight/${TAG}.box >/dev/null 2>&1 && echo yes || echo no")
  if [ "$BOXPUT" = "yes" ]; then
    ok "box can upload to gs://$BUCKET (driver's own cp will work)"
  else
    warn "box CANNOT upload to gs://$BUCKET -- scopes/IAM. This is survivable ONLY if"
    echo "        the round is collected by PULL (ops/collect_r13b.sh pattern). Do not"
    echo "        rely on any 'gcloud storage cp' inside the driver. Scopes cannot be"
    echo "        changed while the instance runs -- recreate the VM to fix properly."
  fi
  # disk headroom on the box
  AVGB=$(rssh "df -BG --output=avail \$HOME | tail -1 | tr -dc '0-9'")
  if [ -n "$AVGB" ] && [ "$AVGB" -ge "$NEEDGB" ]; then
    ok "box disk headroom ${AVGB}G >= ${NEEDGB}G"
  else
    bad "box disk headroom ${AVGB:-?}G < ${NEEDGB}G needed for $TAG artifacts"
  fi
fi

# --------------------------------------------------------------------------
sec "3. orchestrator headroom for a PULL-based collection"
OAV=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
if [ "${OAV:-0}" -ge "$NEEDGB" ]; then
  ok "orchestrator headroom ${OAV}G >= ${NEEDGB}G"
else
  bad "orchestrator headroom ${OAV}G < ${NEEDGB}G -- a pull-based collection will fail mid-way"
fi

# --------------------------------------------------------------------------
sec "4. watchdog (cost backstop + retrieval window)"
WD=$(rssh 'pgrep -af "watchdog" | grep -v watchdogd | head -1')
if [ -n "$WD" ]; then
  ok "a watchdog is running on $VM"
  echo "        $WD"
  # A 3-minute post-exit window is not enough when results must be PULLED.
  GRACE=$(rssh 'grep -oE "GRACE=\\\$\(\( *[0-9]+\*?[0-9]* *\)\)|sleep 180" ~/r*/watchdog*.sh 2>/dev/null | head -1')
  [ "$GRACE" = "sleep 180" ] && warn "watchdog appears to use the old 3-min flush; too short for a pull-based collection"
else
  bad "NO watchdog running on $VM -- an unattended GPU round can run until the billing stops it"
fi

# --------------------------------------------------------------------------
sec "5. contamination surface (round-12's poisoned seed 0)"
STALE=$(rssh 'ls ~/r*/sae-identifiability/results/real/sae_*.pt 2>/dev/null | wc -l')
if [ "${STALE:-0}" -eq 0 ]; then
  ok "no pre-existing sae_*.pt in the box's results dir"
else
  warn "${STALE} sae_*.pt already present on the box -- the driver MUST wipe before training"
  echo "        round 12 was poisoned by exactly one stale file with a valid-looking name."
fi

# --------------------------------------------------------------------------
sec "6. prereg discipline (criteria must predate results)"
cd "$REPO" || exit 2
if [ -z "$(git status --porcelain)" ]; then
  ok "repo clean -- prereg/scorer/evaluator are committed, so the lock timestamp is real"
else
  bad "repo DIRTY. Commit the prereg, scorer and evaluator BEFORE training, or the"
  echo "        'locked before results' claim cannot be substantiated:"
  git status --porcelain | sed 's/^/          /' | head -10
fi
PRE=$(ls notes/prereg-${TAG}*.md 2>/dev/null | head -1)
if [ -n "$PRE" ]; then
  ok "prereg present: $PRE"
  for f in analysis/analyze_${TAG}.py; do
    if [ -f "$f" ]; then ok "evaluator committed: $f"; else bad "missing evaluator $f"; fi
  done
else
  bad "no notes/prereg-${TAG}*.md -- do not start a round without a locked prereg"
fi

# --------------------------------------------------------------------------
sec "7. evaluator runs (schema/width drift, e.g. an amendment dropping a cell)"
# Amendment 2 of round 13b dropped m=8192 AFTER the evaluator was locked. That is
# exactly the kind of drift that only surfaces when the evaluator finally runs --
# by then the GPU money is spent. Synthetic rows, /tmp only, never results/real.
EV=analysis/analyze_${TAG}.py
if [ -f "$EV" ]; then
  if python3 - "$EV" <<'PY' 2>/dev/null
import importlib.util, sys, os, json, random, re, tempfile
path = sys.argv[1]
src = open(path).read()
m = re.search(r"^WIDTHS\s*=\s*(\[[0-9, ]+\])", src, re.M)
widths = json.loads(m.group(1)) if m else [2048]
n = len(re.findall(r"len\(rows\) == (\d+)", src) or ["0"])
exp = re.search(r"len\(rows\) == (\d+)", src)
total = int(exp.group(1)) if exp else len(widths)*2*8
seeds = total // (len(widths)*2)
rows = []
random.seed(0)
for a in ("l1", "topk"):
    for w in widths:
        for s in range(seeds):
            rows.append(dict(sae=f"sae_pythia-1.4b_L12_{a}_x{w//2048}_s{s}.pt", arch=a, m=w,
                expansion=w//2048, seed=s, model="EleutherAI/pythia-1.4b", layer=12,
                theta=0.0, tau=0.30, k=32 if a=="topk" else None,
                lam=None if a=="topk" else 4.5, dead_pct=0.05+0.5*(w/max(widths)),
                fvu=0.07, l0=31.5, rate_single=0.07, rate_family=0.055,
                rate_lost=0.01, sha256=f"{a}{w}{s}".ljust(64,"0"),
                per_letter={c: {"fam_size": 2.0, "clean_latent": True} for c in "abcdefg"}))
fd, tmp = tempfile.mkstemp(suffix=".json"); json.dump(rows, open(tmp,"w"))
os.environ[ "R13B" if "R13B" in src else "RESULTS" ] = tmp
spec = importlib.util.spec_from_file_location("ev", path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    mod.main()
os.unlink(tmp)
PY
  then ok "$EV executes end-to-end on synthetic rows matching its own WIDTHS"
  else bad "$EV FAILED on synthetic input -- fix before spending GPU hours"
  fi
fi

# --------------------------------------------------------------------------
sec "8. environment pins"
if [ -f ENVIRONMENT.md ]; then
  BOXPY=$(rssh 'python3 -c "import sys,torch;print(sys.version.split()[0], torch.__version__)"')
  echo "        box: ${BOXPY:-<unavailable>}"
  if [ -n "$BOXPY" ]; then
    if grep -qF "$(echo "$BOXPY" | awk '{print $2}')" ENVIRONMENT.md; then
      ok "box torch version appears in ENVIRONMENT.md"
    else
      warn "box torch $(echo "$BOXPY" | awk '{print $2}') not found in ENVIRONMENT.md -- results may not be comparable across rounds"
    fi
  fi
else
  warn "no ENVIRONMENT.md"
fi

# --------------------------------------------------------------------------
echo
echo "==================================================================="
echo "PREFLIGHT $TAG: $PASS pass, $WARN warn, $FAIL fail"
if [ "$FAIL" -gt 0 ]; then
  echo "DO NOT START THE ROUND. Each failure above has already cost this project"
  echo "a round or a re-score at least once."
  exit 1
fi
[ "$WARN" -gt 0 ] && echo "Warnings are survivable but must be consciously accepted (e.g. pull-based collection)."
echo "OK to start $TAG."
