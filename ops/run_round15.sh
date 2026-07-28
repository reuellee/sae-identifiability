#!/usr/bin/env bash
# Round 15 driver: Gemma Scope 2 cross-validation, CPU-only ephemeral VM.
# Prereg: notes/prereg-round15-gemmascope-crossval.md (LOCKED with the harness).
#
# LAPTOP-DRIVEN (Git Bash). The laptop's gcloud ssh uses plink and rejects
# OpenSSH -o flags (round-12 note), so all remote calls use --command=. On the
# FIRST contact with a new VM, plink prompts to cache the host key — pipe `y`:
#   echo y | gcloud compute ssh r15 --zone=... --command="echo ok"
# The VM is created --no-service-account --no-scopes (no GCS access; HF repos
# are public). Self-poweroff watchdog armed on the full run with a 1h
# post-completion grace so collection can happen first (GPT pre-lock P2.10).
# Staged + resumable: setup | deps | push | dl | pilot | full | status | collect | clean
set -uo pipefail

NAME=${NAME:-r15}
TYPE=${TYPE:-e2-standard-8}
ZONE=${ZONE:-us-central1-a}
PROJECT=${PROJECT:-project-ebd5a273-53ea-4c8b-81a}
REPO=${REPO:-/e/Projects/sae-identifiability}
# The 8 registered cells (prereg Design): width series + L0 series + layer series.
DIRS="layer_13_width_16k_l0_medium layer_13_width_65k_l0_medium layer_13_width_262k_l0_medium layer_13_width_65k_l0_small layer_13_width_65k_l0_big layer_7_width_16k_l0_medium layer_17_width_16k_l0_medium layer_22_width_16k_l0_medium"

log() { echo "$(date -u +%H:%M:%S) $*"; }
vssh() { gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --command="$1" 2>&1; }
vscp() { gcloud compute scp --project="$PROJECT" --zone="$ZONE" "$@" 2>&1 | tail -1; }

case "${1:-}" in
setup)
  gcloud compute instances create "$NAME" --project="$PROJECT" --zone="$ZONE" \
    --machine-type="$TYPE" --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=50GB --boot-disk-type=pd-balanced \
    --no-service-account --no-scopes
  ;;
deps)
  vssh 'sudo apt-get update -qq >/dev/null 2>&1; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip >/dev/null 2>&1
        pip3 -q install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1
        pip3 -q install --break-system-packages "transformers>=4.50" safetensors scikit-learn numpy huggingface_hub 2>&1 | tail -1
        python3 -c "import torch,sklearn,transformers,safetensors;print(torch.__version__, transformers.__version__)"'
  ;;
push)
  vssh 'mkdir -p ~/r15/results'
  vscp "$REPO/experiments/gemmascope_crossval.py" "$REPO/analysis/analyze_round15.py" \
       "$REPO/experiments/gemmascope_indist.py" "$REPO/ops/vm_watchdog.sh" "$NAME:r15/"
  vssh 'chmod +x ~/r15/vm_watchdog.sh && ls ~/r15/'
  ;;
dl)
  # Downloads + provenance manifest (GPT pre-lock P2.9): resolved HF commit
  # SHAs for both repos, per-file sha256, pip versions.
  vssh "python3 - <<'EOF'
from huggingface_hub import hf_hub_download, HfApi
import shutil, os, hashlib, json
api = HfApi()
prov = {'sae_repo_sha': api.model_info('google/gemma-scope-2-1b-pt').sha,
        'model_repo_sha': api.model_info('unsloth/gemma-3-1b-pt').sha,
        'files': {}}
dirs = \"$DIRS\".split()
for d in dirs:
    for f in ('params.safetensors', 'config.json'):
        p = hf_hub_download('google/gemma-scope-2-1b-pt', f'resid_post/{d}/{f}')
        dst = os.path.expanduser(f'~/r15/sae/{d}/{f}')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(p, dst)
        h = hashlib.sha256(open(dst,'rb').read()).hexdigest()
        prov['files'][f'{d}/{f}'] = h
    print('ok', d, flush=True)
json.dump(prov, open(os.path.expanduser('~/r15/results/PROVENANCE.json'),'w'), indent=1)
print('provenance written')
EOF
pip3 freeze 2>/dev/null | grep -iE 'torch|transformers|safetensors|scikit|numpy|scipy|huggingface' >> ~/r15/results/PROVENANCE.txt
ls ~/r15/sae/"
  ;;
pilot)
  # Prereg Pilot: SMOKE on the (16k, medium, layer 13) cell only, PLUS an
  # official-loader oracle check (GPT pre-lock P2.7): sae-lens encode vs ours
  # on a small batch, tight tolerance. Sets NOTHING; fixes commit as
  # pre-results amendments if defects surface.
  vssh 'pip3 -q install --break-system-packages sae-lens 2>&1 | tail -1
        cd ~/r15 && SMOKE=1 MODE=words LAYERS=13 OUTDIR=~/r15/results python3 gemmascope_crossval.py 2>&1 | tail -4
        cd ~/r15 && SMOKE=1 MODE=score LAYERS=13 OUTDIR=~/r15/results \
          WORDS_DIR=~/r15/results \
          SAE_DIRS=~/r15/sae/layer_13_width_16k_l0_medium \
          OUT=~/r15/results/pilot_row.json python3 gemmascope_crossval.py 2>&1 | tail -6
        python3 - <<'\''EOF'\''
import torch, numpy as np, os
try:
    from sae_lens import SAE
    sae = SAE.from_pretrained(release="gemma-scope-2-1b-pt-resid_post",
                              sae_id="layer_13_width_16k_l0_medium")
    if isinstance(sae, tuple): sae = sae[0]
    W = torch.load(os.path.expanduser("~/r15/results/words_gemma-3-1b_L13.pt"), weights_only=True)
    x = W["acts"][:64]
    from safetensors.torch import load_file
    P = load_file(os.path.expanduser("~/r15/sae/layer_13_width_16k_l0_medium/params.safetensors"))
    pre = x @ P["W_enc"].float() + P["b_enc"].float()
    ours = torch.relu(pre) * (pre > P["threshold"].float()).float()
    theirs = sae.encode(x)
    d = (ours - theirs).abs().max().item()
    print(f"ORACLE max|ours-saelens| = {d:.2e}  ({'PASS' if d < 1e-3 else 'FAIL'})")
except Exception as e:
    print(f"ORACLE UNAVAILABLE ({type(e).__name__}: {e}) - relying on the conformance gate")
EOF'
  ;;
full)
  vssh 'cd ~/r15 && rm -f ~/r15/results/words_gemma-3-1b_L*.pt ~/r15/results/round15_rows.json ~/r15/results/results_round15.txt
        cat > ~/r15/run_full.sh <<'"'"'EOS'"'"'
#!/usr/bin/env bash
# Fail-closed full run (GPT pre-lock P1.3): DONE_ROUND15 only after both
# result files exist and are non-empty.
set -euo pipefail
cd ~/r15
D=~/r15/sae
MODE=words LAYERS=7,13,17,22 OUTDIR=~/r15/results python3 gemmascope_crossval.py
SAE_DIRS=$(ls -d $D/*/ | tr "\n" "," | sed "s/,$//")
MODE=score LAYERS=7,13,17,22 OUTDIR=~/r15/results \
  WORDS_DIR=~/r15/results \
  SAE_DIRS="$SAE_DIRS" OUT=~/r15/results/round15_rows.json \
  python3 gemmascope_crossval.py
SAE_ROOT=~/r15/sae OUT=~/r15/results/indist.json python3 gemmascope_indist.py
ROWS=~/r15/results/round15_rows.json INDIST=~/r15/results/indist.json \
  OUT=~/r15/results/results_round15.txt python3 analyze_round15.py
test -s ~/r15/results/round15_rows.json
test -s ~/r15/results/results_round15.txt
echo DONE_ROUND15
EOS
        chmod +x ~/r15/run_full.sh
        nohup setsid ~/r15/run_full.sh > ~/r15/full.log 2>&1 < /dev/null &
        PID=$!
        echo $PID > ~/r15/job.pid
        nohup setsid ~/r15/vm_watchdog.sh $PID 6 3600 > /dev/null 2>&1 < /dev/null &
        echo "launched pid $PID (watchdog: 6h cap, 1h post-exit grace)"'
  ;;
status)
  vssh 'echo "job pid: $(cat ~/r15/job.pid 2>/dev/null) alive: $(kill -0 $(cat ~/r15/job.pid 2>/dev/null) 2>/dev/null && echo yes || echo no)"
        tail -8 ~/r15/full.log 2>/dev/null; ls -la ~/r15/results/ 2>/dev/null | tail -8'
  ;;
collect)
  mkdir -p "$REPO/results/real"
  for f in round15_rows.json results_round15.txt indist.json PROVENANCE.json PROVENANCE.txt; do
    vscp "$NAME:r15/results/$f" "$REPO/results/real/" || log "missing: $f"
  done
  vscp "$NAME:r15/full.log" "$REPO/results/real/round15_full.log"
  vscp "$NAME:vm_watchdog.log" "$REPO/results/real/round15_watchdog.log" || true
  log "collected. Upload words + rows to GCS from the laptop with ops/gcs_adc.sh (user ADC)."
  ;;
clean)
  gcloud compute instances delete "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet
  ;;
*)
  echo "usage: $0 setup|deps|push|dl|pilot|full|status|collect|clean"; exit 2
  ;;
esac
