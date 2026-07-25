#!/usr/bin/env bash
# GCS access via USER ADC, not the attached service account.
#
# ROUND-12 OPS BUG this fixes: drive_r12.sh collected results with `gcloud storage`
# while running as the orchestrator's attached SA (claude-mobile@...), which has
# ZERO access to gs://sae-identifiability-artifacts-ebd5a273 (no list/get/describe).
# Collection silently failed ("no results_round12.txt collected"), the L4 was
# deleted, and no SUMMARY was committed. The results were fine in GCS the whole
# time -- only the reader was wrong.
#
# The orchestrator DOES have user application-default credentials at
# ~/.config/gcloud/application_default_credentials.json (type=authorized_user).
# Those work. So: mint a token from ADC and hit the GCS JSON/media API directly.
# Do NOT use `gcloud storage` / `gsutil` here -- they pick up the SA.
#
# Usage:
#   ops/gcs_adc.sh ls [PREFIX]
#   ops/gcs_adc.sh get OBJECT LOCAL_PATH
#   ops/gcs_adc.sh put LOCAL_PATH OBJECT
set -euo pipefail

BUCKET="${BUCKET:-sae-identifiability-artifacts-ebd5a273}"
API="https://storage.googleapis.com/storage/v1/b/${BUCKET}/o"
UPLOAD="https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o"

tok() {
  gcloud auth application-default print-access-token 2>/dev/null || {
    echo "ERROR: no user ADC on this box. Round-12's bug was using the SA instead." >&2
    echo "Fix: gcloud auth application-default login" >&2
    exit 1
  }
}

urlenc() { python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }

case "${1:-}" in
  ls)
    T=$(tok); PREFIX="${2:-}"
    curl -sf -H "Authorization: Bearer $T" \
      "${API}?prefix=$(urlenc "$PREFIX")&fields=items(name,size,updated)" |
      python3 -c 'import json,sys
d=json.load(sys.stdin)
for it in d.get("items",[]):
    size, upd, name = int(it["size"]), it.get("updated","")[:19], it["name"]
    print("%14s  %s  %s" % (format(size,","), upd, name))'
    ;;
  get)
    T=$(tok); OBJ="$2"; DEST="$3"
    mkdir -p "$(dirname "$DEST")"
    curl -sf -H "Authorization: Bearer $T" -o "$DEST" \
      "${API}/$(urlenc "$OBJ")?alt=media"
    echo "got ${OBJ} -> ${DEST} ($(stat -c%s "$DEST") bytes)"
    ;;
  put)
    T=$(tok); SRC="$2"; OBJ="$3"
    curl -sf -X POST -H "Authorization: Bearer $T" \
      -H "Content-Type: application/octet-stream" \
      --data-binary "@${SRC}" \
      "${UPLOAD}?uploadType=media&name=$(urlenc "$OBJ")" >/dev/null
    echo "put ${SRC} -> gs://${BUCKET}/${OBJ}"
    ;;
  *)
    sed -n '2,22p' "$0"; exit 1
    ;;
esac
