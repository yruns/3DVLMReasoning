#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/nr3d}"
RAW="${DATA_ROOT}/raw"
mkdir -p "${RAW}"

require_nonempty() {
    local path="$1"
    if [[ ! -s "${path}" ]]; then
        echo "ERROR: required file is missing or empty: ${path}" >&2
        exit 1
    fi
}

validate_json_string_array() {
    local path="$1"
    python - "${path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
    raise SystemExit(f"ERROR: {path}: expected JSON list of strings")
PY
}

validate_bad_contexts_csv() {
    local path="$1"
    local header
    header=$(head -n 1 "${path}" | tr -d '\r')
    if [[ "${header}" != "stimulus_id" ]] && [[ "${header}" != *,* ]]; then
        echo "ERROR: ${path}: expected CSV header row" >&2
        exit 1
    fi
    local nlines
    nlines=$(wc -l < "${path}")
    if [[ "${nlines}" -lt 2 ]]; then
        echo "ERROR: ${path}: expected header row plus at least one data row" >&2
        exit 1
    fi
}

# 1. NR3D CSV (10.7 MB)
NR3D_DRIVE_ID="1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC"
if [[ ! -f "${RAW}/nr3d.csv" ]]; then
    if command -v gdown >/dev/null 2>&1; then
        gdown --id "${NR3D_DRIVE_ID}" -O "${RAW}/nr3d.csv"
    else
        # confirm=t bypasses Drive's HTML interstitial for files <100MB.
        curl -fL --retry 3 --retry-delay 2 -o "${RAW}/nr3d.csv" \
            "https://drive.usercontent.google.com/download?id=${NR3D_DRIVE_ID}&export=download&confirm=t"
    fi
fi

# 2. Auxiliaries from referit3d eccv branch.
RAW_BASE="https://raw.githubusercontent.com/referit3d/referit3d/eccv/referit3d/data/language/nr3d"
for f in train_scans.txt test_scans.txt manually_inspected_bad_contexts.csv; do
    if [[ ! -f "${RAW}/${f}" ]]; then
        curl -fL --retry 3 --retry-delay 2 -o "${RAW}/${f}" "${RAW_BASE}/${f}"
    fi
done

# 3. Sanity checks (FAIL-LOUD).
require_nonempty "${RAW}/nr3d.csv"
NLINES=$(wc -l < "${RAW}/nr3d.csv")
if [[ "${NLINES}" -ne 41504 ]]; then
    echo "ERROR: ${RAW}/nr3d.csv has ${NLINES} lines, expected 41504 (41503 data + 1 header)" >&2
    exit 1
fi
require_nonempty "${RAW}/train_scans.txt"
require_nonempty "${RAW}/test_scans.txt"
require_nonempty "${RAW}/manually_inspected_bad_contexts.csv"
validate_json_string_array "${RAW}/train_scans.txt"
validate_json_string_array "${RAW}/test_scans.txt"
validate_bad_contexts_csv "${RAW}/manually_inspected_bad_contexts.csv"

echo "OK: NR3D raw data ready at ${RAW}"
echo "Note: ScanNet meshes / aggregations are required for ScanNet-raw bbox derivation"
echo "      and are TOS-gated. This pipeline uses EmbodiedScan PKL bboxes instead, so"
echo "      no extra ScanNet download is needed for the plumbing pass."
