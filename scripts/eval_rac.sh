#!/usr/bin/env bash
#
# Evaluate oracle RAC on the held-out eval docs saved by prepare_rac_data.py.
#
# Override via env:
#   DATA MODEL M_VALUES DEVICE N_DOCS OUTDIR DECODE_CHECK EXTRA_ARGS
#
# Examples:
#   bash scripts/eval_rac.sh
#   M_VALUES="1 4 16" N_DOCS=200 bash scripts/eval_rac.sh
#   EXTRA_ARGS="--cascade --cascade-retriever" bash scripts/eval_rac.sh
set -euo pipefail

DATA="${DATA:-results/rac_c_db}"
MODEL="${MODEL:-pretrained/SmolLM2-135M}"
M_VALUES="${M_VALUES:-16}"
DEVICE="${DEVICE:-cuda}"
N_DOCS="${N_DOCS:-}"
OUTDIR="${OUTDIR:-results/rac_eval}"
DECODE_CHECK="${DECODE_CHECK:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$(dirname "$0")/.."

[[ -d "$DATA" ]] || { echo "ERROR: database dir not found: $DATA" >&2; exit 1; }
[[ -f "$DATA/base_chunks.json" ]] || { echo "ERROR: missing $DATA/base_chunks.json" >&2; exit 1; }
[[ -d "$DATA/retriever" ]] || { echo "ERROR: missing $DATA/retriever" >&2; exit 1; }
[[ -f "$DATA/meta.json" ]] || { echo "ERROR: missing $DATA/meta.json" >&2; exit 1; }

mkdir -p "$OUTDIR"
read -r -a EXTRA <<< "$EXTRA_ARGS"

N_DOCS_ARG=()
if [[ -n "$N_DOCS" ]]; then
    N_DOCS_ARG+=(--n-docs "$N_DOCS")
fi

run_eval () {
    local m="$1"; shift
    python evaluation/eval_rac.py \
        --database "$DATA" \
        --model "$MODEL" \
        --m "$m" \
        --device "$DEVICE" \
        "$@" \
        "${EXTRA[@]}"
}

echo "database=$DATA  model=$MODEL  m={$M_VALUES}  device=$DEVICE  n_docs=${N_DOCS:-all}"

if [[ "$DECODE_CHECK" -gt 0 ]]; then
    echo
    echo "### decode check — $DECODE_CHECK doc(s), m=1"
    run_eval 1 --n-docs "$DECODE_CHECK" --output "$OUTDIR/decode_check.csv"
fi

for m in $M_VALUES; do
    echo
    echo "### RAC ratio — m=$m"
    run_eval "$m" --no-decompress "${N_DOCS_ARG[@]}" \
        --output "$OUTDIR/eval_rac_m${m}.csv"
done

echo
echo "CSVs -> $OUTDIR/"
