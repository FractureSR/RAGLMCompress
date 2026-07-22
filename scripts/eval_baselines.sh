#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 {text|image|audio} PREPARED_DATABASE" >&2
    exit 2
fi

modality=$1
database=$2
case "$modality" in
    text)
        default_codecs="zstd,openzl,cmix"
        ;;
    image)
        default_codecs="zstd,openzl,cmix,png,jpegxl,webp"
        ;;
    audio)
        default_codecs="zstd,openzl,cmix,flac"
        ;;
    *)
        echo "unsupported modality: $modality" >&2
        exit 2
        ;;
esac

output_dir=${OUTPUT_DIR:-results/baselines/$modality}
codecs=${BASELINE_CODECS:-$default_codecs}
delta_codecs=${DELTA_CODECS:-zstd-patch,bsdiff,open-vcdiff}
candidate_policy=${CANDIDATE_POLICY:-retrieve}
m=${M:-16}

mkdir -p "$output_dir"

split_args=()
if [[ -n "${N_SAMPLES:-}" ]]; then
    split_args+=(--n-samples "$N_SAMPLES")
fi
if [[ -n "${CALIB_SAMPLES:-}" ]]; then
    split_args+=(--calib-samples "$CALIB_SAMPLES")
fi

python evaluation/eval_baselines.py \
    --database "$database" \
    --modality "$modality" \
    --codecs "$codecs" \
    "${split_args[@]}" \
    --allow-missing \
    --output "$output_dir/standalone.csv" \
    --summary-output "$output_dir/standalone_summary.csv"

python evaluation/eval_delta_baselines.py \
    --database "$database" \
    --modality "$modality" \
    --codecs "$delta_codecs" \
    --candidate-policy "$candidate_policy" \
    --m "$m" \
    "${split_args[@]}" \
    --keep-going \
    --output "$output_dir/delta.csv"
