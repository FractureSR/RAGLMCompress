#!/usr/bin/env bash
# Main experiment: every baseline vs. oracle RAC, on one prepared held-out split
# per dataset.  Assumes the datasets are already downloaded under $DATA_DIR.
#
# Per dataset the pipeline is:
#   1. prepare   utils/prepare_rac_data_{llm,bgpt}.py   -> base corpus + held-out split + index
#   2. neural    evaluation/eval_{llm,bgpt}.py          -> no-condition LM/bGPT baseline
#   3. rac       evaluation/eval_rac_{llm,bgpt}.py      -> ours (fixed index, and calibrated)
#   4. codecs    evaluation/eval_baselines.py           -> zstd/openzl/cmix/png/jxl/webp/flac
#   5. delta     evaluation/eval_delta_baselines.py     -> zstd --patch-from / bsdiff / vcdiff
#
# Every stage evaluates the SAME first N samples of the split persisted in step 1
# (the last CALIB samples of the loaded prefix are reserved for index calibration
# and excluded from every method's test set).
#
# Comparability notes — read before making a table:
# Every baseline is run at its strongest: whole samples (never a matched
# window), the LM baseline gets the full context window, and zstd is also given
# a dictionary trained on the same base corpus RAC retrieves from.
#
# Image patches are clipped to the image, so every track codes exactly the
# source pixels and each modality's original_bytes is the true sample size.
# Tabulate with scripts/collect_results.py, which also puts the conventional
# codecs (tight RGB8) on that same denominator and reports bits per pixel.
#
# Usage:
#   scripts/run_main_experiment.sh                  # everything
#   DATASETS="arxiv_cl eurosat_forest" scripts/run_main_experiment.sh
#   STAGES="prepare rac" DEVICE=cuda:1 scripts/run_main_experiment.sh
#   DRY_RUN=1 scripts/run_main_experiment.sh        # print the plan only
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR=${DATA_DIR:-datasets}
OUT_DIR=${OUT_DIR:-results/main}
LLM_MODEL=${LLM_MODEL:-pretrained/SmolLM2-135M}
BGPT_IMAGE_MODEL=${BGPT_IMAGE_MODEL:-pretrained/bgpt/weights-image.pth}
BGPT_AUDIO_MODEL=${BGPT_AUDIO_MODEL:-pretrained/bgpt/weights-audio.pth}
DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-42}
M=${M:-16}                       # retrieval candidates the oracle tries per piece
BASE_FRAC=${BASE_FRAC:-0.5}

# Corpus sizes.  PREP_* is what gets split into base+held-out; N_* is how many
# held-out samples every method is scored on; CALIB_* is the index-calibration
# tail excluded from all test sets.
TEXT_PREP_DOCS=${TEXT_PREP_DOCS:-300}
TEXT_N_DOCS=${TEXT_N_DOCS:-100}
TEXT_CALIB_DOCS=${TEXT_CALIB_DOCS:-10}
IMAGE_PREP_SAMPLES=${IMAGE_PREP_SAMPLES:-2000}
IMAGE_N_SAMPLES=${IMAGE_N_SAMPLES:-200}
IMAGE_CALIB_SAMPLES=${IMAGE_CALIB_SAMPLES:-20}
AUDIO_PREP_SAMPLES=${AUDIO_PREP_SAMPLES:-200}
AUDIO_N_SAMPLES=${AUDIO_N_SAMPLES:-50}
AUDIO_CALIB_SAMPLES=${AUDIO_CALIB_SAMPLES:-10}

# Retrieval-unit / payload geometry.  See the comparability notes above; these
# defaults keep each modality's neural baseline and RAC on a shared denominator.
TEXT_CHUNK_TOKENS=${TEXT_CHUNK_TOKENS:-512}      # one condition = 512 LM tokens
TEXT_CTX_TOKENS=${TEXT_CTX_TOKENS:-8191}         # SmolLM2-135M max_position_embeddings - 1, for a bos token
TEXT_PAYLOAD_TOKENS=$((TEXT_CTX_TOKENS - TEXT_CHUNK_TOKENS))
# The no-condition LM baseline gets the whole context window; reserving room for
# a condition is a cost RAC pays and the baseline should not.
TEXT_BASELINE_TOKENS=${TEXT_BASELINE_TOKENS:-$TEXT_CTX_TOKENS}
# 64x64 and 128x128 are both multiples of 32, so a 32x8 condition is always a
# full 768 B chunk and the remaining 3072-768 = 2304 B body is exactly 32x24.
# Change all three together: eval_rac_bgpt rejects a payload rectangle that does
# not fill the body beside the condition.
IMAGE_COND_W=${IMAGE_COND_W:-32}                 # condition patch  32x8  =  768 B
IMAGE_COND_H=${IMAGE_COND_H:-8}
IMAGE_PAYLOAD_W=${IMAGE_PAYLOAD_W:-32}           # RAC payload      32x24 = 2304 B
IMAGE_PAYLOAD_H=${IMAGE_PAYLOAD_H:-24}
IMAGE_BASE_PATCH_W=${IMAGE_BASE_PATCH_W:-32}     # no-condition bGPT 32x32 = 3072 B
IMAGE_BASE_PATCH_H=${IMAGE_BASE_PATCH_H:-32}
AUDIO_CHUNK_BYTES=${AUDIO_CHUNK_BYTES:-512}      # one condition = 512 PCM bytes
AUDIO_KGRAM=${AUDIO_KGRAM:-8}
IMAGE_KGRAM=${IMAGE_KGRAM:-4}

# Conventional codecs.  cmix is excluded by default: it is orders of magnitude
# slower than everything else here.  WITH_CMIX=1 adds it back.
WITH_CMIX=${WITH_CMIX:-0}
TEXT_CODECS=${TEXT_CODECS:-zstd,openzl}
IMAGE_CODECS=${IMAGE_CODECS:-zstd,openzl,png,jpegxl,webp}
AUDIO_CODECS=${AUDIO_CODECS:-zstd,openzl,flac}
DELTA_CODECS=${DELTA_CODECS:-zstd-patch,bsdiff,open-vcdiff}
# zstd with a dictionary trained on the base corpus is the closest conventional
# analogue of RAC: same base split, same "shared state" idea, no per-sample id.
# It runs as its own step so a dictionary-training failure cannot take out the
# plain-codec rows.
WITH_ZSTD_DICT=${WITH_ZSTD_DICT:-1}
DICT_SIZE=${DICT_SIZE:-112640}
if [[ "$WITH_CMIX" == "1" ]]; then
    TEXT_CODECS="$TEXT_CODECS,cmix"
    IMAGE_CODECS="$IMAGE_CODECS,cmix"
    AUDIO_CODECS="$AUDIO_CODECS,cmix"
fi

# VERIFY=1 runs each RAC config once over VERIFY_N samples *with* decoding first,
# so roundtrip_ok is confirmed before the fast --no-decompress measurement runs.
VERIFY=${VERIFY:-1}
VERIFY_N=${VERIFY_N:-3}
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

ALL_DATASETS="arxiv_cl arxiv_ar eurlex_caselaw eurlex_regulation \
eurosat_forest eurosat_industry medmnist_retina \
ljspeech vctk_p225"
DATASETS=${DATASETS:-$ALL_DATASETS}
ALL_STAGES="prepare neural rac codecs delta collect"
STAGES=${STAGES:-$ALL_STAGES}

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/run_summary.tsv"
FAILURES=0
STEPS_RUN=0
STEPS_SKIPPED=0

# ---------------------------------------------------------------------------
# Dataset table:  name -> modality, source path
# ---------------------------------------------------------------------------
dataset_modality() {
    case "$1" in
        arxiv_cl|arxiv_ar|eurlex_caselaw|eurlex_regulation) echo text ;;
        eurosat_forest|eurosat_industry|medmnist_retina) echo image ;;
        ljspeech|vctk_p225) echo audio ;;
        *) echo "unknown" ;;
    esac
}

dataset_path() {
    case "$1" in
        arxiv_cl)           echo "$DATA_DIR/arxiv_tex/cs_cl.jsonl" ;;
        arxiv_ar)           echo "$DATA_DIR/arxiv_tex/cs_ar.jsonl" ;;
        eurlex_caselaw)     echo "$DATA_DIR/eurlex/caselaw.jsonl" ;;
        eurlex_regulation)  echo "$DATA_DIR/eurlex/regulation.jsonl" ;;
        eurosat_forest)     echo "$DATA_DIR/eurosat/Forest" ;;
        eurosat_industry)   echo "$DATA_DIR/eurosat/Industry" ;;
        # The registered loader names the test split explicitly, so the
        # dataset path has to include it (utils/img_utils.py).
        medmnist_retina)    echo "$DATA_DIR/medmnist/retinamnist128/test" ;;
        ljspeech)           echo "$DATA_DIR/ljspeech_wav" ;;
        vctk_p225)          echo "$DATA_DIR/vctk/p225" ;;
        *)                  echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
# Step runner: log, skip when already done, never abort the whole sweep
# ---------------------------------------------------------------------------
stage_enabled() { [[ " $STAGES " == *" $1 "* ]]; }

run_step() {
    # run_step <dataset> <step-name> <marker-path> <cmd...>
    local dataset=$1 step=$2 marker=$3
    shift 3
    local log_dir="$OUT_DIR/$dataset/logs"
    local log="$log_dir/$step.log"
    mkdir -p "$log_dir"

    if [[ "$SKIP_EXISTING" == "1" && -e "$marker" ]]; then
        echo "  [skip] $dataset/$step (exists: $marker)"
        STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
        return 0
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [plan] $dataset/$step: $*"
        return 0
    fi

    echo "  [run ] $dataset/$step -> $log"
    local start=$SECONDS
    if "$@" >"$log" 2>&1; then
        local elapsed=$((SECONDS - start))
        # Steps with no CSV of their own record completion with a .done marker.
        [[ "$marker" == *.done ]] && touch "$marker"
        echo "  [ ok ] $dataset/$step (${elapsed}s)"
        printf '%s\t%s\tok\t%s\n' "$dataset" "$step" "$elapsed" >>"$SUMMARY"
        STEPS_RUN=$((STEPS_RUN + 1))
    else
        local elapsed=$((SECONDS - start))
        echo "  [FAIL] $dataset/$step (${elapsed}s) — tail of $log:"
        tail -n 15 "$log" | sed 's/^/         /'
        printf '%s\t%s\tFAILED\t%s\n' "$dataset" "$step" "$elapsed" >>"$SUMMARY"
        FAILURES=$((FAILURES + 1))
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Preflight: fail early and loudly on things that would break mid-sweep
# ---------------------------------------------------------------------------
preflight() {
    echo "== Preflight =================================================="
    local problems=0

    for model in "$LLM_MODEL" "$BGPT_IMAGE_MODEL" "$BGPT_AUDIO_MODEL"; do
        if [[ -e "$model" ]]; then
            echo "  model    OK      $model"
        else
            echo "  model    MISSING $model"
            problems=$((problems + 1))
        fi
    done

    for dataset in $DATASETS; do
        local path modality
        path=$(dataset_path "$dataset")
        modality=$(dataset_modality "$dataset")
        if [[ -z "$path" || "$modality" == "unknown" ]]; then
            echo "  data     UNKNOWN dataset name: $dataset"
            problems=$((problems + 1))
        elif [[ -e "$path" ]]; then
            echo "  data     OK      $dataset -> $path"
        else
            echo "  data     MISSING $dataset -> $path"
            problems=$((problems + 1))
        fi
    done

    # A registered loader must resolve every dataset path, otherwise prepare
    # dies after the sweep has already started.
    echo "  -- registered dataset loaders --"
    for dataset in $DATASETS; do
        local path modality
        path=$(dataset_path "$dataset")
        modality=$(dataset_modality "$dataset")
        [[ -z "$path" ]] && continue
        if ! python - "$modality" "$path" <<'PY'
import sys
modality, path = sys.argv[1], sys.argv[2]
finder = {
    "text": ("utils.text_utils", "_find_text_loader"),
    "image": ("utils.img_utils", "_find_image_loader"),
    "audio": ("utils.audio_utils", "_find_audio_loader"),
}[modality]
module = __import__(finder[0], fromlist=[finder[1]])
try:
    loader = getattr(module, finder[1])(path)
except Exception as exc:
    print(f"  loader   MISSING {path}: {type(exc).__name__}")
    print(f"           register one in {finder[0]} whose name is a substring "
          f"of the path (see the module docstring)")
    raise SystemExit(1)
print(f"  loader   OK      {path} -> {loader.__name__}")
PY
        then
            problems=$((problems + 1))
        fi
    done

    echo "  -- external codec binaries --"
    local need_bins="zstd"
    stage_enabled codecs && need_bins="$need_bins zli cjxl djxl flac"
    stage_enabled delta && need_bins="$need_bins bsdiff bspatch vcdiff"
    [[ "$WITH_CMIX" == "1" ]] && need_bins="$need_bins cmix"
    for binary in $need_bins; do
        if command -v "$binary" >/dev/null 2>&1; then
            echo "  binary   OK      $binary"
        else
            # Optional codecs are skipped by --allow-missing / --keep-going.
            echo "  binary   MISSING $binary (that codec's rows will be skipped)"
        fi
    done

    echo "==============================================================="
    if [[ "$problems" -gt 0 ]]; then
        echo "Preflight found $problems blocking problem(s); fix them or narrow"
        echo "DATASETS=... before running the sweep." >&2
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Per-modality pipelines
# ---------------------------------------------------------------------------
run_text() {
    local dataset=$1 source_path=$2
    local root="$OUT_DIR/$dataset" db="$OUT_DIR/$dataset/db" csv="$OUT_DIR/$dataset/csv"
    mkdir -p "$csv"

    if stage_enabled prepare; then
        run_step "$dataset" prepare "$db/meta.json" \
            python utils/prepare_rac_data_llm.py \
                --dataset "$source_path" \
                --n-docs "$TEXT_PREP_DOCS" \
                --base-frac "$BASE_FRAC" \
                --chunk-size "$TEXT_CHUNK_TOKENS" --chunk-overlap 0 \
                --retriever bm25 \
                --seed "$SEED" \
                --model "$LLM_MODEL" \
                --out "$db" || return 1
    fi
    if [[ ! -f "$db/meta.json" && "$DRY_RUN" != "1" ]]; then
        echo "  [skip] $dataset: no prepared database at $db"
        return 1
    fi

    # No-condition LM baseline on the same held-out docs, same window size.
    if stage_enabled neural; then
        run_step "$dataset" llm_baseline "$csv/llm_baseline.csv" \
            python evaluation/eval_llm.py \
                --dataset "$db/eval_docs.jsonl" \
                --model "$LLM_MODEL" \
                --n-docs "$TEXT_N_DOCS" \
                --max-tokens "$TEXT_BASELINE_TOKENS" \
                --device "$DEVICE" \
                --no-decompress \
                --output "$csv/llm_baseline.csv"
    fi

    if stage_enabled rac; then
        if [[ "$VERIFY" == "1" ]]; then
            run_step "$dataset" rac_verify "$root/logs/rac_verify.done" \
                python evaluation/eval_rac_llm.py \
                    --database "$db" --model "$LLM_MODEL" \
                    --n-docs "$VERIFY_N" --m "$M" \
                    --device "$DEVICE"
        fi
        run_step "$dataset" rac_fixed "$csv/rac_fixed.csv" \
            python evaluation/eval_rac_llm.py \
                --database "$db" --model "$LLM_MODEL" \
                --n-docs "$TEXT_N_DOCS" --calib-docs "$TEXT_CALIB_DOCS" --m "$M" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/rac_fixed.csv"
        run_step "$dataset" rac_calibrated "$csv/rac_calibrated.csv" \
            python evaluation/eval_rac_llm.py \
                --database "$db" --model "$LLM_MODEL" \
                --n-docs "$TEXT_N_DOCS" --calib-docs "$TEXT_CALIB_DOCS" --m "$M" \
                --calibrate --save-index "$root/index_calibrated.json" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/rac_calibrated.csv"
    fi

    run_conventional "$dataset" text "$db" "$csv" \
        "$TEXT_CODECS" "$TEXT_N_DOCS" "$TEXT_CALIB_DOCS"
}

run_image() {
    local dataset=$1 source_path=$2
    local db="$OUT_DIR/$dataset/db" csv="$OUT_DIR/$dataset/csv"
    mkdir -p "$csv"

    if stage_enabled prepare; then
        run_step "$dataset" prepare "$db/meta.json" \
            python utils/prepare_rac_data_bgpt.py \
                --dataset "$source_path" --modality image \
                --n-samples "$IMAGE_PREP_SAMPLES" \
                --base-frac "$BASE_FRAC" \
                --image-patch-width "$IMAGE_COND_W" \
                --image-patch-height "$IMAGE_COND_H" \
                --retriever bm25 --kgram "$IMAGE_KGRAM" \
                --seed "$SEED" \
                --out "$db" || return 1
    fi
    if [[ ! -f "$db/meta.json" && "$DRY_RUN" != "1" ]]; then
        echo "  [skip] $dataset: no prepared database at $db"
        return 1
    fi

    if stage_enabled neural; then
        run_step "$dataset" bgpt_baseline "$csv/bgpt_baseline.csv" \
            python evaluation/eval_bgpt.py \
                --modality image \
                --dataset "$db/eval_samples.pkl" \
                --model "$BGPT_IMAGE_MODEL" \
                --n-samples "$IMAGE_N_SAMPLES" \
                --image-patch-width "$IMAGE_BASE_PATCH_W" \
                --image-patch-height "$IMAGE_BASE_PATCH_H" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/bgpt_baseline.csv"
    fi

    if stage_enabled rac; then
        local root="$OUT_DIR/$dataset"
        if [[ "$VERIFY" == "1" ]]; then
            run_step "$dataset" rac_verify "$root/logs/rac_verify.done" \
                python evaluation/eval_rac_bgpt.py \
                    --database "$db" --model "$BGPT_IMAGE_MODEL" \
                    --n-samples "$VERIFY_N" --m "$M" \
                    --image-payload-width "$IMAGE_PAYLOAD_W" \
                    --image-payload-height "$IMAGE_PAYLOAD_H" \
                    --device "$DEVICE"
        fi
        run_step "$dataset" rac_fixed "$csv/rac_fixed.csv" \
            python evaluation/eval_rac_bgpt.py \
                --database "$db" --model "$BGPT_IMAGE_MODEL" \
                --n-samples "$IMAGE_N_SAMPLES" --calib-samples "$IMAGE_CALIB_SAMPLES" \
                --m "$M" \
                --image-payload-width "$IMAGE_PAYLOAD_W" \
                --image-payload-height "$IMAGE_PAYLOAD_H" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/rac_fixed.csv"
        run_step "$dataset" rac_calibrated "$csv/rac_calibrated.csv" \
            python evaluation/eval_rac_bgpt.py \
                --database "$db" --model "$BGPT_IMAGE_MODEL" \
                --n-samples "$IMAGE_N_SAMPLES" --calib-samples "$IMAGE_CALIB_SAMPLES" \
                --m "$M" \
                --image-payload-width "$IMAGE_PAYLOAD_W" \
                --image-payload-height "$IMAGE_PAYLOAD_H" \
                --calibrate --save-index "$OUT_DIR/$dataset/index_calibrated.json" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/rac_calibrated.csv"
    fi

    run_conventional "$dataset" image "$db" "$csv" \
        "$IMAGE_CODECS" "$IMAGE_N_SAMPLES" "$IMAGE_CALIB_SAMPLES"
}

run_audio() {
    local dataset=$1 source_path=$2
    local db="$OUT_DIR/$dataset/db" csv="$OUT_DIR/$dataset/csv"
    mkdir -p "$csv"

    if stage_enabled prepare; then
        run_step "$dataset" prepare "$db/meta.json" \
            python utils/prepare_rac_data_bgpt.py \
                --dataset "$source_path" --modality audio \
                --n-samples "$AUDIO_PREP_SAMPLES" \
                --base-frac "$BASE_FRAC" \
                --audio-chunk-bytes "$AUDIO_CHUNK_BYTES" \
                --retriever bm25 --kgram "$AUDIO_KGRAM" \
                --seed "$SEED" \
                --out "$db" || return 1
    fi
    if [[ ! -f "$db/meta.json" && "$DRY_RUN" != "1" ]]; then
        echo "  [skip] $dataset: no prepared database at $db"
        return 1
    fi

    if stage_enabled neural; then
        run_step "$dataset" bgpt_baseline "$csv/bgpt_baseline.csv" \
            python evaluation/eval_bgpt.py \
                --modality audio \
                --dataset "$db/eval_samples.pkl" \
                --model "$BGPT_AUDIO_MODEL" \
                --n-samples "$AUDIO_N_SAMPLES" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/bgpt_baseline.csv"
    fi

    if stage_enabled rac; then
        local root="$OUT_DIR/$dataset"
        if [[ "$VERIFY" == "1" ]]; then
            run_step "$dataset" rac_verify "$root/logs/rac_verify.done" \
                python evaluation/eval_rac_bgpt.py \
                    --database "$db" --model "$BGPT_AUDIO_MODEL" \
                    --n-samples "$VERIFY_N" --m "$M" \
                    --device "$DEVICE"
        fi
        run_step "$dataset" rac_fixed "$csv/rac_fixed.csv" \
            python evaluation/eval_rac_bgpt.py \
                --database "$db" --model "$BGPT_AUDIO_MODEL" \
                --n-samples "$AUDIO_N_SAMPLES" --calib-samples "$AUDIO_CALIB_SAMPLES" \
                --m "$M" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/rac_fixed.csv"
        run_step "$dataset" rac_calibrated "$csv/rac_calibrated.csv" \
            python evaluation/eval_rac_bgpt.py \
                --database "$db" --model "$BGPT_AUDIO_MODEL" \
                --n-samples "$AUDIO_N_SAMPLES" --calib-samples "$AUDIO_CALIB_SAMPLES" \
                --m "$M" \
                --calibrate --save-index "$OUT_DIR/$dataset/index_calibrated.json" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/rac_calibrated.csv"
    fi

    run_conventional "$dataset" audio "$db" "$csv" \
        "$AUDIO_CODECS" "$AUDIO_N_SAMPLES" "$AUDIO_CALIB_SAMPLES"
}

run_conventional() {
    # Standalone + whole-reference delta codecs, on the same test prefix.
    local dataset=$1 modality=$2 db=$3 csv=$4 codecs=$5 n_samples=$6 calib=$7

    if stage_enabled codecs; then
        run_step "$dataset" codecs "$csv/codecs.csv" \
            python evaluation/eval_baselines.py \
                --database "$db" --modality "$modality" \
                --codecs "$codecs" \
                --n-samples "$n_samples" --calib-samples "$calib" \
                --allow-missing \
                --output "$csv/codecs.csv" \
                --summary-output "$csv/codecs_summary.csv"
    fi

    if stage_enabled codecs && [[ "$WITH_ZSTD_DICT" == "1" ]]; then
        run_step "$dataset" codecs_dict "$csv/codecs_dict.csv" \
            python evaluation/eval_baselines.py \
                --database "$db" --modality "$modality" \
                --codecs zstd-dict \
                --n-samples "$n_samples" --calib-samples "$calib" \
                --dictionary-size "$DICT_SIZE" \
                --save-zstd-dictionary "$OUT_DIR/$dataset/zstd_dict.bin" \
                --output "$csv/codecs_dict.csv" \
                --summary-output "$csv/codecs_dict_summary.csv"
    fi

    if stage_enabled delta; then
        run_step "$dataset" delta "$csv/delta.csv" \
            python evaluation/eval_delta_baselines.py \
                --database "$db" --modality "$modality" \
                --codecs "$DELTA_CODECS" \
                --candidate-policy retrieve --m "$M" \
                --n-samples "$n_samples" --calib-samples "$calib" \
                --keep-going \
                --output "$csv/delta.csv"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "Main experiment"
echo "  datasets: $DATASETS"
echo "  stages:   $STAGES"
echo "  device:   $DEVICE   m=$M   out=$OUT_DIR"
echo

preflight || exit 1
[[ "$DRY_RUN" == "1" ]] || printf 'dataset\tstep\tstatus\tseconds\n' >"$SUMMARY"

for dataset in $DATASETS; do
    modality=$(dataset_modality "$dataset")
    source_path=$(dataset_path "$dataset")
    echo
    echo "== $dataset ($modality) — $source_path"
    case "$modality" in
        text)  run_text  "$dataset" "$source_path" ;;
        image) run_image "$dataset" "$source_path" ;;
        audio) run_audio "$dataset" "$source_path" ;;
        *)     echo "  [FAIL] unknown dataset $dataset"; FAILURES=$((FAILURES + 1)) ;;
    esac
done

if stage_enabled collect && [[ "$DRY_RUN" != "1" ]]; then
    echo
    echo "== Collecting ================================================="
    python scripts/collect_results.py --run-dir "$OUT_DIR" \
        --datasets "$DATASETS" || FAILURES=$((FAILURES + 1))
fi

echo
echo "== Done ======================================================="
echo "  steps run: $STEPS_RUN   skipped: $STEPS_SKIPPED   failed: $FAILURES"
echo "  per-step status: $SUMMARY"
echo "  per-dataset CSVs: $OUT_DIR/<dataset>/csv/"
echo "  comparable table: $OUT_DIR/summary.csv"
echo
echo "  Tabulate from summary.csv, not from the evaluators' own bpb/ratio: the"
echo "  neural image rows divide by padded patch bytes, so a padded run can"
echo "  look better than an identical unpadded one. collect_results.py divides"
echo "  every method by the true sample size and shows the padding share."
[[ "$FAILURES" -eq 0 ]]
