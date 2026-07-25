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

PROJ_HOME=$HOME/projects/RAGLMCompress

export PATH="$PROJ_HOME/baseline_coders/flac-1.4.3/_install/bin:$PROJ_HOME/baseline_coders/bsdiff:$PROJ_HOME/baseline_coders/cmix:$PROJ_HOME/baseline_coders/openzl:$PROJ_HOME/baseline_coders/libjxl/build/tools:$PATH"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR=${DATA_DIR:-datasets}
OUT_DIR=${OUT_DIR:-results/main}
LLM_MODEL=${LLM_MODEL:-pretrained/SmolLM2-135M}
BGPT_IMAGE_MODEL=${BGPT_IMAGE_MODEL:-pretrained/bgpt/weights-image.pth}
BGPT_AUDIO_MODEL=${BGPT_AUDIO_MODEL:-pretrained/bgpt/weights-audio.pth}
DEVICE=${DEVICE:-cuda:2,cuda:3}
SEED=${SEED:-42}
M=${M:-4}                       # retrieval candidates the oracle tries per piece
BASE_FRAC=${BASE_FRAC:-0.4}

# Everything below is a *modality* default.  Any of these settings can be
# overridden for a single dataset in the per-dataset table further down, so
# datasets of the same modality need not share corpus sizes or geometry.

# Corpus sizes.  PREP_* is what gets split into base+held-out; N_* is how many
# held-out samples every method is scored on; CALIB_* is the index-calibration
# tail excluded from all test sets.
TEXT_PREP_DOCS=${TEXT_PREP_DOCS:-300}
TEXT_N_DOCS=${TEXT_N_DOCS:-150}
TEXT_CALIB_DOCS=${TEXT_CALIB_DOCS:-30}
IMAGE_PREP_SAMPLES=${IMAGE_PREP_SAMPLES:-1000}
IMAGE_N_SAMPLES=${IMAGE_N_SAMPLES:-500}
IMAGE_CALIB_SAMPLES=${IMAGE_CALIB_SAMPLES:-100}
AUDIO_PREP_SAMPLES=${AUDIO_PREP_SAMPLES:-700}
AUDIO_N_SAMPLES=${AUDIO_N_SAMPLES:-350}
AUDIO_CALIB_SAMPLES=${AUDIO_CALIB_SAMPLES:-50}

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
IMAGE_KGRAM=${IMAGE_KGRAM:-8}

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

# This script is compress-only: every neural stage passes --no-decompress, so no
# GPU decode round-trip runs here (that is what was OOMing on long windows). The
# reported sizes are the honest arithmetic-coded bytes the oracle selects; to
# confirm roundtrip_ok on a new config, run one eval_rac_* by hand on a few
# samples *without* --no-decompress. Note the conventional codec baselines
# (eval_baselines / eval_delta_baselines) still decode on CPU internally to
# byte-verify — that is intrinsic to them, cheap, and never touches the GPU.
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

ALL_DATASETS="arxiv_cl arxiv_ar eurlex_caselaw eurlex_regulation \
eurosat_forest eurosat_highway medmnist_blood medmnist_retina \
ljspeech vctk_p225"
DATASETS=${DATASETS:-$ALL_DATASETS}
ALL_STAGES="prepare neural rac codecs delta collect"
STAGES=${STAGES:-$ALL_STAGES}

# ---------------------------------------------------------------------------
# Per-dataset overrides
#
# Define <dataset>_<SETTING> to override one modality default for one dataset;
# anything left undefined falls back to the modality value above.  These are
# ordinary variables, so the environment can set them too:
#
#     medmnist_retina_N_SAMPLES=120 scripts/run_main_experiment.sh
#
# Settings available per dataset:
#   text   PREP_DOCS N_DOCS CALIB_DOCS CHUNK_TOKENS BASELINE_TOKENS
#   image  PREP_SAMPLES N_SAMPLES CALIB_SAMPLES KGRAM
#          COND_W COND_H PAYLOAD_W PAYLOAD_H BASE_PATCH_W BASE_PATCH_H
#   audio  PREP_SAMPLES N_SAMPLES CALIB_SAMPLES CHUNK_BYTES KGRAM
#   any    BASE_FRAC M CODECS
# ---------------------------------------------------------------------------

# retinamnist's test split holds only ~400 images, so its held-out half is ~200.
# N + CALIB must fit inside it: otherwise eval_bgpt scores N samples while the
# RAC evaluators carve the calibration tail out of a shorter prefix and score
# fewer, and the two rows stop describing the same images.
# medmnist_retina_PREP_SAMPLES=${medmnist_retina_PREP_SAMPLES:-1000}
# medmnist_retina_N_SAMPLES=${medmnist_retina_N_SAMPLES:-500}
# medmnist_retina_CALIB_SAMPLES=${medmnist_retina_CALIB_SAMPLES:-100}

# EUR-Lex documents are far shorter than arXiv TeX sources, so more of them fit
# in a comparable corpus. Tune once you have seen the real document lengths
# (scripts/probe_dataset.py reports them).
eurlex_caselaw_PREP_DOCS=1000
eurlex_caselaw_N_DOCS=500
eurlex_caselaw_CALIB_DOCS=100
eurlex_regulation_PREP_DOCS=1000
eurlex_regulation_N_DOCS=500
eurlex_regulation_CALIB_DOCS=100

# VCTK p225 is one speaker with many short clips; LJSpeech clips are longer.
vctk_p225_PREP_SAMPLES=200
vctk_p225_N_SAMPLES=100
vctk_p225_CALIB_SAMPLES=20

# Resolve one setting for the dataset being run: the <dataset>_<KEY> override if
# it is defined and non-empty, else the modality default named by <FALLBACK>.
setting() {   # setting <dataset> <KEY> <FALLBACK_VAR>
    local override="${1}_${2}"
    if [[ -n "${!override:-}" ]]; then
        printf '%s' "${!override}"
    else
        printf '%s' "${!3}"
    fi
}

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
        eurosat_forest|eurosat_highway|medmnist_blood|medmnist_retina) echo image ;;
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
        eurosat_highway)    echo "$DATA_DIR/eurosat/Highway" ;;
        # The registered loader names the test split explicitly, so the
        # dataset path has to include it (utils/img_utils.py).
        medmnist_blood)     echo "$DATA_DIR/medmnist/bloodmnist_128/train" ;;
        medmnist_retina)    echo "$DATA_DIR/medmnist/retinamnist_128/train" ;;
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

    local prep_docs n_docs calib_docs chunk_tokens baseline_tokens base_frac m codecs
    prep_docs=$(setting "$dataset" PREP_DOCS TEXT_PREP_DOCS)
    n_docs=$(setting "$dataset" N_DOCS TEXT_N_DOCS)
    calib_docs=$(setting "$dataset" CALIB_DOCS TEXT_CALIB_DOCS)
    chunk_tokens=$(setting "$dataset" CHUNK_TOKENS TEXT_CHUNK_TOKENS)
    baseline_tokens=$(setting "$dataset" BASELINE_TOKENS TEXT_BASELINE_TOKENS)
    base_frac=$(setting "$dataset" BASE_FRAC BASE_FRAC)
    m=$(setting "$dataset" M M)
    codecs=$(setting "$dataset" CODECS TEXT_CODECS)
    echo "  config: prep=$prep_docs docs, test=$n_docs, calib=$calib_docs," \
         "condition=$chunk_tokens tok, baseline window=$baseline_tokens tok, m=$m"

    if stage_enabled prepare; then
        run_step "$dataset" prepare "$db/meta.json" \
            python utils/prepare_rac_data_llm.py \
                --dataset "$source_path" \
                --n-docs "$prep_docs" \
                --base-frac "$base_frac" \
                --chunk-size "$chunk_tokens" --chunk-overlap 0 \
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
                --n-docs "$n_docs" \
                --max-tokens "$baseline_tokens" \
                --device "$DEVICE" \
                --no-decompress \
                --output "$csv/llm_baseline.csv"
    fi

    if stage_enabled rac; then
        run_step "$dataset" rac_fixed "$csv/rac_fixed.csv" \
            python evaluation/eval_rac_llm.py \
                --database "$db" --model "$LLM_MODEL" \
                --n-docs "$n_docs" --calib-docs "$calib_docs" --m "$m" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/rac_fixed.csv"
        run_step "$dataset" rac_calibrated "$csv/rac_calibrated.csv" \
            python evaluation/eval_rac_llm.py \
                --database "$db" --model "$LLM_MODEL" \
                --n-docs "$n_docs" --calib-docs "$calib_docs" --m "$m" \
                --calibrate --save-index "$root/index_calibrated.json" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/rac_calibrated.csv"
    fi

    run_conventional "$dataset" text "$db" "$csv" \
        "$codecs" "$n_docs" "$calib_docs" "$m"
}

run_image() {
    local dataset=$1 source_path=$2
    local db="$OUT_DIR/$dataset/db" csv="$OUT_DIR/$dataset/csv"
    mkdir -p "$csv"

    local prep n_samples calib kgram base_frac m codecs
    local cond_w cond_h payload_w payload_h base_w base_h
    prep=$(setting "$dataset" PREP_SAMPLES IMAGE_PREP_SAMPLES)
    n_samples=$(setting "$dataset" N_SAMPLES IMAGE_N_SAMPLES)
    calib=$(setting "$dataset" CALIB_SAMPLES IMAGE_CALIB_SAMPLES)
    kgram=$(setting "$dataset" KGRAM IMAGE_KGRAM)
    base_frac=$(setting "$dataset" BASE_FRAC BASE_FRAC)
    m=$(setting "$dataset" M M)
    codecs=$(setting "$dataset" CODECS IMAGE_CODECS)
    cond_w=$(setting "$dataset" COND_W IMAGE_COND_W)
    cond_h=$(setting "$dataset" COND_H IMAGE_COND_H)
    payload_w=$(setting "$dataset" PAYLOAD_W IMAGE_PAYLOAD_W)
    payload_h=$(setting "$dataset" PAYLOAD_H IMAGE_PAYLOAD_H)
    base_w=$(setting "$dataset" BASE_PATCH_W IMAGE_BASE_PATCH_W)
    base_h=$(setting "$dataset" BASE_PATCH_H IMAGE_BASE_PATCH_H)
    echo "  config: prep=$prep imgs, test=$n_samples, calib=$calib, m=$m," \
         "condition=${cond_w}x${cond_h}, payload=${payload_w}x${payload_h}," \
         "baseline patch=${base_w}x${base_h}"
    # Condition + payload must fill one bGPT image body exactly, or
    # eval_rac_bgpt rejects the rectangle. Catch a bad override before prepare.
    if ! python -c "
import sys
sys.path.insert(0, '.')
from compression.bgpt_compressor import IMAGE_BODY_BYTES
from utils.img_utils import bmp_payload_nbytes
cond = bmp_payload_nbytes($cond_w, $cond_h)
payload = bmp_payload_nbytes($payload_w, $payload_h)
base = bmp_payload_nbytes($base_w, $base_h)
ok = True
if cond + payload != IMAGE_BODY_BYTES:
    print(f'  condition {cond} B + payload {payload} B = {cond+payload} B != '
          f'{IMAGE_BODY_BYTES} B image body'); ok = False
if base != IMAGE_BODY_BYTES:
    print(f'  baseline patch {base} B != {IMAGE_BODY_BYTES} B image body'); ok = False
raise SystemExit(0 if ok else 1)
"; then
        echo "  [FAIL] $dataset: image geometry does not fill the bGPT body"
        FAILURES=$((FAILURES + 1))
        return 1
    fi

    if stage_enabled prepare; then
        run_step "$dataset" prepare "$db/meta.json" \
            python utils/prepare_rac_data_bgpt.py \
                --dataset "$source_path" --modality image \
                --n-samples "$prep" \
                --base-frac "$base_frac" \
                --image-patch-width "$cond_w" \
                --image-patch-height "$cond_h" \
                --retriever bm25 --kgram "$kgram" \
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
                --n-samples "$n_samples" \
                --image-patch-width "$base_w" \
                --image-patch-height "$base_h" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/bgpt_baseline.csv"
    fi

    if stage_enabled rac; then
        local root="$OUT_DIR/$dataset"
        run_step "$dataset" rac_fixed "$csv/rac_fixed.csv" \
            python evaluation/eval_rac_bgpt.py \
                --database "$db" --model "$BGPT_IMAGE_MODEL" \
                --n-samples "$n_samples" --calib-samples "$calib" \
                --m "$m" \
                --image-payload-width "$payload_w" \
                --image-payload-height "$payload_h" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/rac_fixed.csv"
        run_step "$dataset" rac_calibrated "$csv/rac_calibrated.csv" \
            python evaluation/eval_rac_bgpt.py \
                --database "$db" --model "$BGPT_IMAGE_MODEL" \
                --n-samples "$n_samples" --calib-samples "$calib" \
                --m "$m" \
                --image-payload-width "$payload_w" \
                --image-payload-height "$payload_h" \
                --calibrate --save-index "$OUT_DIR/$dataset/index_calibrated.json" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/rac_calibrated.csv"
    fi

    run_conventional "$dataset" image "$db" "$csv" \
        "$codecs" "$n_samples" "$calib" "$m"
}

run_audio() {
    local dataset=$1 source_path=$2
    local db="$OUT_DIR/$dataset/db" csv="$OUT_DIR/$dataset/csv"
    mkdir -p "$csv"

    local prep n_samples calib chunk_bytes kgram base_frac m codecs
    prep=$(setting "$dataset" PREP_SAMPLES AUDIO_PREP_SAMPLES)
    n_samples=$(setting "$dataset" N_SAMPLES AUDIO_N_SAMPLES)
    calib=$(setting "$dataset" CALIB_SAMPLES AUDIO_CALIB_SAMPLES)
    chunk_bytes=$(setting "$dataset" CHUNK_BYTES AUDIO_CHUNK_BYTES)
    kgram=$(setting "$dataset" KGRAM AUDIO_KGRAM)
    base_frac=$(setting "$dataset" BASE_FRAC BASE_FRAC)
    m=$(setting "$dataset" M M)
    codecs=$(setting "$dataset" CODECS AUDIO_CODECS)
    echo "  config: prep=$prep clips, test=$n_samples, calib=$calib," \
         "condition=$chunk_bytes B, kgram=$kgram, m=$m"

    if stage_enabled prepare; then
        run_step "$dataset" prepare "$db/meta.json" \
            python utils/prepare_rac_data_bgpt.py \
                --dataset "$source_path" --modality audio \
                --n-samples "$prep" \
                --base-frac "$base_frac" \
                --audio-chunk-bytes "$chunk_bytes" \
                --retriever bm25 --kgram "$kgram" \
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
                --n-samples "$n_samples" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/bgpt_baseline.csv"
    fi

    if stage_enabled rac; then
        local root="$OUT_DIR/$dataset"
        run_step "$dataset" rac_fixed "$csv/rac_fixed.csv" \
            python evaluation/eval_rac_bgpt.py \
                --database "$db" --model "$BGPT_AUDIO_MODEL" \
                --n-samples "$n_samples" --calib-samples "$calib" \
                --m "$m" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/rac_fixed.csv"
        run_step "$dataset" rac_calibrated "$csv/rac_calibrated.csv" \
            python evaluation/eval_rac_bgpt.py \
                --database "$db" --model "$BGPT_AUDIO_MODEL" \
                --n-samples "$n_samples" --calib-samples "$calib" \
                --m "$m" \
                --calibrate --save-index "$OUT_DIR/$dataset/index_calibrated.json" \
                --device "$DEVICE" --no-decompress \
                --output "$csv/rac_calibrated.csv"
    fi

    run_conventional "$dataset" audio "$db" "$csv" \
        "$codecs" "$n_samples" "$calib" "$m"
}

run_conventional() {
    # Standalone + whole-reference delta codecs, on the same test prefix.
    local dataset=$1 modality=$2 db=$3 csv=$4 codecs=$5 n_samples=$6 calib=$7 m=$8

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
                --candidate-policy retrieve --m "$m" \
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
