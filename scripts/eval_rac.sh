#!/usr/bin/env bash
#
# Compare the RAW LLM baseline (`none`) against RAC on the *same* test pieces and
# PRINT one consolidated ratio table. `none` is plain LLMCompressor on the exact
# test_pieces.json — byte-for-byte comparable to rac/rac_rerank (same model,
# tokenizer, byte denominator, arithmetic coder) — so this is the honest A/B.
#
# By default it PRINTS only (intermediate CSVs go to a temp dir, auto-removed).
# Pass OUTDIR=<dir> to also keep the CSVs there.
#
# Usage:
#   bash scripts/compare_raw_rac_ratio.sh
#   N_TEST=200 bash scripts/compare_raw_rac_ratio.sh           # quick subset
#   OUTDIR=results/rac_c_1k/compare bash scripts/compare_raw_rac_ratio.sh   # keep CSVs
#
# Override via env: DATA MODEL ENCODER RERANKER M_VALUES DEVICE N_TEST DECODE_CHECK OUTDIR
set -euo pipefail

# --- config ---------------------------------------------------------------
DATA="${DATA:-results/rac_c_1k}"
MODEL="${MODEL:-pretrained/SmolLM2-135M}"
ENCODER="${ENCODER:-$DATA/rac_encoder}"
RERANKER="${RERANKER:-$DATA/rac_reranker}"
M_VALUES="${M_VALUES:-1}"          # eval_rac --m is a single int → loop it
DEVICE="${DEVICE:-cuda}"
N_TEST="${N_TEST:-}"                 # empty = full test set
DECODE_CHECK="${DECODE_CHECK:-1}"   # decode+verify this many pieces (0 = skip)
OUTDIR="${OUTDIR:-}"                 # empty = print only (temp CSVs, auto-removed)
METHODS="none,token_rag,rac,rac_rerank"

cd "$(dirname "$0")/.."              # repo root

# --- best-effort conda env (per project: needs LMCompress) ----------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "LMCompress" ]]; then
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate LMCompress 2>/dev/null || true
fi

# --- sanity: required inputs ----------------------------------------------
[[ -d "$DATA"    ]] || { echo "ERROR: --data dir not found: $DATA"     >&2; exit 1; }
[[ -d "$ENCODER" ]] || { echo "ERROR: encoder dir not found: $ENCODER" >&2; exit 1; }
[[ -f "$DATA/test_pieces.json" ]] \
    || { echo "ERROR: $DATA has no test_pieces.json"                   >&2; exit 1; }
if [[ ! -d "$RERANKER" ]]; then
    echo "WARN: reranker not found at $RERANKER — dropping rac_rerank" >&2
    METHODS="none,token_rag,rac"
fi

# --- workdir: keep iff OUTDIR given, else temp (auto-removed) --------------
if [[ -n "$OUTDIR" ]]; then
    WORKDIR="$OUTDIR"; mkdir -p "$WORKDIR"
else
    WORKDIR="$(mktemp -d)"; trap 'rm -rf "$WORKDIR"' EXIT
fi

echo "data=$DATA  encoder=$ENCODER  reranker=$RERANKER"
echo "methods=$METHODS  m={$M_VALUES}  device=$DEVICE  n_test=${N_TEST:-all}"

run_eval () {                        # $1 = m ; remaining = extra args
    local m="$1"; shift
    python evaluation/eval_rac.py \
        --data "$DATA" --model "$MODEL" \
        --encoder "$ENCODER" --reranker "$RERANKER" \
        --methods "$METHODS" --m "$m" --device "$DEVICE" "$@"
}

# --- 1. decode correctness on a single instance (WITH decompress) ---------
if [[ "$DECODE_CHECK" -gt 0 ]]; then
    echo; echo "### decode check — $DECODE_CHECK instance(s), m=1 (lossless roundtrip) ..."
    run_eval 1 --n-test "$DECODE_CHECK" --output "$WORKDIR/decode_check.csv"
fi

# --- 2. full comparison per m (fast: --no-decompress; bpb/ratio still exact)
NTEST_ARG=()
if [[ -n "$N_TEST" ]]; then NTEST_ARG+=(--n-test "$N_TEST"); fi
for m in $M_VALUES; do
    echo; echo "### ratio comparison — m=$m ..."
    run_eval "$m" --no-decompress ${NTEST_ARG[@]+"${NTEST_ARG[@]}"} \
        --output "$WORKDIR/eval_m${m}.csv"
done

# --- 3. consolidated, corpus-aggregate table ------------------------------
echo; echo "==================== RESULT ===================="
python - "$WORKDIR" $M_VALUES <<'PY'
import csv, os, sys
workdir, ms = sys.argv[1], sys.argv[2:]
ORDER = ["none", "token_rag", "rac", "rac_rerank"]

def load(path):
    g = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            meth = row["sample_id"].split(":", 1)[0]
            o, c, rt = int(row["original_bytes"]), int(row["compressed_bytes"]), int(row["roundtrip_ok"])
            d = g.setdefault(meth, [0, 0, 0, 0, 0])   # orig, comp, n, rt_ok, rt_seen
            d[0] += o; d[1] += c; d[2] += 1
            if rt >= 0:
                d[4] += 1; d[3] += rt
    return g

# decode correctness (the single-instance, WITH-decompress run)
dp = os.path.join(workdir, "decode_check.csv")
if os.path.exists(dp):
    dg = load(dp)
    print("decode correctness (single instance, lossless roundtrip):")
    for meth in ORDER:
        if meth in dg:
            *_, n, ok, seen = dg[meth]
            tag = "PASS" if seen and ok == seen else ("FAIL" if seen else "n/a")
            print(f"  {meth:<12} {tag} ({ok}/{seen})")

# corpus-aggregate ratio/bpb per method, Δ vs raw baseline
for m in ms:
    path = os.path.join(workdir, f"eval_m{m}.csv")
    if not os.path.exists(path):
        continue
    g = load(path)
    n_pieces = next(iter(g.values()))[2]
    none_bpb = (g["none"][1] * 8 / g["none"][0]) if "none" in g else None
    print(f"\nm={m} | {n_pieces} pieces (corpus-aggregate, side-info charged)")
    print(f"{'method':<12}{'ratio':>10}{'bpb':>10}{'vs none(bpb)':>14}")
    for meth in ORDER:
        if meth not in g:
            continue
        o, c, *_ = g[meth]
        ratio, bpb = o / max(c, 1), c * 8 / max(o, 1)
        delta = "—" if (none_bpb is None or meth == "none") else f"{(none_bpb - bpb) / none_bpb * 100:+.2f}%"
        print(f"{meth:<12}{ratio:>10.4f}{bpb:>10.4f}{delta:>14}")
    if none_bpb is not None and "rac_rerank" in g:
        o, c, *_ = g["rac_rerank"]
        rr_bpb = c * 8 / max(o, 1)
        verdict = "HELPS ✓" if rr_bpb < none_bpb else "does NOT help ✗"
        print(f"  -> retrieval {verdict}  (rac_rerank vs raw none)")
PY

[[ -n "$OUTDIR" ]] && echo && echo "CSVs kept in: $OUTDIR/" || true
