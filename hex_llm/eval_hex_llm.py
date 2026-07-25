"""Stage-0 de-risk: an LLM as a universal byte compressor for IMAGE and AUDIO.

NO retrieval, NO RAC. Feeds raw bytes (image RGB8 / audio PCM_U8) to a causal LM
one byte per token (the "byte route", needs a 256-byte vocab — Qwen3) or two hex
chars per byte (the "hex route", any tokenizer), masked to the alphabet, and asks
whether it compresses and how the absolute bpb compares to the conventional
codecs on the SAME samples:
    image : gzip, PNG, JPEG XL (cjxl)
    audio : gzip, FLAC

    python hex_llm/eval_hex_llm.py --modality image --dataset datasets/eurosat/Forest \
        --model Qwen/Qwen3-0.6B --n-samples 20 --max-payload-bytes 12288 --device cuda:0
    python hex_llm/eval_hex_llm.py --modality audio --dataset datasets/ljspeech_wav \
        --model Qwen/Qwen3-0.6B --n-samples 10 --device cuda:0

TEXT is intentionally NOT the target here — for text use the project's native-BPE
pipeline (evaluation/eval_llm.py), which is more efficient than a byte stream.
--modality text still works for a byte-route sanity check but is not the plan.

The byte route at Qwen3's 40960 context holds ~40k bytes per window, so a whole
image fits in one window (no cold-start split) — set --max-payload-bytes to the
image byte size (e.g. 12288 for 64x64 RGB). Audio clips still window. The
reference codecs compress each WHOLE sample; bpb = compressed_bits/original_bytes
for all, shared denominator. cjxl/djxl/flac are auto-discovered under
--codec-path (default ./baseline_coders) and silently skipped if absent.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import zlib
from typing import Any, Callable, Dict, List, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from hex_llm.hex_llm_compressor import build_compressor

# (bytes, metadata) per sample; metadata feeds the image/audio codec adapters.
Sample = Tuple[bytes, Dict[str, Any]]
# label -> callable(data, metadata) -> compressed byte length
RefCodec = Tuple[str, Callable[[bytes, Dict[str, Any]], int]]


def _load_samples(modality: str, dataset: str, n: int) -> List[Sample]:
    """Canonical bytes + metadata per sample: UTF-8 text, RGB8 image, PCM_U8 audio."""
    if modality == "text":
        from utils.text_utils import load_text_documents
        return [(doc.encode("utf-8"), {}) for doc in load_text_documents(dataset, num_documents=n)]
    if modality == "image":
        from utils.img_utils import load_image_files
        from utils.baseline_data import canonical_image
        return [canonical_image(p) for p in load_image_files(dataset, n)]
    if modality == "audio":
        from utils.audio_utils import load_audio_samples
        from utils.baseline_data import canonical_audio
        return [canonical_audio(w) for w in load_audio_samples(dataset, n)]
    raise ValueError(f"unknown modality {modality!r}")


def _windows(data: bytes, size: int) -> List[bytes]:
    return [data[i:i + size] for i in range(0, len(data), size)]


def _augment_path_for_codecs(root: str) -> None:
    """Prepend any dir under *root* that holds cjxl/djxl/flac to PATH.

    Mirrors what scripts/run_main_experiment.sh does, so the reference codecs are
    found without the caller having to set PATH by hand.
    """
    if not os.path.isdir(root):
        return
    found: Dict[str, str] = {}
    dirs = set()
    for name in ("cjxl", "djxl", "flac"):
        for hit in glob.glob(os.path.join(root, "**", name), recursive=True):
            if os.path.isfile(hit) and os.access(hit, os.X_OK):
                dirs.add(os.path.dirname(hit))
                found[name] = hit
                break
    if dirs:
        os.environ["PATH"] = os.pathsep.join(sorted(dirs)) + os.pathsep + os.environ.get("PATH", "")
        print(f"  codec binaries under {root}: {', '.join(sorted(found))}")


def _reference_codecs(modality: str, tmp_dir: str | None) -> List[RefCodec]:
    """gzip + the modality's conventional lossless codecs; missing ones are skipped."""
    refs: List[RefCodec] = [("gzip", lambda d, m: len(zlib.compress(d, 9)))]
    names = {"text": [], "image": ["png", "jpegxl"], "audio": ["flac"]}[modality]
    if not names:
        return refs
    from compression.baselines import create_codec
    for name in names:
        try:
            codec = create_codec(name, temp_dir=tmp_dir)
        except Exception as exc:  # missing binary / build lacking a feature
            print(f"  reference codec {name}: unavailable "
                  f"({type(exc).__name__}: {str(exc).splitlines()[0][:80]}) — skipped")
            continue
        refs.append((name, lambda d, m, c=codec: len(c.encode(d, m).artifact)))
    return refs


def _load_model(model_path: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
    return model, tok


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-0 byte-LLM compressor vs codecs (no RAC)")
    p.add_argument("--modality", required=True, choices=("text", "image", "audio"))
    p.add_argument("--dataset", required=True, help="path a registered loader resolves")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B",
                   help="causal LM (AutoModelForCausalLM); byte route needs a 256-byte vocab")
    p.add_argument("--route", choices=("byte", "hex"), default="byte",
                   help="byte: 1 token/byte (needs 256-byte vocab); hex: 2 tokens/byte (any tokenizer)")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--max-payload-bytes", type=int, default=8192,
                   help="bytes per window; tokens = route_factor * this (+1) must be <= ctx. "
                        "Set to the whole-image byte size to code an image in one window.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verify-bytes", type=int, default=128,
                   help="round-trip check this many bytes of the first sample (0=skip)")
    p.add_argument("--codec-path", default=os.path.join(_REPO_ROOT, "baseline_coders"),
                   help="tree searched for cjxl/djxl/flac (added to PATH)")
    p.add_argument("--tmp-dir", default=None, help="temp dir for external codecs")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    device = torch.device(args.device)
    _augment_path_for_codecs(args.codec_path)
    model, tok = _load_model(os.path.normpath(args.model) if os.path.isdir(args.model)
                             else args.model, device)
    comp = build_compressor(model, tok, route=args.route, device=device)

    samples = _load_samples(args.modality, args.dataset, args.n_samples)
    if not samples:
        raise SystemExit("no samples loaded")
    spb = comp.symbols_per_byte
    ctx = getattr(model.config, "max_position_embeddings", None)
    if ctx is not None and spb * args.max_payload_bytes + 1 > ctx:
        raise SystemExit(
            f"--max-payload-bytes {args.max_payload_bytes} -> "
            f"{spb * args.max_payload_bytes + 1} tokens > model context {ctx}; lower it")

    total_orig = sum(len(data) for data, _ in samples)
    print(f"byte-LLM Stage-0 | route={args.route} | {args.modality} | "
          f"{len(samples)} samples ({total_orig:,} B) | "
          f"window {args.max_payload_bytes} B ({spb * args.max_payload_bytes} tok) | "
          f"model {args.model} | {device}")

    if args.verify_bytes > 0:
        probe = samples[0][0][: args.verify_bytes]
        print(f"  round-trip check on {len(probe)} B "
              f"({2 * len(probe)} decode steps, O(n^2)) ...", flush=True)
        if not comp.roundtrip_ok(probe):
            raise SystemExit("round-trip failed — the hex coder is not lossless here")
        print("  round-trip: OK")

    import tqdm

    # ── byte-LLM: one bar tick per window (= one forward), live bpb ────────────
    windows = [w for data, _ in samples for w in _windows(data, args.max_payload_bytes) if w]
    llm_comp = 0
    orig_so_far = 0
    t0 = time.monotonic()
    bar = tqdm.tqdm(windows, desc=f"{args.route}-LLM {args.modality}", unit="win")
    for window in bar:
        compressed, n = comp.compress(window)
        llm_comp += len(compressed)
        orig_so_far += n
        # Running bpb over bytes processed SO FAR, so it converges to the true
        # value instead of ramping up from ~0 (numerator fills a fixed total).
        bar.set_postfix(bpb=f"{llm_comp * 8 / max(orig_so_far, 1):.4f}")
    bar.close()
    hex_elapsed = time.monotonic() - t0

    # ── reference codecs: whole sample each; disable a codec if it errors ──────
    refs = _reference_codecs(args.modality, args.tmp_dir)
    ref_comp = {label: 0 for label, _ in refs}
    alive = {label: True for label, _ in refs}
    for data, meta in tqdm.tqdm(samples, desc="codecs", unit="samp"):
        for label, fn in refs:
            if not alive[label]:
                continue
            try:
                ref_comp[label] += fn(data, meta)
            except Exception as exc:
                alive[label] = False
                print(f"\n  {label} disabled: {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:100]}")

    # ── report ────────────────────────────────────────────────────────────────
    def _row(label: str, comp_bytes: int) -> str:
        return (f"  {label:<10} bpb={comp_bytes * 8 / max(total_orig, 1):.4f}  "
                f"ratio={total_orig / max(comp_bytes, 1):.3f}x")

    print(f"\n== {args.modality} ({len(samples)} samples, {total_orig:,} B) ==")
    print(_row(f"{args.route}-LLM", llm_comp) + "   [windowed]")
    for label, _ in refs:
        if alive[label]:
            print(_row(label, ref_comp[label]) + "   [whole-sample]")
    print(f"\n  {args.route}-LLM: {hex_elapsed:.1f}s ({hex_elapsed / len(samples):.2f}s/sample)")
    print("  Note: the LLM is windowed (context-limited); codecs see the whole sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
