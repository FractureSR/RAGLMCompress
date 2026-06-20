#!/usr/bin/env python3

import os

# MUST be set before any huggingface import so the mirror is used.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm


REPO = "codeparrot/github-code"
NUM_SHARDS = 1126

# The script-derived `language` column does not exist in the raw parquet,
# so we recompute it from the file path.
_PARQUET_COLUMNS = ["content", "repo_name", "path", "license", "size"]

_LANG_TO_EXTENSION = {
    "Assembly": [".asm"],
    "Batchfile": [".bat", ".cmd"],
    "C": [".c", ".h"],
    "C#": [".cs"],
    "C++": [".cpp", ".hpp", ".c++", ".h++", ".cc", ".hh", ".C", ".H"],
    "CMake": [".cmake"],
    "CSS": [".css"],
    "Dockerfile": [".dockerfile", "Dockerfile"],
    "FORTRAN": [".f90", ".f", ".f03", ".f08", ".f77", ".f95", ".for", ".fpp"],
    "GO": [".go"],
    "Haskell": [".hs"],
    "HTML": [".html"],
    "Java": [".java"],
    "JavaScript": [".js"],
    "Julia": [".jl"],
    "Lua": [".lua"],
    "Makefile": ["Makefile"],
    "Markdown": [".md", ".markdown"],
    "PHP": [".php", ".php3", ".php4", ".php5", ".phps", ".phpt"],
    "Perl": [".pl", ".pm", ".pod", ".perl"],
    "PowerShell": [".ps1", ".psd1", ".psm1"],
    "Python": [".py"],
    "Ruby": [".rb"],
    "Rust": [".rs"],
    "SQL": [".sql"],
    "Scala": [".scala"],
    "Shell": [".sh", ".bash", ".command", ".zsh"],
    "TypeScript": [".ts", ".tsx"],
    "TeX": [".tex"],
    "Visual Basic": [".vb"],
}
_EXT_TO_LANG = {
    ext: lang for lang, exts in _LANG_TO_EXTENSION.items() for ext in exts
}


def lang_from_name(name: str):
    for ext, lang in _EXT_TO_LANG.items():
        if name.endswith(ext):
            return lang
    return None


def shard_path(i: int) -> str:
    return f"data/train-{i:05d}-of-{NUM_SHARDS:05d}.parquet"


class LangSink:
    """Holds the open file, byte budget, counters, and progress bar for one language."""

    def __init__(self, language, out_dir, target_bytes, position):
        self.language = language
        self.target_bytes = target_bytes
        self.path = out_dir / f"{language}.jsonl"
        self.f = self.path.open("w", encoding="utf-8")
        self.total_bytes = 0
        self.num_files = 0
        self.done = False
        self.bar = tqdm(
            total=target_bytes,
            desc=f"{language:<12}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            position=position,
            leave=True,
        )

    def write(self, code, repo, path, lic, size):
        n = len(code.encode("utf-8"))
        record = {
            "repo_name": repo,
            "path": path,
            "language": self.language,
            "license": lic,
            "size": size,
            "code_bytes": n,
            "code": code,
        }
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.total_bytes += n
        self.num_files += 1
        self.bar.update(min(n, self.target_bytes - self.bar.n))
        self.bar.set_postfix(files=self.num_files, mib=f"{self.total_bytes / 1024 / 1024:.2f}")
        if self.total_bytes >= self.target_bytes:
            self.done = True

    def close(self):
        self.f.close()
        self.bar.close()

    def summary(self):
        return {
            "language": self.language,
            "output": str(self.path),
            "num_files": self.num_files,
            "code_bytes": self.total_bytes,
            "code_mib": self.total_bytes / 1024 / 1024,
        }


def extract_languages(languages, out_dir, target_mib, shard_dir, keep_shards=False):
    for language in languages:
        if language not in _LANG_TO_EXTENSION:
            raise ValueError(
                f"Unknown language {language!r}. Choose from: "
                f"{', '.join(sorted(_LANG_TO_EXTENSION))}"
            )

    target_bytes = int(target_mib * 1024 * 1024)
    sinks = {
        lang: LangSink(lang, out_dir, target_bytes, position=i)
        for i, lang in enumerate(languages)
    }

    try:
        for i in range(NUM_SHARDS):
            if all(s.done for s in sinks.values()):
                break

            # Single resolve/ fetch per shard. No repo-tree listing, so the
            # mirror works and nothing falls back to huggingface.co.
            local = hf_hub_download(
                REPO,
                shard_path(i),
                repo_type="dataset",
                local_dir=str(shard_dir),
            )

            try:
                pf = pq.ParquetFile(local)
                for batch in pf.iter_batches(batch_size=10_000, columns=_PARQUET_COLUMNS):
                    d = batch.to_pydict()
                    for code, repo, path, lic, size in zip(
                        d["content"], d["repo_name"], d["path"], d["license"], d["size"]
                    ):
                        lang = lang_from_name(path)
                        sink = sinks.get(lang)
                        if sink is None or sink.done:
                            continue
                        sink.write(code, repo, path, lic, size)

                    if all(s.done for s in sinks.values()):
                        break
            finally:
                if not keep_shards:
                    try:
                        os.remove(local)
                    except OSError:
                        pass
    finally:
        for s in sinks.values():
            s.close()

    return {lang: s.summary() for lang, s in sinks.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-mib", type=float, default=100)
    parser.add_argument("--languages", nargs="+", default=["C", "Python"])
    parser.add_argument(
        "--shard-dir",
        default=None,
        help="Where to download shards temporarily (default: <out-dir>/_shards).",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep downloaded parquet shards instead of deleting them.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dir = Path(args.shard_dir) if args.shard_dir else out_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {args.languages} to {args.target_mib} MiB each")
    summary = extract_languages(
        args.languages, out_dir, args.target_mib, shard_dir, keep_shards=args.keep_shards
    )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()