import argparse
import gzip
import io
import json
import re
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm


API_URL = "https://export.arxiv.org/api/query"
SOURCE_URL = "https://export.arxiv.org/e-print/{}"
ATOM = {"a": "http://www.w3.org/2005/Atom"}

TEXT_EXTS = {".tex", ".bib", ".bbl", ".sty", ".cls", ".txt"}


def parse_size(s):
    m = re.fullmatch(r"([\d.]+)\s*(B|KB|MB|GB|KIB|MIB|GIB)?", s.upper())
    if not m:
        raise ValueError(f"Invalid size: {s}")

    scales = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
    }
    return int(float(m.group(1)) * scales[m.group(2) or "B"])


def extract_text(data):
    def extract_tar(raw):
        parts = []

        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    if Path(member.name).suffix.lower() not in TEXT_EXTS:
                        continue

                    f = tar.extractfile(member)
                    if f:
                        text = f.read().decode("utf-8", errors="replace")
                        parts.append(f"% FILE: {member.name}\n{text}")
        except tarfile.TarError:
            return None

        return "\n\n".join(parts) or None

    text = extract_tar(data)
    if text:
        return text

    try:
        raw = gzip.decompress(data)
    except OSError:
        raw = data

    text = extract_tar(raw)
    if text:
        return text

    decoded = raw.decode("utf-8", errors="replace")

    if "\\documentclass" in decoded or "\\begin{document}" in decoded:
        return decoded

    return None


def fetch_page(session, category, start, batch_size):
    response = session.get(
        API_URL,
        params={
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        },
        timeout=60,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    papers = []

    for entry in root.findall("a:entry", ATOM):
        url = entry.findtext("a:id", namespaces=ATOM)
        arxiv_id = re.sub(r"v\d+$", "", url.rsplit("/", 1)[-1])

        categories = [
            x.attrib["term"]
            for x in entry.findall("a:category", ATOM)
        ]

        papers.append((arxiv_id, categories))

    return papers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--delay", type=float, default=3.0)
    args = parser.parse_args()

    target_size = parse_size(args.volume)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "arxiv-source-downloader/1.0"

    written = 0
    papers_saved = 0
    start = 0

    with output_path.open("wb") as output, tqdm(
        total=target_size,
        desc=args.category,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        while written < target_size:
            papers = fetch_page(
                session,
                args.category,
                start,
                args.batch_size,
            )

            if not papers:
                break

            for arxiv_id, categories in papers:
                try:
                    response = session.get(
                        SOURCE_URL.format(arxiv_id),
                        timeout=120,
                    )
                    response.raise_for_status()
                    text = extract_text(response.content)
                except requests.RequestException:
                    text = None

                if text:
                    record = {
                        "id": arxiv_id,
                        "categories": categories,
                        "text": text,
                    }

                    line = (
                        json.dumps(record, ensure_ascii=False) + "\n"
                    ).encode("utf-8")

                    if written + len(line) > target_size:
                        break

                    output.write(line)
                    written += len(line)
                    papers_saved += 1

                    bar.update(len(line))
                    bar.set_postfix(papers=papers_saved)

                time.sleep(args.delay)

            else:
                start += len(papers)
                continue

            break

    print(f"Saved {papers_saved} papers to {output_path}")
    print(f"Output size: {written / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
