#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
import uuid
from pathlib import Path
from typing import Dict, Optional
import zipfile
import shutil

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader

# Stable namespace for UUIDv5
NAMESPACE = uuid.UUID("6d3e1a10-7e7c-4a2a-9f7d-6e4a4e1c7d11")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()


def safe_filename(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def uuid5_for_source(source_uri: str) -> str:
    return str(uuid.uuid5(NAMESPACE, source_uri))


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def ensure_dataset(dataset: str, raw_base: Path, url_override: Optional[str]) -> Path:
    # Official BEIR hosting under thakur/ path; allow override
    url = url_override or f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    ds_dir = raw_base / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    print(f"[beir] downloading: {url}")
    path_returned = beir_util.download_and_unzip(url, str(raw_base))

    # If ds_dir/corpus.jsonl missing, try local unzip into ds_dir
    corpus = ds_dir / "corpus.jsonl"
    if not corpus.exists():
        zip_path = raw_base / f"{dataset}.zip"
        if zip_path.exists():
            print(f"[beir] local unzip: {zip_path} -> {ds_dir}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(ds_dir)

    # Flatten raw/<dataset>/<dataset>/* if present
    nested = ds_dir / dataset / "corpus.jsonl"
    if nested.exists():
        for p in (ds_dir / dataset).iterdir():
            target = ds_dir / p.name
            shutil.move(str(p), str(target))
        shutil.rmtree(ds_dir / dataset, ignore_errors=True)

    if not (ds_dir / "corpus.jsonl").exists():
        raise FileNotFoundError(f"Expected {ds_dir}/corpus.jsonl; got path_returned={path_returned}")
    return ds_dir


def prepare_dataset(root: Path, dataset: str, split: str = "test", max_docs: int | None = None, dataset_url: str | None = None) -> None:
    base = root / "evals" / "datasets" / "beir"
    raw_base = base / "raw"
    proc_base = base / "processed" / dataset
    eval_base = base / "eval" / dataset

    proc_docs = proc_base / "docs"
    proc_docs.mkdir(parents=True, exist_ok=True)
    eval_base.mkdir(parents=True, exist_ok=True)

    ds_dir = ensure_dataset(dataset, raw_base, url_override=dataset_url)
    dl = GenericDataLoader(str(ds_dir))
    corpus, queries, qrels = dl.load(split=split)

    if max_docs is not None:
        corpus = dict(list(corpus.items())[: max_docs])

    manifest_path = proc_base / "manifest.jsonl"
    written = 0
    skipped = 0

    with manifest_path.open("w", encoding="utf-8") as man:
        for beir_id, obj in corpus.items():
            title = (obj.get("title") or "").strip()
            body = (obj.get("text") or "")
            body = normalize_text(body)

            parts = [title] if title else []
            parts.append(body)
            full_text = normalize_text("\n\n".join([p for p in parts if p is not None]))

            if len(full_text) < 40:
                skipped += 1
                continue

            source_uri = f"beir:{dataset}:{beir_id}"
            doc_uuid = uuid5_for_source(source_uri)
            sha_norm = sha256_hex(full_text)

            fname = safe_filename(str(beir_id)) or sha_norm[:16]
            out_path = proc_docs / f"{fname}.txt"
            write_atomic(out_path, full_text + "\n")

            manifest = {
                "dataset": dataset,
                "split": split,
                "doc_uuid": doc_uuid,
                "beir_id": beir_id,
                "source_uri": source_uri,
                "path": str(out_path),
                "title": title or None,
                "sha256_normalized": sha_norm,
                "length_chars": len(full_text),
            }
            man.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            written += 1

    # Write eval artifacts (BEIR-style) for convenience
    (eval_base).mkdir(parents=True, exist_ok=True)
    (eval_base / "qrels").mkdir(parents=True, exist_ok=True)
    # queries.jsonl
    with (eval_base / "queries.jsonl").open("w", encoding="utf-8") as qout:
        for qid, qtext in queries.items():
            qout.write(json.dumps({"_id": qid, "text": qtext}, ensure_ascii=False) + "\n")
    # qrels/test.tsv (stick to common naming)
    with (eval_base / "qrels" / "test.tsv").open("w", encoding="utf-8") as qr:
        for qid, rels in qrels.items():
            for did, rel in rels.items():
                qr.write(f"{qid}\t0\t{did}\t{rel}\n")

    print(f"[ok] {dataset} split={split} docs_written={written} skipped={skipped}")
    print(f"[out] docs dir: {proc_docs}")
    print(f"[out] manifest: {manifest_path}")
    print(f"[out] queries : {eval_base / 'queries.jsonl'}")
    print(f"[out] qrels   : {eval_base / 'qrels' / 'test.tsv'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare BEIR dataset as per-doc text files + manifest (auto-download)")
    ap.add_argument("--dataset", default="scifact", help="BEIR dataset name (default: scifact)")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"], help="Split for queries/qrels (default: test)")
    ap.add_argument("--max-docs", type=int, default=None, help="limit for quick smoke")
    ap.add_argument("--dataset-url", default=None, help="Override download URL for the dataset .zip")
    args = ap.parse_args()

    root = Path.cwd()
    prepare_dataset(root, dataset=args.dataset, split=args.split, max_docs=args.max_docs, dataset_url=args.dataset_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
