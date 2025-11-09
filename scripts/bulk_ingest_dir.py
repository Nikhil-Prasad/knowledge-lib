#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
import uuid as _uuid

import httpx


def _bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help: str) -> None:
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(f"--{name}", dest=name, action="store_true", help=help)
    grp.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help}")
    parser.set_defaults(**{name: default})


def build_body(file_path: Path, dedupe: bool, collection_id: str | None, content_type_hint: str | None) -> dict:
    source = {
        "source_type": "upload_ref",
        "upload_uri": f"file://{file_path.resolve()}",
    }
    if content_type_hint:
        source["content_type_hint"] = content_type_hint
    body = {"source": source, "options": {"dedupe": dedupe}}
    if collection_id:
        body["collection_id"] = collection_id
    return body


def ingest_many(
    api: str,
    files: List[Path],
    *,
    collection_id: str,
    dedupe: bool,
    content_type_hint: str | None,
) -> List[Tuple[Path, dict]]:
    results: List[Tuple[Path, dict]] = []
    url = f"{api.rstrip('/')}/v1/ingest"
    with httpx.Client(timeout=120.0) as client:
        for fp in files:
            body = build_body(fp, dedupe=dedupe, collection_id=collection_id, content_type_hint=content_type_hint)
            try:
                r = client.post(url, json=body)
                r.raise_for_status()
                results.append((fp, r.json()))
            except httpx.HTTPError as e:
                results.append((fp, {"error": str(e)}))
    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Bulk-ingest all text files under a directory into one collection")
    p.add_argument("dir", help="Directory containing files to ingest")
    p.add_argument("--api", default="http://127.0.0.1:8000", help="API base URL (default: %(default)s)")
    p.add_argument("--glob", default="*.txt", help="Glob pattern relative to dir (default: %(default)s)")
    p.add_argument("--content-type-hint", default="text/plain", help="Content-Type hint (default: %(default)s)")
    p.add_argument("--collection-id", default=None, help="Use this collection UUID (if omitted, a new one is generated)")
    _bool_flag(p, "dedupe", True, "server-side dedupe by normalized text")
    args = p.parse_args()

    base = Path(args.dir).expanduser().resolve()
    if not base.is_dir():
        print(f"ERROR: not a directory: {base}", file=sys.stderr)
        return 2

    files = sorted([fp for fp in base.rglob(args.glob) if fp.is_file()])
    if not files:
        print("No files matched.")
        return 0

    collection_id = args.collection_id or str(_uuid.uuid4())
    print(f"Using collection_id: {collection_id}")
    print(f"Found {len(files)} files. Ingesting with dedupe={'on' if args.dedupe else 'off'} ...")

    results = ingest_many(
        args.api,
        files,
        collection_id=collection_id,
        dedupe=args.dedupe,
        content_type_hint=args.content_type_hint,
    )

    created = 0
    deduped = 0
    failed = 0
    for fp, resp in results:
        if "error" in resp:
            failed += 1
            print(f"FAIL {fp.name}: {resp['error']}")
            continue
        pc = int(resp.get("pages_created", 0))
        sc = int(resp.get("segments_created", 0))
        cid = resp.get("container_id")
        if pc == 0 and sc == 0:
            deduped += 1
            print(f"DEDUP {fp.name} → container {cid}")
        else:
            created += 1
            print(f"OK    {fp.name} → container {cid} (pages={pc}, segments={sc})")

    print()
    print(json.dumps({
        "collection_id": collection_id,
        "total": len(results),
        "created": created,
        "deduped": deduped,
        "failed": failed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

