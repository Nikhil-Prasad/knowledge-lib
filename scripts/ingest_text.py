#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def _bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help)
    group.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help}")
    parser.set_defaults(**{name: default})


def ingest_raw(api: str, text: str, title: str | None, dedupe: bool, collection_id: str | None = None) -> dict:
    body = {
        "source": {"source_type": "raw_text", "text": text, "title": title},
        "options": {"dedupe": dedupe},
    }
    if collection_id:
        body["collection_id"] = collection_id
    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{api.rstrip('/')}/v1/ingest", json=body)
        r.raise_for_status()
        return r.json()


def ingest_file(api: str, path: str, content_type_hint: str | None, dedupe: bool, collection_id: str | None = None) -> dict:
    p = Path(path).expanduser().resolve()
    source = {
        "source_type": "upload_ref",
        "upload_uri": f"file://{p}",
    }
    if content_type_hint:
        source["content_type_hint"] = content_type_hint
    body = {"source": source, "options": {"dedupe": dedupe}}
    if collection_id:
        body["collection_id"] = collection_id
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{api.rstrip('/')}/v1/ingest", json=body)
        r.raise_for_status()
        return r.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest text into the Knowledge-Lib API")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL (default: %(default)s)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    raw = sub.add_parser("raw", help="Ingest raw text from argument or stdin")
    raw.add_argument("--text", help="Text to ingest; if omitted, reads from stdin")
    raw.add_argument("--title", help="Optional title hint", default=None)
    raw.add_argument("--collection-id", help="Optional collection UUID to associate", default=None)
    _bool_flag(raw, "dedupe", True, "dedupe identical normalized text")

    filep = sub.add_parser("file", help="Ingest a local text file (uses file:// upload_ref)")
    filep.add_argument("path", help="Path to local text/markdown file")
    filep.add_argument("--content-type-hint", default=None, help="Optional content-type hint (e.g., text/plain)")
    filep.add_argument("--collection-id", help="Optional collection UUID to associate", default=None)
    _bool_flag(filep, "dedupe", True, "dedupe identical normalized text")

    args = parser.parse_args()

    if args.cmd == "raw":
        text = args.text
        if text is None:
            text = sys.stdin.read()
        resp = ingest_raw(args.api, text=text, title=args.title, dedupe=args.dedupe, collection_id=args.collection_id)
    elif args.cmd == "file":
        resp = ingest_file(args.api, path=args.path, content_type_hint=args.content_type_hint, dedupe=args.dedupe, collection_id=args.collection_id)
    else:
        parser.error("unknown command")
        return 2

    print(json.dumps(resp, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
