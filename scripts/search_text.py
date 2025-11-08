#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import httpx


def search(api: str, query: str, k: int) -> dict:
    body = {"query": query, "k": k}
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{api.rstrip('/')}/v1/search", json=body)
        r.raise_for_status()
        return r.json()


def main() -> int:
    p = argparse.ArgumentParser(description="Search text segments via the Knowledge-Lib API")
    p.add_argument("--api", default="http://localhost:8000", help="API base URL (default: %(default)s)")
    p.add_argument("query", help="Search query string")
    p.add_argument("-k", type=int, default=10, help="Number of results (default: %(default)s)")
    args = p.parse_args()

    resp = search(args.api, args.query, args.k)
    print(json.dumps(resp, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

