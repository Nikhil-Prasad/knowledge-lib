#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

import httpx
from sqlalchemy import create_engine, text as sql_text


# --------- Configuration (set variables here) ---------

# Path to BEIR SciFact queries.jsonl
QUERIES_PATH = \
    "/Users/nikhilprasad/crown/knowledge-lib/evals/datasets/beir/eval/scifact/queries.jsonl"

# API and DB endpoints
API_BASE = "http://127.0.0.1:8000"
DB_URL = "postgresql+psycopg2://kl:klpass@localhost:5432/knowledge"

# Eval params
N_QUERIES = 10         # how many queries to test
TOP_K = 10             # top-k per method
MODE = "both"         # one of: "fts", "ann", "both", "hybrid"
INCLUDE_HYBRID = True  # also call the hybrid endpoint and print results
PRINT_SNIPPET = True   # whether to print snippet text

# Optional: scope to a single collection (UUID string) or set to None
COLLECTION_ID: Optional[str] = None

# Optional: name of JSON key in queries holding expected title (to compute HIT@k). Set to None to disable.
EXPECTED_FIELD: Optional[str] = None

# Optional: BEIR qrels path (TSV with columns: qid\t0\tdoc_id\trel). If set, we match expected doc_ids via title→container map
QRELS_PATH: Optional[str] = \
    "/Users/nikhilprasad/crown/knowledge-lib/evals/datasets/beir/eval/scifact/qrels/test.tsv"


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(obj)
            if limit is not None and len(items) >= limit:
                break
    return items


def extract_query(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    # Returns (qid, query_text)
    qid_keys = ("qid", "id", "_id", "query_id")
    text_keys = ("query", "text", "question", "claim", "title")
    qid = None
    for k in qid_keys:
        if k in obj:
            qid = str(obj[k])
            break
    qtext = None
    for k in text_keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            qtext = v.strip()
            break
    return qid, qtext


def request_search(api: str, path: str, body: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"{api.rstrip('/')}{path}"
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
        return data.get("results", [])


def lookup_titles(db_url: str, container_ids: Iterable[str]) -> Dict[str, Optional[str]]:
    ids = list({str(cid) for cid in container_ids})
    if not ids:
        return {}
    engine = create_engine(db_url)
    out: Dict[str, Optional[str]] = {}
    q = sql_text("SELECT title FROM containers WHERE container_id = CAST(:cid AS uuid)")
    with engine.begin() as conn:
        for cid in ids:
            row = conn.execute(q, {"cid": cid}).first()
            out[cid] = row[0] if row else None
    return out


def build_title_to_container_map(db_url: str, collection_id: Optional[str] = None) -> Dict[str, str]:
    """Build a map from title text (text_segments.object_type='title') -> container_id.

    If collection_id is provided, restrict to containers in that collection.
    Adds both exact and normalized keys to be more forgiving during lookup.
    """
    engine = create_engine(db_url)
    if collection_id:
        sql = sql_text(
            """
            SELECT ts.text AS title, ts.container_id::text AS cid
            FROM text_segments ts
            JOIN containers_collections cc ON cc.container_id = ts.container_id
            WHERE ts.object_type = 'title' AND cc.collection_id = CAST(:cid AS uuid)
            """
        )
        params = {"cid": collection_id}
    else:
        sql = sql_text(
            """
            SELECT text AS title, container_id::text AS cid
            FROM text_segments
            WHERE object_type = 'title'
            """
        )
        params = {}

    out: Dict[str, str] = {}
    with engine.begin() as conn:
        for row in conn.execute(sql, params).mappings():
            t = row["title"]
            cid = row["cid"]
            if isinstance(t, str) and t:
                out[t] = cid
                out[_norm_title(t)] = cid
    return out


def _norm_title(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_qrels(path: Optional[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    if not path:
        return mapping
    p = Path(path).expanduser()
    if not p.is_file():
        return mapping
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            qid, _zero, doc_id, rel = parts[0], parts[1], parts[2], parts[3]
            try:
                rel_i = int(rel)
            except ValueError:
                rel_i = 0
            if rel_i <= 0:
                continue
            mapping.setdefault(qid, []).append(doc_id)
    return mapping


def main() -> int:
    qpath = Path(QUERIES_PATH).expanduser()
    if not qpath.is_file():
        print(f"ERROR: queries file not found: {qpath}", file=sys.stderr)
        return 2

    items = load_jsonl(qpath, limit=N_QUERIES)
    if not items:
        print("No queries loaded.")
        return 0

    # DB URL
    db_url = DB_URL

    title_to_cid: Dict[str, str] = {}
    if db_url:
        title_to_cid = build_title_to_container_map(db_url, COLLECTION_ID)

    qrels = load_qrels(QRELS_PATH)

    # Aggregates for HIT@k summary
    total_eval = 0
    hits_fts = 0
    hits_ann = 0
    hits_hyb = 0

    for idx, obj in enumerate(items, start=1):
        qid, qtext = extract_query(obj)
        if not qtext:
            continue
        exp_title = None
        exp_cids: List[str] = []
        exp_from_field = None
        if EXPECTED_FIELD:
            v = obj.get(EXPECTED_FIELD)
            if isinstance(v, str) and v.strip():
                exp_from_field = v.strip()
                cid_field = title_to_cid.get(exp_from_field) or title_to_cid.get(_norm_title(exp_from_field))
                if cid_field:
                    exp_cids.append(cid_field)

        # Also check qrels mapping
        if qid and qid in qrels:
            for doc_id in qrels[qid]:
                cid = title_to_cid.get(doc_id) or title_to_cid.get(_norm_title(doc_id))
                if cid:
                    exp_cids.append(cid)

        # Deduplicate expected container ids
        exp_cids = list(dict.fromkeys(exp_cids))

        head = f"Q{idx} [{qid or '-'}]: {qtext}"
        if exp_from_field:
            head += f"  | expected_title='{exp_from_field}'"
        if exp_cids:
            head += f"  expected_cids={exp_cids}"
        print(head)

        cid = COLLECTION_ID
        body = {"query": qtext, "k": TOP_K}
        if cid:
            body["collection_id"] = cid

        all_hits: List[Tuple[str, Dict[str, Any]]] = []

        if MODE in ("fts", "both"):
            try:
                fts_hits = request_search(API_BASE, "/v1/search/fts", body)
            except Exception as e:
                print(f"  FTS error: {e}")
                fts_hits = []
            for h in fts_hits:
                all_hits.append(("FTS", h))

        if MODE in ("ann", "both"):
            try:
                ann_hits = request_search(API_BASE, "/v1/search/ann", body)
            except Exception as e:
                print(f"  ANN error: {e}")
                ann_hits = []
            for h in ann_hits:
                all_hits.append(("ANN", h))

        if INCLUDE_HYBRID or MODE == "hybrid":
            try:
                hyb_hits = request_search(API_BASE, "/v1/search/hybrid", body)
            except Exception as e:
                print(f"  HYBRID error: {e}")
                hyb_hits = []
            for h in hyb_hits:
                all_hits.append(("HYBRID", h))

        # Lookup container titles
        title_map: Dict[str, Optional[str]] = {}
        if db_url:
            title_map = lookup_titles(db_url, (h[1]["container_id"] for h in all_hits))

        # Pretty print
        by_method: Dict[str, List[Dict[str, Any]]] = {"FTS": [], "ANN": [], "HYBRID": []}
        for method, hit in all_hits:
            by_method[method].append(hit)

        for method in ("FTS", "ANN", "HYBRID"):
            if MODE not in (method.lower(), "both"):
                # Still print HYBRID if INCLUDE_HYBRID is True
                if not (method == "HYBRID" and INCLUDE_HYBRID):
                    continue
            hits = by_method[method]
            if not hits:
                print(f"  {method}: (no results)")
                continue
            hit_pos: Optional[int] = None
            if exp_cids:
                for i, h in enumerate(hits, start=1):
                    if str(h.get("container_id")) in exp_cids:
                        hit_pos = i
                        break
            prefix = f"  {method}:"
            if hit_pos is not None:
                prefix += f" HIT@{hit_pos}"
            print(prefix)
            for rank, h in enumerate(hits, start=1):
                cid = h["container_id"]
                title = title_map.get(str(cid)) if title_map else None
                score = h.get("score")
                seg = h.get("segment_id")
                line = f"    {rank:>2}. score={score:.4f}  cid={cid}  seg={seg}"
                if title:
                    line += f"  title={title}"
                print(line)
                if PRINT_SNIPPET and h.get("snippet"):
                    print(f"        {h['snippet']}")

        # Update aggregates if we had any expected cids for this query
        if exp_cids:
            total_eval += 1
            if by_method["FTS"]:
                if any(str(h.get("container_id")) in exp_cids for h in by_method["FTS"][:TOP_K]):
                    hits_fts += 1
            if by_method["ANN"]:
                if any(str(h.get("container_id")) in exp_cids for h in by_method["ANN"][:TOP_K]):
                    hits_ann += 1
            if by_method["HYBRID"]:
                if any(str(h.get("container_id")) in exp_cids for h in by_method["HYBRID"][:TOP_K]):
                    hits_hyb += 1

        print()

    if total_eval:
        print("Summary (HIT@k):")
        print(f"  evaluated: {total_eval}")
        print(f"  FTS:    {hits_fts}/{total_eval}  ({(hits_fts/total_eval)*100:.1f}%)")
        print(f"  ANN:    {hits_ann}/{total_eval}  ({(hits_ann/total_eval)*100:.1f}%)")
        print(f"  HYBRID: {hits_hyb}/{total_eval}  ({(hits_hyb/total_eval)*100:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
