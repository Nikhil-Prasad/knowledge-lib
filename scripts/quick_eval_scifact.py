#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re
import time

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
N_QUERIES: Optional[int] = None   # how many queries to test; None = all
TOP_K = 20             # top-k per method
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

# Output report directory (Markdown reports are timestamped). If None, skip file creation
REPORT_DIR: Optional[str] = "reports"


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

    # Aggregates for metrics
    total_eval = 0
    methods = [m for m in ["FTS", "ANN", "HYBRID"] if (m != "HYBRID" or INCLUDE_HYBRID or MODE == "hybrid")] 
    hits = {m: 0 for m in methods}
    mrr = {m: 0.0 for m in methods}
    recall_sum = {m: 0.0 for m in methods}
    precision_sum = {m: 0.0 for m in methods}
    ndcg_sum = {m: 0.0 for m in methods}
    map_sum = {m: 0.0 for m in methods}
    lat_sum = {m: 0.0 for m in methods}
    lat_cnt = {m: 0 for m in methods}
    # Candidate-stage recall evaluation (pre-rerank pools)
    CAND_EVAL = True
    CAND_FTS_N = 200
    CAND_ANN_N = 400
    cand_recall_sum = {"FTS": 0.0, "ANN": 0.0}
    cand_eval_count = 0

    per_query_rows = []  # collect lightweight per-query results for the report

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
                t0 = time.time()
                fts_hits = request_search(API_BASE, "/v1/search/fts", body)
                lat_sum["FTS"] += (time.time() - t0)
                lat_cnt["FTS"] += 1
            except Exception as e:
                print(f"  FTS error: {e}")
                fts_hits = []
            for h in fts_hits:
                all_hits.append(("FTS", h))

        if MODE in ("ann", "both"):
            try:
                t0 = time.time()
                ann_hits = request_search(API_BASE, "/v1/search/ann", body)
                lat_sum["ANN"] += (time.time() - t0)
                lat_cnt["ANN"] += 1
            except Exception as e:
                print(f"  ANN error: {e}")
                ann_hits = []
            for h in ann_hits:
                all_hits.append(("ANN", h))

        if INCLUDE_HYBRID or MODE == "hybrid":
            try:
                t0 = time.time()
                hyb_hits = request_search(API_BASE, "/v1/search/hybrid", body)
                lat_sum["HYBRID"] += (time.time() - t0)
                lat_cnt["HYBRID"] += 1
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
            hits_list = by_method[method]
            if not hits_list:
                print(f"  {method}: (no results)")
                continue
            hit_pos: Optional[int] = None
            if exp_cids:
                for i, h in enumerate(hits_list, start=1):
                    if str(h.get("container_id")) in exp_cids:
                        hit_pos = i
                        break
            prefix = f"  {method}:"
            if hit_pos is not None:
                prefix += f" HIT@{hit_pos}"
            print(prefix)
            for rank, h in enumerate(hits_list, start=1):
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
            exp_set = set(exp_cids)
            # Gather a compact per-query row for the report
            row_for_report = {"qid": qid, "query": qtext, "expected": exp_cids}
            for method in methods:
                method_hits = by_method.get(method, [])
                topk = method_hits[:TOP_K]
                # dedupe containers for stats to avoid counting multiple segments per doc
                topk_containers = [str(h.get("container_id")) for h in topk]
                topk_set = set(topk_containers)
                # Hit@k
                found = None
                for i, cid in enumerate(topk_containers, start=1):
                    if cid in exp_set:
                        found = i
                        break
                if found is not None:
                    hits[method] += 1
                    mrr[method] += 1.0 / found
                # Recall@k based on doc-level presence
                recall = 0.0
                if exp_set:
                    present = len(exp_set.intersection(topk_set))
                    recall = present / len(exp_set)
                    recall_sum[method] += recall
                # Precision@k (doc-level, unique containers)
                denom = min(TOP_K, len(topk_set)) if TOP_K else len(topk_set)
                prec = (len(exp_set.intersection(topk_set)) / denom) if denom else 0.0
                precision_sum[method] += prec
                # nDCG@k (binary gains, first occurrence per container)
                dcg = 0.0
                seen = set()
                rel_seen = set()
                rank_pos = 0
                for cid in topk_containers:
                    if cid in seen:
                        continue
                    seen.add(cid)
                    rank_pos += 1
                    if cid in exp_set:
                        rel_seen.add(cid)
                        dcg += 1.0 / (math.log2(rank_pos + 1))
                ideal = min(len(exp_set), TOP_K)
                idcg = sum(1.0 / (math.log2(i + 1)) for i in range(1, ideal + 1)) if ideal > 0 else 0.0
                ndcg = (dcg / idcg) if idcg > 0 else 0.0
                ndcg_sum[method] += ndcg
                # MAP@k (AP@k with doc-level, first occurrence per container)
                ap = 0.0
                seen = set()
                rel_count = 0
                rank_pos = 0
                for cid in topk_containers:
                    if cid in seen:
                        continue
                    seen.add(cid)
                    rank_pos += 1
                    if cid in exp_set:
                        rel_count += 1
                        ap += rel_count / rank_pos
                ap = ap / len(exp_set) if len(exp_set) > 0 else 0.0
                map_sum[method] += ap
                # Save rank summary for report
                row_for_report[f"{method}_rank"] = found
                row_for_report[f"{method}_recall"] = recall
            per_query_rows.append(row_for_report)

        print()

    # Candidate-stage recall: issue broader FTS/ANN queries per item (optional)
    if CAND_EVAL:
        # Reload items to avoid duplicating network calls earlier
        # We recompute candidate recall using the same iteration
        cand_recall_sum = {"FTS": 0.0, "ANN": 0.0}
        cand_eval_count = 0
        for obj in items:
            qid, qtext = extract_query(obj)
            if not qtext:
                continue
            # expected set
            exp_cids: List[str] = []
            if qid and qid in qrels:
                for doc_id in qrels[qid]:
                    cid = title_to_cid.get(doc_id) or title_to_cid.get(_norm_title(doc_id))
                    if cid:
                        exp_cids.append(cid)
            exp_set = set(exp_cids)
            if not exp_set:
                continue
            body_cand_fts = {"query": qtext, "k": CAND_FTS_N}
            body_cand_ann = {"query": qtext, "k": CAND_ANN_N}
            if COLLECTION_ID:
                body_cand_fts["collection_id"] = COLLECTION_ID
                body_cand_ann["collection_id"] = COLLECTION_ID
            try:
                fts_cand_hits = request_search(API_BASE, "/v1/search/fts", body_cand_fts)
                ann_cand_hits = request_search(API_BASE, "/v1/search/ann", body_cand_ann)
            except Exception:
                continue
            fts_set = set(str(h.get("container_id")) for h in fts_cand_hits)
            ann_set = set(str(h.get("container_id")) for h in ann_cand_hits)
            cand_recall_sum["FTS"] += (len(exp_set.intersection(fts_set)) / len(exp_set))
            cand_recall_sum["ANN"] += (len(exp_set.intersection(ann_set)) / len(exp_set))
            cand_eval_count += 1

    if total_eval:
        print("Summary (HIT@k / MRR@k / Recall@k):")
        print(f"  evaluated: {total_eval}")
        for method in methods:
            hit_rate = (hits[method] / total_eval) * 100.0
            mrr_avg = (mrr[method] / total_eval)
            recall_avg = (recall_sum[method] / total_eval)
            print(f"  {method:<6} HIT@{TOP_K}: {hits[method]}/{total_eval} ({hit_rate:.1f}%)  MRR@{TOP_K}: {mrr_avg:.3f}  Recall@{TOP_K}: {recall_avg:.3f}")
        print("\nAdditional metrics:")
        for method in methods:
            p_avg = precision_sum[method] / total_eval
            ndcg_avg = ndcg_sum[method] / total_eval
            map_avg = map_sum[method] / total_eval
            lat_ms = (lat_sum[method] / lat_cnt[method] * 1000.0) if lat_cnt[method] else 0.0
            print(f"  {method:<6} P@{TOP_K}: {p_avg:.3f}  nDCG@{TOP_K}: {ndcg_avg:.3f}  MAP@{TOP_K}: {map_avg:.3f}  |  avg latency: {lat_ms:.1f} ms")
        # Candidate-stage recall
        if cand_eval_count:
            print("\nCandidate-stage recall:")
            print(f"  FTS@{CAND_FTS_N}: {cand_recall_sum['FTS']/cand_eval_count:.3f}")
            print(f"  ANN@{CAND_ANN_N}: {cand_recall_sum['ANN']/cand_eval_count:.3f}")

    # Write Markdown report if configured
    if REPORT_DIR and total_eval:
        ts_file = datetime.now().strftime('%Y%m%d_%H%M%S')
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_dir = Path(REPORT_DIR)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"scifact_eval_{ts_file}.md"
        lines: List[str] = []
        lines.append(f"# SciFact Retrieval Evaluation\n")
        lines.append(f"Generated: {ts}\n\n")
        lines.append(f"- Queries: `{QUERIES_PATH}`\n")
        lines.append(f"- Qrels: `{QRELS_PATH}`\n")
        lines.append(f"- Collection: `{COLLECTION_ID or 'ALL'}`\n")
        lines.append(f"- K: `{TOP_K}`  •  Total evaluated: `{total_eval}`\n\n")
        # Metrics table
        lines.append("## Summary\n")
        lines.append("Method | HIT@k | MRR@k | Recall@k | P@k | nDCG@k | MAP@k | Avg Latency (ms)\n")
        lines.append("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: \n")
        for method in methods:
            hit_rate = (hits[method] / total_eval) * 100.0
            mrr_avg = (mrr[method] / total_eval)
            recall_avg = (recall_sum[method] / total_eval)
            p_avg = precision_sum[method] / total_eval
            ndcg_avg = ndcg_sum[method] / total_eval
            map_avg = map_sum[method] / total_eval
            lat_ms = (lat_sum[method] / lat_cnt[method] * 1000.0) if lat_cnt[method] else 0.0
            lines.append(f"{method} | {hit_rate:.1f}% | {mrr_avg:.3f} | {recall_avg:.3f} | {p_avg:.3f} | {ndcg_avg:.3f} | {map_avg:.3f} | {lat_ms:.1f}\n")
        lines.append("\n")
        # Candidate-stage recall section
        if cand_eval_count:
            lines.append("## Candidate-stage Recall\n")
            lines.append(f"FTS@{CAND_FTS_N}: {cand_recall_sum['FTS']/cand_eval_count:.3f}  ")
            lines.append(f"ANN@{CAND_ANN_N}: {cand_recall_sum['ANN']/cand_eval_count:.3f}\n\n")
        # Sample details (first 10 queries)
        lines.append("## Sample Queries (first 10)\n")
        for row in per_query_rows[:10]:
            lines.append(f"- QID `{row.get('qid')}`: {row.get('query')}\n")
            lines.append(f"  - expected: {row.get('expected')}\n")
            for method in methods:
                r = row.get(f"{method}_rank")
                rec = row.get(f"{method}_recall")
                if r is not None:
                    lines.append(f"  - {method}: rank={r}, recall@{TOP_K}={rec:.2f}\n")
                else:
                    lines.append(f"  - {method}: no hit in top-{TOP_K}, recall@{TOP_K}={rec:.2f}\n")
        report_path.write_text("".join(lines), encoding="utf-8")
        print(f"\nWrote report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
