#!/usr/bin/env python3
import json
import math
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Tuple, Optional

BASE = os.environ.get("BASE", "http://127.0.0.1:8080").rstrip("/")
QUERIES_PATH = os.environ.get("QUERIES_PATH", "queries_train.json")

HTTP_TIMEOUT_SEC = float(os.environ.get("HTTP_TIMEOUT_SEC", "36"))

ENDPOINTS = os.environ.get("ENDPOINTS", "search_title,search_body,search_anchor,search").split(",")

K_AP = int(os.environ.get("K_AP", "10"))      # AP@10 (quality requirement mentions AP@10)
K_P  = int(os.environ.get("K_P", "10"))        # Precision@10
K_F  = int(os.environ.get("K_F", "30"))       # F1@30 (useful if you track it)

MAX_Q = int(os.environ.get("MAX_Q", "999999"))  # cap how many queries to run
SAVE_JSON = os.environ.get("SAVE_JSON", "")     # optional path: save per-query scores as json


def http_get_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "eval_queries/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return json.loads(data.decode("utf-8"))


def load_qrels(path: str) -> Dict[str, List[int]]:
    """
    Expected: { "query": [doc_id1, doc_id2, ...], ... }
    doc_id can be str/int in file; we normalize to int where possible.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise RuntimeError("queries_train.json should be a dict: {query: [relevant_doc_ids...]}")

    qrels: Dict[str, List[int]] = {}
    for q, rel_list in obj.items():
        if not isinstance(rel_list, list):
            continue
        out: List[int] = []
        for x in rel_list:
            try:
                out.append(int(x))
            except Exception:
                # skip anything non-numeric
                continue
        qrels[str(q)] = out
    if not qrels:
        raise RuntimeError("No queries loaded from qrels file.")
    return qrels


def extract_doc_ids(engine_output: Any) -> List[int]:
    """
    Your endpoints usually return: [[doc_id, title], ...]
    We extract doc_id robustly from:
      - list of [doc_id, ...]
      - list of dicts with 'doc_id'
      - list of ints
    """
    doc_ids: List[int] = []

    if not isinstance(engine_output, list):
        return doc_ids

    for item in engine_output:
        doc_id = None
        if isinstance(item, int):
            doc_id = item
        elif isinstance(item, str):
            try:
                doc_id = int(item)
            except Exception:
                doc_id = None
        elif isinstance(item, list) and item:
            try:
                doc_id = int(item[0])
            except Exception:
                doc_id = None
        elif isinstance(item, dict):
            if "doc_id" in item:
                try:
                    doc_id = int(item["doc_id"])
                except Exception:
                    doc_id = None

        if doc_id is not None:
            doc_ids.append(doc_id)

    return doc_ids


def precision_at_k(pred: List[int], rel_set: set, k: int) -> float:
    if k <= 0:
        return 0.0
    topk = pred[:k]
    if not topk:
        return 0.0
    hits = sum(1 for d in topk if d in rel_set)
    return hits / float(k)


def recall_at_k(pred: List[int], rel_set: set, k: int) -> float:
    if not rel_set:
        return 0.0
    topk = pred[:k]
    hits = sum(1 for d in topk if d in rel_set)
    return hits / float(len(rel_set))


def f1(p: float, r: float) -> float:
    if p <= 0.0 or r <= 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def average_precision_at_k(pred: List[int], rel_set: set, k: int) -> float:
    """
    Binary AP@k (standard IR): average of precision at each rank i (<=k) where pred[i] is relevant.
    Normalized by min(|rel|, k) to avoid rewarding queries with huge qrels.
    """
    if k <= 0:
        return 0.0
    denom = min(len(rel_set), k)
    if denom == 0:
        return 0.0

    score = 0.0
    hits = 0
    for i, d in enumerate(pred[:k], start=1):
        if d in rel_set:
            hits += 1
            score += hits / float(i)
    return score / float(denom)


def timed_fetch(endpoint: str, query: str) -> Tuple[Optional[float], Optional[List[int]], Optional[str]]:
    q = urllib.parse.quote(query)
    url = f"{BASE}/{endpoint}?query={q}"
    t0 = time.time()
    try:
        out = http_get_json(url, timeout=HTTP_TIMEOUT_SEC)
        dt = time.time() - t0
        return dt, extract_doc_ids(out), None
    except Exception as e:
        dt = time.time() - t0
        return None, None, f"{type(e).__name__}: {e} (elapsed={dt:.2f}s)"


def mean(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / float(len(xs))


def p90(values: List[float]) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = max(0, math.ceil(0.90 * len(xs)) - 1)
    return xs[k]


def main():
    qrels = load_qrels(QUERIES_PATH)
    queries = list(qrels.keys())[:MAX_Q]

    print(f"Loaded {len(qrels)} queries from {QUERIES_PATH}")
    print(f"BASE={BASE}")
    print(f"ENDPOINTS={ENDPOINTS}")
    print(f"Metrics: AP@{K_AP}, P@{K_P}, F1@{K_F}")

    # Aggregations per endpoint
    agg: Dict[str, Dict[str, List[float]]] = {
        ep: {"ap": [], "p": [], "r": [], "f1": [], "time": []} for ep in ENDPOINTS
    }
    failures: List[Tuple[str, str, str]] = []
    per_query_dump: Dict[str, Any] = {}

    for i, q in enumerate(queries, 1):
        rel = qrels.get(q, [])
        rel_set = set(rel)

        print(f"\n[{i}/{len(queries)}] {q!r}  (|qrels|={len(rel_set)})")

        per_query_dump[q] = {"qrels_len": len(rel_set), "endpoints": {}}

        for ep in ENDPOINTS:
            dt, pred, err = timed_fetch(ep, q)

            if err is not None or pred is None:
                failures.append((ep, q, err or "unknown error"))
                # score 0 if failed / timeout
                ap = p = r = f1v = 0.0
                print(f"  {ep:12s} FAIL  {err}")
            else:
                ap = average_precision_at_k(pred, rel_set, K_AP)
                p = precision_at_k(pred, rel_set, K_P)
                r = recall_at_k(pred, rel_set, K_F)
                f1v = f1(p, r)

                agg[ep]["time"].append(float(dt))
                print(f"  {ep:12s} t={dt:.3f}s  AP@{K_AP}={ap:.4f}  P@{K_P}={p:.4f}  R@{K_F}={r:.4f}  F1@{K_F}={f1v:.4f}")

            agg[ep]["ap"].append(ap)
            agg[ep]["p"].append(p)
            agg[ep]["r"].append(r)
            agg[ep]["f1"].append(f1v)

            per_query_dump[q]["endpoints"][ep] = {
                f"AP@{K_AP}": ap,
                f"P@{K_P}": p,
                f"R@{K_F}": r,
                f"F1@{K_F}": f1v,
                "time_sec": dt
            }

    print("\n================== OVERALL SUMMARY ==================")
    for ep in ENDPOINTS:
        apm = mean(agg[ep]["ap"])
        pm  = mean(agg[ep]["p"])
        rm  = mean(agg[ep]["r"])
        f1m = mean(agg[ep]["f1"])
        tm  = mean(agg[ep]["time"])
        tp90 = p90(agg[ep]["time"])
        tmax = max(agg[ep]["time"]) if agg[ep]["time"] else float("nan")

        print(f"\n[{ep}]")
        print(f"  Mean AP@{K_AP}: {apm:.4f}")
        print(f"  Mean P@{K_P} : {pm:.4f}")
        print(f"  Mean R@{K_F} : {rm:.4f}")
        print(f"  Mean F1@{K_F}: {f1m:.4f}")
        print(f"  Time (sec): avg={tm:.3f}  p90={tp90:.3f}  max={tmax:.3f}   (only successful calls counted)")

    if failures:
        print("\n================== FAILURES (first 25) ==================")
        for ep, q, err in failures[:25]:
            print(f"- ep={ep} query={q!r} err={err}")
        if len(failures) > 25:
            print(f"... and {len(failures)-25} more")

    if SAVE_JSON:
        with open(SAVE_JSON, "w", encoding="utf-8") as f:
            json.dump(per_query_dump, f, ensure_ascii=False, indent=2)
        print(f"\nSaved per-query details to: {SAVE_JSON}")


if __name__ == "__main__":
    main()
