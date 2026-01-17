# search_frontend.py
# ------------------------------------------------------------
# IR Final Project Frontend (GCP-friendly)
# Endpoints:
#   /health
#   /search_body    -> top 100 TFIDF + cosine (BODY)
#   /search_title   -> ALL docs with >=1 query term in TITLE, ranked by DISTINCT matched query terms
#   /search_anchor  -> ALL docs with >=1 query term in ANCHOR, ranked by DISTINCT matched query terms
#   /search         -> best blend
#   /get_pagerank   -> POST [doc_id,...] -> [pr,...]
#   /get_pageview   -> POST [doc_id,...] -> [pv,...]
#
# Notes:
# - NO caching of query->result.
# - Local caching of GCS objects (pickles/bin shards) IS allowed.
# - Robust to weird prefixes like "gs://<bucket>/postings_gcp_title/..." in posting_locs.
# ------------------------------------------------------------

import os
import re
import math
import pickle
import time
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from flask import Flask, request, jsonify

from inverted_index_gcp import InvertedIndex, MultiFileReader, TUPLE_SIZE


# Flask
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False


# Config
def _env_first(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return default


BUCKET = _env_first("IR_BUCKET_NAME", "IR_BUCKET", "IR_BUCKETNAME", default="").strip()
PORT = int(os.environ.get("PORT", "8080"))

BODY_DIR = os.environ.get("IR_BODY_DIR", "postings_gcp").strip()
TITLE_DIR = os.environ.get("IR_TITLE_DIR", "postings_gcp_title").strip()
ANCHOR_DIR = os.environ.get("IR_ANCHOR_DIR", "postings_gcp_anchor").strip()

BODY_INDEX_NAME = os.environ.get("IR_BODY_INDEX_NAME", "index").strip()
TITLE_INDEX_NAME = os.environ.get("IR_TITLE_INDEX_NAME", "title_index").strip()
ANCHOR_INDEX_NAME = os.environ.get("IR_ANCHOR_INDEX_NAME", "anchor").strip()

META_NAME = os.environ.get("IR_META_NAME", "id_title_pr_pv_dict.pkl").strip()
ID_TITLE_NAME = os.environ.get("IR_ID_TITLE_NAME", "id_title.pkl").strip()

# These are the *actual* title artifacts you built / want:
TITLE_DF_NAME = os.environ.get("IR_TITLE_DF_NAME", "title_df.pickle").strip()
TITLE_LOCS_NAME = os.environ.get("IR_TITLE_LOCS_NAME", "title_posting_locs.pickle").strip()

CACHE_DIR = Path(os.environ.get("IR_CACHE_DIR", "/tmp/ir_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

RE_WORD = re.compile(r"[\#\@\w](['\-]?\w){2,24}", re.UNICODE)
ENGLISH_STOPWORDS = set(stopwords.words('english'))
CORPUS_STOPWORDS = {
    "category", "categories",
    "also", "external", "links",
    "references", "see", "thumb",
    "page", "pages",
    "article", "articles",
    "history", "overview",
    "list", "lists",
    "name", "names",
    "used", "use",
    "one", "two", "three",
    "first", "second",
    "may", "many",
    "however", "often",
}
STOPWORDS = ENGLISH_STOPWORDS | CORPUS_STOPWORDS

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = [m.group().lower() for m in RE_WORD.finditer(text.lower())]
    return [t for t in tokens if t not in STOPWORDS]


# GCS helpers + cache w/ generation check
def _gcs_bucket():
    from google.cloud import storage
    if not BUCKET:
        raise RuntimeError("Bucket is empty. Set IR_BUCKET_NAME (or IR_BUCKET).")
    return storage.Client().bucket(BUCKET)


def _cache_paths(rel_path: str) -> Tuple[Path, Path]:
    # filesystem-safe key
    safe = rel_path.replace("/", "__")
    return (CACHE_DIR / safe, CACHE_DIR / (safe + ".meta"))


def load_pickle_from_bucket(rel_path: str):
    """
    Load a pickle from GCS with local caching.
    Cache is invalidated when blob.generation changes.
    """
    local, meta = _cache_paths(rel_path)

    bkt = _gcs_bucket()
    blob = bkt.blob(rel_path)
    if not blob.exists():
        raise FileNotFoundError(f"Missing in bucket: gs://{BUCKET}/{rel_path}")

    blob.reload()  # ensure generation/size known
    gen = str(blob.generation)

    if local.exists() and meta.exists():
        try:
            cached_gen = meta.read_text().strip()
            if cached_gen == gen and local.stat().st_size > 0:
                with open(local, "rb") as f:
                    return pickle.load(f)
        except Exception:
            pass  # fallthrough to re-download

    # download fresh
    blob.download_to_filename(str(local))
    meta.write_text(gen)
    with open(local, "rb") as f:
        return pickle.load(f)


def try_load_pickle(paths: List[str]) -> Optional[Any]:
    for p in paths:
        try:
            return load_pickle_from_bucket(p)
        except Exception:
            continue
    return None


# Index loading
def try_load_index_anyname(dir_name: str, preferred_name: str) -> Optional[InvertedIndex]:
    candidates = []
    if preferred_name:
        candidates.append(f"{preferred_name}.pkl")
    candidates += ["index.pkl", "anchor.pkl", "title_index.pkl", "body_index.pkl", "inverted_index.pkl"]

    paths = [str(Path(dir_name) / c) for c in candidates]
    obj = try_load_pickle(paths)
    return obj if isinstance(obj, InvertedIndex) else None


def try_load_meta() -> Optional[dict]:
    """
    Load doc_id -> title/(title,pr,pv).
    Supports being stored at root OR under CODE/ or code/.
    """
    candidates = [
        META_NAME,
        ID_TITLE_NAME,
        f"CODE/{META_NAME}",
        f"CODE/{ID_TITLE_NAME}",
        f"code/{META_NAME}",
        f"code/{ID_TITLE_NAME}",
    ]
    meta = try_load_pickle(candidates)
    return meta if isinstance(meta, dict) else None


META = try_load_meta()


def get_title(doc_id: int) -> str:
    if not META:
        return ""
    v = META.get(doc_id)
    if v is None:
        return ""
    if isinstance(v, tuple) and len(v) >= 1:
        return v[0] if isinstance(v[0], str) else ""
    if isinstance(v, dict):
        return str(v.get("title", ""))
    if isinstance(v, str):
        return v
    return ""


def get_pagerank_value(doc_id: int) -> float:
    if not META:
        return 0.0
    v = META.get(doc_id)
    if v is None:
        return 0.0
    if isinstance(v, tuple) and len(v) >= 2:
        try:
            return float(v[1])
        except Exception:
            return 0.0
    if isinstance(v, dict):
        try:
            return float(v.get("pagerank", 0.0))
        except Exception:
            return 0.0
    return 0.0


def get_pageview_value(doc_id: int) -> int:
    if not META:
        return 0
    v = META.get(doc_id)
    if v is None:
        return 0
    if isinstance(v, tuple) and len(v) >= 3:
        try:
            return int(v[2])
        except Exception:
            return 0
    if isinstance(v, dict):
        try:
            return int(v.get("pageviews", 0))
        except Exception:
            return 0
    return 0


BODY_INDEX = try_load_index_anyname(BODY_DIR, BODY_INDEX_NAME)
ANCHOR_INDEX = try_load_index_anyname(ANCHOR_DIR, ANCHOR_INDEX_NAME)

# TITLE is NOT loaded as InvertedIndex here; we use df + posting_locs pickles you built:
TITLE_LOCS = load_pickle_from_bucket(f"{TITLE_DIR}/{TITLE_LOCS_NAME}")
TITLE_DF = load_pickle_from_bucket(f"{TITLE_DIR}/{TITLE_DF_NAME}")

if not isinstance(TITLE_LOCS, dict):
    raise RuntimeError("Bad TITLE_LOCS: expected dict from title_posting_locs.pickle")
if not isinstance(TITLE_DF, dict):
    raise RuntimeError("Bad TITLE_DF: expected dict term->df from title_df.pickle")


# Readers (lazy)
_body_reader = None
_title_reader = None
_anchor_reader = None


def _get_reader(which: str) -> MultiFileReader:
    global _body_reader, _title_reader, _anchor_reader
    if which == "body":
        if _body_reader is None:
            _body_reader = MultiFileReader(BODY_DIR, bucket_name=BUCKET)
        return _body_reader
    if which == "title":
        if _title_reader is None:
            _title_reader = MultiFileReader(TITLE_DIR, bucket_name=BUCKET)
        return _title_reader
    if which == "anchor":
        if _anchor_reader is None:
            _anchor_reader = MultiFileReader(ANCHOR_DIR, bucket_name=BUCKET)
        return _anchor_reader
    raise ValueError("bad reader type")


def _strip_gs_prefix(fname: str) -> str:
    """
    Accepts:
      - "title_000.bin"
      - "postings_gcp_title/title_000.bin"
      - "gs://bucket/postings_gcp_title/title_000.bin"
      - "gs:/bucket/postings_gcp_title/title_000.bin" (weird)
    Returns file name relative to base_dir.
    """
    if not isinstance(fname, str):
        return fname
    s = fname

    # normalize weird gs:/ to gs://
    if s.startswith("gs:/") and not s.startswith("gs://"):
        s = "gs://" + s[len("gs:/"):]

    if s.startswith("gs://"):
        # remove "gs://bucket/"
        parts = s.split("/", 3)
        if len(parts) >= 4:
            s = parts[3]  # path inside bucket

    return s.lstrip("/")


def _normalize_locs(locs: List[Tuple[str, int]], base_dir: str) -> List[Tuple[str, int]]:
    if not locs:
        return locs
    base = base_dir.strip("/")

    out = []
    for fname, off in locs:
        f = _strip_gs_prefix(fname)

        # repeatedly strip base_dir prefix if present (handles double prefixes)
        prefix = base + "/"
        while isinstance(f, str) and f.startswith(prefix):
            f = f[len(prefix):]

        out.append((f, int(off)))
    return out


# Posting list readers
def read_posting_list(index: Optional[InvertedIndex], term: str, which_reader: str, base_dir: str) -> List[Tuple[int, int]]:
    if index is None:
        return []
    df = int(index.df.get(term, 0))
    locs = index.posting_locs.get(term)
    if df <= 0 or not locs:
        return []

    locs2 = _normalize_locs(locs, base_dir)
    n_bytes = df * TUPLE_SIZE
    reader = _get_reader(which_reader)

    try:
        b = reader.read(locs2, n_bytes)
    except Exception:
        return []

    pl = []
    for i in range(df):
        start = i * TUPLE_SIZE
        doc_id = int.from_bytes(b[start:start + 4], "big")
        tf = int.from_bytes(b[start + 4:start + 6], "big")
        pl.append((doc_id, tf))
    return pl


def read_posting_list_title(term: str) -> List[Tuple[int, int]]:
    df = int(TITLE_DF.get(term, 0))
    locs = TITLE_LOCS.get(term)
    if df <= 0 or not locs:
        return []

    locs2 = _normalize_locs(locs, TITLE_DIR)
    n_bytes = df * TUPLE_SIZE
    reader = _get_reader("title")

    try:
        b = reader.read(locs2, n_bytes)
    except Exception:
        return []

    pl = []
    for i in range(df):
        start = i * TUPLE_SIZE
        doc_id = int.from_bytes(b[start:start + 4], "big")
        tf = int.from_bytes(b[start + 4:start + 6], "big")
        pl.append((doc_id, tf))
    return pl


# Ranking
def rank_title_or_anchor(which: str, query_terms: List[str]) -> List[int]:
    """
    Rank by number of DISTINCT matched query terms (descending).
    """
    doc2score = defaultdict(int)
    qset = set(query_terms)

    if which == "title":
        for t in qset:
            pl = read_posting_list_title(t)
            for doc_id, _tf in pl:
                doc2score[doc_id] += 1
    elif which == "anchor":
        for t in qset:
            pl = read_posting_list(ANCHOR_INDEX, t, "anchor", ANCHOR_DIR)
            for doc_id, _tf in pl:
                doc2score[doc_id] += 1
    else:
        raise ValueError("which must be title/anchor")

    ranked = sorted(doc2score.items(), key=lambda x: (-x[1], x[0]))
    return [doc_id for doc_id, _ in ranked]


def rank_body_tfidf_cosine(query_terms: List[str], topk: int = 100) -> List[int]:
    if BODY_INDEX is None:
        return []

    N = len(META) if isinstance(META, dict) and len(META) > 0 else 6_000_000

    qcnt = Counter(query_terms)
    q_weights: Dict[str, float] = {}

    for t, tfq in qcnt.items():
        df = int(BODY_INDEX.df.get(t, 0))
        if df <= 0:
            continue
        idf = math.log((N + 1) / (df + 1))
        q_weights[t] = (1.0 + math.log(tfq)) * idf

    if not q_weights:
        return []

    doc2score = defaultdict(float)

    for t, wq in q_weights.items():
        pl = read_posting_list(BODY_INDEX, t, "body", BODY_DIR)
        if not pl:
            continue
        df = int(BODY_INDEX.df.get(t, 1))
        idf = math.log((N + 1) / (df + 1))
        for doc_id, tf in pl:
            wdt = (1.0 + math.log(tf)) * idf
            doc2score[doc_id] += wdt * wq

    ranked = sorted(doc2score.items(), key=lambda x: (-x[1], x[0]))[:topk]
    return [doc_id for doc_id, _ in ranked]


def blend_search(query_terms: List[str]) -> List[int]:
    w_anchor = float(os.environ.get("IR_W_ANCHOR", "0.1"))
    w_body = float(os.environ.get("IR_W_BODY", "0.85"))
    w_title = float(os.environ.get("IR_W_TITLE", "0.05"))

    body_docs = rank_body_tfidf_cosine(query_terms, topk=100)
    title_docs = rank_title_or_anchor("title", query_terms)
    anchor_docs = rank_title_or_anchor("anchor", query_terms) if ANCHOR_INDEX else []

    scores = defaultdict(float)

    for r, d in enumerate(anchor_docs[:1000]):
        scores[d] += w_anchor * (1.0 / (r + 1.0))
    for r, d in enumerate(body_docs):
        scores[d] += w_body * (1.0 / (r + 1.0))
    for r, d in enumerate(title_docs[:1000]):
        scores[d] += w_title * (1.0 / (r + 1.0))

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:100]
    return [doc_id for doc_id, _ in ranked]


# Routes
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/search")
def search():
    query = request.args.get("query", "")
    if not query:
        return jsonify([])
    q_terms = tokenize(query)
    if not q_terms:
        return jsonify([])

    doc_ids = blend_search(q_terms)
    return jsonify([(int(d), get_title(int(d))) for d in doc_ids])


@app.route("/search_body")
def search_body():
    query = request.args.get("query", "")
    if not query:
        return jsonify([])
    q_terms = tokenize(query)
    if not q_terms:
        return jsonify([])

    doc_ids = rank_body_tfidf_cosine(q_terms, topk=100)
    return jsonify([(int(d), get_title(int(d))) for d in doc_ids])


@app.route("/search_title")
def search_title():
    query = request.args.get("query", "")
    if not query:
        return jsonify([])
    q_terms = tokenize(query)
    if not q_terms:
        return jsonify([])

    doc_ids = rank_title_or_anchor("title", q_terms)
    return jsonify([(int(d), get_title(int(d))) for d in doc_ids])


@app.route("/search_anchor")
def search_anchor():
    query = request.args.get("query", "")
    if not query:
        return jsonify([])
    q_terms = tokenize(query)
    if not q_terms:
        return jsonify([])

    doc_ids = rank_title_or_anchor("anchor", q_terms)
    return jsonify([(int(d), get_title(int(d))) for d in doc_ids])


@app.route("/get_pagerank", methods=["POST"])
def get_pagerank():
    wiki_ids = request.get_json(silent=True) or []
    if not wiki_ids:
        return jsonify([])
    return jsonify([float(get_pagerank_value(int(i))) for i in wiki_ids])


@app.route("/get_pageview", methods=["POST"])
def get_pageview():
    wiki_ids = request.get_json(silent=True) or []
    if not wiki_ids:
        return jsonify([])
    return jsonify([int(get_pageview_value(int(i))) for i in wiki_ids])


# Main
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
