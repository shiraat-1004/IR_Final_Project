import os
import re
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from flask import Flask, request, jsonify

from inverted_index_gcp import InvertedIndex, MultiFileReader, TUPLE_SIZE


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

BUCKET = os.environ.get("IR_BUCKET_NAME", "").strip()
PORT = int(os.environ.get("PORT", "8080"))

BODY_DIR = os.environ.get("IR_BODY_DIR", "postings_gcp")
TITLE_DIR = os.environ.get("IR_TITLE_DIR", "postings_gcp_title")
ANCHOR_DIR = os.environ.get("IR_ANCHOR_DIR", "postings_gcp_anchor")

BODY_INDEX_NAME = os.environ.get("IR_BODY_INDEX_NAME", "index")
TITLE_INDEX_NAME = os.environ.get("IR_TITLE_INDEX_NAME", "title_index")
ANCHOR_INDEX_NAME = os.environ.get("IR_ANCHOR_INDEX_NAME", "anchor")


META_NAME = os.environ.get("IR_META_NAME", "id_title_pr_pv_dict.pkl")
ID_TITLE_NAME = os.environ.get("IR_ID_TITLE_NAME", "id_title.pkl")

CACHE_DIR = Path(os.environ.get("IR_CACHE_DIR", "/tmp/ir_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()


RE_WORD = re.compile(r"[\#\@\w](['\-]?\w){2,24}", re.UNICODE)

STOPWORDS = {
    "the", "and", "is", "in", "it", "of", "to", "a", "an", "on", "for", "by", "with",
    "as", "at", "from", "this", "that", "these", "those", "be", "are", "was", "were",
    "or", "not", "but", "into", "about", "over", "after", "before", "between", "during",
}
def _gcs_bucket():
    from google.cloud import storage
    if not BUCKET:
        raise RuntimeError("IR_BUCKET_NAME is empty")
    return storage.Client().bucket(BUCKET)

def _cache_path(rel_path: str) -> Path:
    # Keep a filesystem-safe cache key for each GCS object path
    safe = rel_path.replace("/", "__")
    return CACHE_DIR / safe

def load_pickle_from_bucket(rel_path: str):
    """
    Load a pickle object from GCS with local disk caching under /tmp (or IR_CACHE_DIR).
    This avoids repeated downloads on subsequent requests.
    """
    local = _cache_path(rel_path)
    if local.exists() and local.stat().st_size > 0:
        with open(local, "rb") as f:
            return pickle.load(f)

    bkt = _gcs_bucket()
    blob = bkt.blob(rel_path)
    if not blob.exists():
        raise FileNotFoundError(f"Missing in bucket: gs://{BUCKET}/{rel_path}")
    blob.download_to_filename(str(local))
    with open(local, "rb") as f:
        return pickle.load(f)

def try_load_pickle(paths: List[str]) -> Optional[Any]:
    """
    Try to load the first existing pickle among a list of candidate paths.
    Returns None if all candidates fail.
    """
    for p in paths:
        try:
            return load_pickle_from_bucket(p)
        except Exception:
            continue
    return None

TITLE_DF = load_pickle_from_bucket(f"{TITLE_DIR}/title_df.pickle")
TITLE_LOCS = load_pickle_from_bucket(f"{TITLE_DIR}/title_posting_locs.pickle")


def read_posting_list_title(term: str) -> List[Tuple[int, int]]:
    """
    Read title posting list for a term using title_df.pickle + title_posting_locs.pickle.
    Returns list of (doc_id, tf).
    """
    if not TITLE_DF or not TITLE_LOCS:
        return []
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
        doc_id = int.from_bytes(b[start:start+4], "big")
        tf = int.from_bytes(b[start+4:start+6], "big")
        pl.append((doc_id, tf))
    return pl


def tokenize(text: str) -> List[str]:
    """
    Tokenize a free-text query using the course regex and a stopword filter.
    No stemming is applied.
    """
    if not text:
        return []
    tokens = [m.group().lower() for m in RE_WORD.finditer(text.lower())]
    return [t for t in tokens if t not in STOPWORDS]


def try_load_index_anyname(dir_name: str, preferred_name: str) -> Optional[InvertedIndex]:
    """
    Loads an InvertedIndex pickle from <dir>/<name>.pkl, trying several common names.
    """
    candidates = []
    if preferred_name:
        candidates.append(f"{preferred_name}.pkl")
    candidates += ["index.pkl", "anchor.pkl", "title_index.pkl", "body_index.pkl", "inverted_index.pkl"]

    paths = [str(Path(dir_name) / c) for c in candidates]
    obj = try_load_pickle(paths)
    if isinstance(obj, InvertedIndex):
        return obj
    return None

def try_load_meta() -> Optional[dict]:
    """
    Load the metadata mapping doc_id -> title / (title, pr, pv) from GCS.
    Supports either the combined dict or a simpler id->title dict.
    """
    meta = try_load_pickle([META_NAME, ID_TITLE_NAME])
    if isinstance(meta, dict):
        return meta
    return None

META = try_load_meta()

def get_title(doc_id: int) -> str:
    """
    Extract a document title from META, supporting multiple storage formats.
    """
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
    """
    Extract a PageRank value from META if available, otherwise return 0.0.
    """
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
    """
    Extract a pageview value from META if available, otherwise return 0.
    """
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
TITLE_INDEX = try_load_index_anyname(TITLE_DIR, TITLE_INDEX_NAME)
ANCHOR_INDEX = try_load_index_anyname(ANCHOR_DIR, ANCHOR_INDEX_NAME)

_body_reader = None
_title_reader = None
_anchor_reader = None

def _get_reader(which: str) -> MultiFileReader:
    """
    Lazy-create and reuse a MultiFileReader per index type.
    Reuse is important for performance (keeps connections and avoids re-init costs).
    """
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
    raise ValueError("bad reader")

def _normalize_locs(locs: List[Tuple[str, int]], base_dir: str) -> List[Tuple[str, int]]:
    """
    Normalize posting locations so file names are relative to base_dir.
    This protects against stored locs that already include the directory prefix.
    """
    if not locs:
        return locs
    prefix = base_dir.rstrip("/") + "/"
    out = []
    for fname, off in locs:
        if isinstance(fname, str) and fname.startswith(prefix):
            out.append((fname[len(prefix):], off))
        else:
            out.append((fname, off))
    return out

def read_posting_list(index: Optional[InvertedIndex], term: str, which_reader: str, base_dir: str) -> List[Tuple[int, int]]:
    """
    Read and decode a posting list for a term using MultiFileReader and posting_locs.
    Returns a list of (doc_id, tf) pairs.
    """
    if index is None:
        return []
    df = index.df.get(term, 0)
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
        doc_id = int.from_bytes(b[start:start+4], "big")
        tf = int.from_bytes(b[start+4:start+6], "big")
        pl.append((doc_id, tf))
    return pl


def rank_title_or_anchor(index: Optional[InvertedIndex], which_reader: str, base_dir: str, query_terms: List[str]) -> List[int]:
    """
    Rank docs for title/anchor by number of distinct query terms matched.
    For title, uses TITLE_DF + TITLE_LOCS if InvertedIndex is not available.
    """
    doc2score = defaultdict(int)

    for t in set(query_terms):
        if which_reader == "title" and index is None:
            pl = read_posting_list_title(t)
        else:
            pl = read_posting_list(index, t, which_reader, base_dir)

        if not pl:
            continue
        for doc_id, _tf in pl:
            doc2score[doc_id] += 1

    ranked = sorted(doc2score.items(), key=lambda x: (-x[1], x[0]))
    return [doc_id for doc_id, _ in ranked]

def rank_body_tfidf_cosine(query_terms: List[str], topk: int = 100) -> List[int]:
    """
    Compute a simple TF-IDF cosine-style score over body postings.
    Uses dot product between query TF-IDF and document TF-IDF weights.
    """
    if BODY_INDEX is None:
        return []

    # Prefer META size as a proxy for corpus size; fall back to a fixed constant if unavailable
    N = len(META) if isinstance(META, dict) and len(META) > 0 else 6_000_000

    qcnt = Counter(query_terms)
    q_weights: Dict[str, float] = {}

    for t, tfq in qcnt.items():
        df = BODY_INDEX.df.get(t, 0)
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
        df = BODY_INDEX.df.get(t, 1)
        idf = math.log((N + 1) / (df + 1))
        for doc_id, tf in pl:
            wdt = (1.0 + math.log(tf)) * idf
            doc2score[doc_id] += wdt * wq

    ranked = sorted(doc2score.items(), key=lambda x: (-x[1], x[0]))[:topk]
    return [doc_id for doc_id, _ in ranked]

def blend_search(query_terms: List[str]) -> List[int]:
    """
    Blend ranking signals from body/title/anchor using reciprocal-rank style contributions.
    Weights can be tuned via environment variables.
    """
    w_anchor = float(os.environ.get("IR_W_ANCHOR", "0.1"))
    w_body = float(os.environ.get("IR_W_BODY", "0.85"))
    w_title = float(os.environ.get("IR_W_TITLE", "0.05"))

    body_docs = rank_body_tfidf_cosine(query_terms, topk=100)
    title_docs = rank_title_or_anchor(TITLE_INDEX, "title", TITLE_DIR, query_terms) if TITLE_INDEX else []
    anchor_docs = rank_title_or_anchor(ANCHOR_INDEX, "anchor", ANCHOR_DIR, query_terms) if ANCHOR_INDEX else []

    scores = defaultdict(float)

    for r, d in enumerate(anchor_docs[:1000]):
        scores[d] += w_anchor * (1.0 / (r + 1.0))
    for r, d in enumerate(body_docs):
        scores[d] += w_body * (1.0 / (r + 1.0))
    for r, d in enumerate(title_docs[:1000]):
        scores[d] += w_title * (1.0 / (r + 1.0))

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:100]
    return [doc_id for doc_id, _ in ranked]


@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''    
    query = request.args.get("query", "")
    if not query:
        return jsonify([])
    q_terms = tokenize(query)
    if not q_terms:
        return jsonify([])

    doc_ids = blend_search(q_terms)
    res = [(int(doc_id), get_title(int(doc_id))) for doc_id in doc_ids]
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''    
    query = request.args.get("query", "")
    if not query:
        return jsonify([])
    q_terms = tokenize(query)
    if not q_terms:
        return jsonify([])

    doc_ids = rank_body_tfidf_cosine(q_terms, topk=100)
    res = [(int(doc_id), get_title(int(doc_id))) for doc_id in doc_ids]
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''    
    query = request.args.get("query", "")
    if not query:
        return jsonify([])
    q_terms = tokenize(query)
    if not q_terms:
        return jsonify([])

    doc_ids = rank_title_or_anchor(TITLE_INDEX, "title", TITLE_DIR, q_terms)
    res = [(int(doc_id), get_title(int(doc_id))) for doc_id in doc_ids]
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''    
    query = request.args.get("query", "")
    if not query:
        return jsonify([])
    q_terms = tokenize(query)
    if not q_terms:
        return jsonify([])

    doc_ids = rank_title_or_anchor(ANCHOR_INDEX, "anchor", ANCHOR_DIR, q_terms)
    res = [(int(doc_id), get_title(int(doc_id))) for doc_id in doc_ids]
    return jsonify(res)

@app.route("/get_pagerank", methods=["POST"])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''    
    wiki_ids = request.get_json(silent=True) or []
    if not wiki_ids:
        return jsonify([])
    return jsonify([float(get_pagerank_value(int(i))) for i in wiki_ids])

@app.route("/get_pageview", methods=["POST"])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
    had in August 2021.

    Test this by issuing a POST request to a URL like:
      http://YOUR_SERVER_DOMAIN/get_pageview
    with a json payload of the list of article ids. In python do:
      import requests
      requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
    As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    wiki_ids = request.get_json(silent=True) or []
    if not wiki_ids:
        return jsonify([])
    return jsonify([int(get_pageview_value(int(i))) for i in wiki_ids])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
