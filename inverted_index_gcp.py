import os
import sys
import pickle
import itertools
from collections import Counter, defaultdict
from pathlib import Path
from contextlib import closing
from google.cloud import storage


PROJECT_ID = os.environ.get("PROJECT_ID", "").strip() or None


def get_bucket(bucket_name: str):
    client = storage.Client(project=PROJECT_ID) if PROJECT_ID else storage.Client()
    return client.bucket(bucket_name)


def _open(path: str, mode: str, bucket=None):
    """
    Open local file or GCS blob path (relative path inside bucket).
    """
    if bucket is None:
        return open(path, mode)
    return bucket.blob(path).open(mode)


BLOCK_SIZE = 1999998
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1


class MultiFileWriter:
    """Sequential binary writer to multiple files of up to BLOCK_SIZE each."""
    def __init__(self, base_dir, name, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._name = name
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._file_gen = (
            _open(str(self._base_dir / f"{name}_{i:03}.bin"), "wb", self._bucket)
            for i in itertools.count()
        )
        self._f = next(self._file_gen)

    def write(self, b: bytes):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            if remaining == 0:
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE

            self._f.write(b[:remaining])
            name = self._f.name if hasattr(self._f, "name") else self._f._blob.name
            locs.append((name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class MultiFileReader:
    """Sequential binary reader of multiple files of up to BLOCK_SIZE each."""
    def __init__(self, base_dir, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._open_files = {}

    def _resolve_name(self, f_name: str) -> str:
        """
        Robustly resolve file name:
        - if f_name already contains base_dir prefix (e.g. "postings_gcp/title_000.bin"),
          avoid duplicating base_dir/base_dir.
        - if running locally and f_name is absolute, keep it.
        - for GCS, f_name should be a path relative to bucket root.
        """
        # If it's an absolute local path (rare), keep as-is
        if os.path.isabs(f_name):
            return f_name

        base = str(self._base_dir).rstrip("/") + "/"
        if f_name.startswith(base):
            # already resolved
            return f_name

        # If f_name already starts with something like "postings_gcp/.."
        # then joining base_dir again would cause "postings_gcp/postings_gcp/.."
        base_last = self._base_dir.name.rstrip("/") + "/"
        if f_name.startswith(base_last):
            # assume base_dir is the parent; just join parent of base_dir
            # but for GCS we want base_dir/f_name (base_dir should be correct root)
            return str(self._base_dir.parent / f_name) if self._bucket is None else str(Path(f_name))

        # default: base_dir / f_name
        return str(self._base_dir / f_name)

    def read(self, locs, n_bytes: int) -> bytes:
        b = []
        for f_name, offset in locs:
            resolved = self._resolve_name(str(f_name))

            if resolved not in self._open_files:
                self._open_files[resolved] = _open(resolved, "rb", self._bucket)

            f = self._open_files[resolved]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
            if n_bytes <= 0:
                break

        return b"".join(b)

    def close(self):
        for f in self._open_files.values():
            try:
                f.close()
            except Exception:
                pass
        self._open_files = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class InvertedIndex:
    def __init__(self, docs=None):
        self.df = Counter()
        self.term_total = Counter()
        self._posting_list = defaultdict(list)
        self.posting_locs = defaultdict(list)
        if docs:
            for doc_id, tokens in docs.items():
                self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name, bucket_name=None):
        self._write_globals(base_dir, name, bucket_name)

    def _write_globals(self, base_dir, name, bucket_name):
        path = str(Path(base_dir) / f"{name}.pkl")
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, "wb", bucket) as f:
            pickle.dump(self, f)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_posting_list" in state:
            del state["_posting_list"]
        return state

    def posting_lists_iter(self, base_dir, bucket_name=None):
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], "big")
                    tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], "big")
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    def read_a_posting_list(self, base_dir, w, bucket_name=None):
        if w not in self.posting_locs:
            return []
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            locs = self.posting_locs[w]
            b = reader.read(locs, self.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(self.df[w]):
                doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], "big")
                tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], "big")
                posting_list.append((doc_id, tf))
            return posting_list

    @staticmethod
    def write_a_posting_list(b_w_pl, base_dir, bucket_name=None):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl
        with closing(MultiFileWriter(base_dir, bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl:
                b = b"".join(
                    [(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, "big")
                     for doc_id, tf in pl]
                )
                locs = writer.write(b)
                posting_locs[w].extend(locs)
            path = str(Path(base_dir) / f"{bucket_id}_posting_locs.pickle")
            bucket = None if bucket_name is None else get_bucket(bucket_name)
            with _open(path, "wb", bucket) as f:
                pickle.dump(posting_locs, f)
        return bucket_id

    @staticmethod
    def read_index(base_dir, name, bucket_name=None):
        path = str(Path(base_dir) / f"{name}.pkl")
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, "rb", bucket) as f:
            return pickle.load(f)
