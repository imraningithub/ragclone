import math
import sqlite3
import threading

import numpy as np
import pytest

from app.store import DimensionMismatch, DuplicateFile, NotFound, VectorStore, _vec_table


def _probe_ann(tmp_path) -> bool:
    s = VectorStore(str(tmp_path / "probe.db"))
    ok = s.ann_available
    s.close()
    return ok


@pytest.fixture
def store(tmp_path):
    s = VectorStore(str(tmp_path / "store.db"))
    yield s
    s.close()


@pytest.fixture(params=["linear", "ann"])
def mstore(request, tmp_path):
    """Store parametrized over search modes; ANN skipped if sqlite-vec unavailable."""
    if request.param == "ann":
        s = VectorStore(str(tmp_path / "ann.db"))
        if not s.ann_available:
            s.close()
            pytest.skip("sqlite-vec not available")
    else:
        s = VectorStore(str(tmp_path / "lin.db"), enable_ann=False)
        assert s.ann_available is False
    s.mode = request.param
    yield s
    s.close()


def _seed(s, coll="c"):
    s.create_collection(coll, 4, "m")
    fa = s.add_file(coll, name="a.txt", path="/a", type="text/plain", size=1, mode="embeddings")
    fb = s.add_file(coll, name="b.txt", path="/b", type="text/plain", size=1, mode="embeddings")
    s.insert_chunks(coll, fa, ["x-axis", "y-axis"], [[1, 0, 0, 0], [0, 1, 0, 0]])
    s.insert_chunks(coll, fb, ["z-axis", "xy"], [[0, 0, 1, 0], [0.7, 0.7, 0, 0]])
    return fa, fb


# ---------------------------------------------------------------- collections
def test_create_collection_idempotent_and_mismatch(store):
    store.create_collection("c", 4, "m")
    first = store.get_collection("c")
    store.create_collection("c", 4, "m")
    assert store.get_collection("c") == first
    assert set(first) == {"name", "dim", "embed_model", "created_at"}
    assert first["dim"] == 4 and first["embed_model"] == "m"
    with pytest.raises(DimensionMismatch):
        store.create_collection("c", 8, "m")
    with pytest.raises(DimensionMismatch):
        store.create_collection("c", 4, "other")
    assert store.get_collection("missing") is None


def test_inline_collection_upgrade(store):
    fid = store.add_file("t", name="n.md", path="/n", type="text/markdown", size=3,
                         mode="inline", inline_text="hello")
    assert store.get_collection("t")["dim"] == 0
    # embeddings mode needs an existing collection
    with pytest.raises(NotFound):
        store.add_file("nope", name="x", path="/x", type="t", size=1, mode="embeddings")
    store.create_collection("t", 3, "emb")  # upgrade, not mismatch
    c = store.get_collection("t")
    assert c["dim"] == 3 and c["embed_model"] == "emb"
    assert store.get_file("t", fid)["inline_text"] == "hello"
    f2 = store.add_file("t", name="e.txt", path="/e", type="text/plain", size=1, mode="embeddings")
    store.insert_chunks("t", f2, ["hi"], [[1, 0, 0]])
    assert store.search("t", [1, 0, 0])[0]["file_id"] == f2
    with pytest.raises(DimensionMismatch):
        store.create_collection("t", 5, "emb")


# ---------------------------------------------------------------------- files
def test_add_list_get_delete_file(store):
    store.create_collection("c", 2, "m")
    f1 = store.add_file("c", name="a.pdf", path="/p/a", type="application/pdf", size=10, mode="embeddings")
    f2 = store.add_file("c", name="b.txt", path="/p/b", type="text/plain", size=5,
                        mode="inline", inline_text="body")
    assert len(f1) == 32
    files = store.list_files("c")
    assert [f["id"] for f in files] == [f1, f2]
    assert set(files[0]) == {"id", "collection", "name", "path", "type", "size", "mode",
                             "chunk_count", "created_at"}
    g = store.get_file("c", f2)
    assert g["inline_text"] == "body" and g["mode"] == "inline" and g["size"] == 5
    store.insert_chunks("c", f1, ["a", "b"], [[1, 0], [0, 1]])
    store.delete_file("c", f1)
    assert [f["id"] for f in store.list_files("c")] == [f2]
    with pytest.raises(NotFound):
        store.get_file("c", f1)
    with pytest.raises(NotFound):
        store.delete_file("c", f1)
    assert store.search("c", [1, 0], threshold=-1) == []
    with pytest.raises(ValueError):
        store.add_file("c", name="z", path="/z", type="t", size=1, mode="bogus")


def test_duplicate_file(store):
    store.create_collection("c", 2, "m")
    store.add_file("c", name="a", path="/a", type="t", size=1, mode="embeddings")
    with pytest.raises(DuplicateFile):
        store.add_file("c", name="a", path="/a", type="t", size=1, mode="embeddings")
    # same name, different path is fine; same file in another collection is fine
    store.add_file("c", name="a", path="/other", type="t", size=1, mode="embeddings")
    store.create_collection("d", 2, "m")
    store.add_file("d", name="a", path="/a", type="t", size=1, mode="embeddings")


def test_insert_and_get_chunks(store):
    store.create_collection("c", 2, "m")
    fid = store.add_file("c", name="a", path="/a", type="t", size=1, mode="embeddings")
    assert store.insert_chunks("c", fid, ["c0", "c1", "c2"], [[1, 0], [0, 1], [1, 1]]) == 3
    assert store.insert_chunks("c", fid, ["c3"], [[1, 2]]) == 1
    assert store.get_file("c", fid)["chunk_count"] == 4
    all_chunks = store.get_chunks("c", fid)
    assert [c["chunk_index"] for c in all_chunks] == [0, 1, 2, 3]
    assert [c["text"] for c in all_chunks] == ["c0", "c1", "c2", "c3"]
    assert set(all_chunks[0]) == {"id", "file_id", "chunk_index", "text", "meta"}
    assert all_chunks[0]["meta"] == {}
    assert [c["text"] for c in store.get_chunks("c", fid, 1, 3)] == ["c1", "c2"]
    assert [c["text"] for c in store.get_chunks("c", fid, 2)] == ["c2", "c3"]
    assert store.get_chunks("c", fid, 10) == []
    with pytest.raises(NotFound):
        store.get_chunks("c", "nope")
    with pytest.raises(NotFound):
        store.insert_chunks("c", "nope", ["x"], [[1, 0]])


@pytest.mark.parametrize(
    "emb",
    [[1.0], [1.0, 2.0, 3.0], [math.nan, 1.0], [math.inf, 0.0], []],
)
def test_invalid_embeddings(store, emb):
    store.create_collection("c", 2, "m")
    fid = store.add_file("c", name="a", path="/a", type="t", size=1, mode="embeddings")
    with pytest.raises(ValueError):
        store.insert_chunks("c", fid, ["x"], [emb])
    assert store.get_file("c", fid)["chunk_count"] == 0
    with pytest.raises(ValueError):
        store.search("c", emb)


def test_insert_length_mismatch_and_atomicity(store):
    store.create_collection("c", 2, "m")
    fid = store.add_file("c", name="a", path="/a", type="t", size=1, mode="embeddings")
    with pytest.raises(ValueError):
        store.insert_chunks("c", fid, ["a", "b"], [[1, 0]])
    with pytest.raises(ValueError, match="non-finite"):
        store.insert_chunks("c", fid, ["a", "b"], [[1, 0], [math.nan, 0]])
    assert store.get_chunks("c", fid) == []


# --------------------------------------------------------------------- search
def test_search_top_hit(mstore):
    fa, fb = _seed(mstore)
    hits = mstore.search("c", [1, 0, 0, 0], limit=3, threshold=0.1, mode=mstore.mode)
    assert hits[0]["text"] == "x-axis" and hits[0]["file_id"] == fa
    assert hits[0]["file_name"] == "a.txt" and hits[0]["chunk_index"] == 0
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert set(hits[0]) == {"chunk_id", "file_id", "file_name", "chunk_index", "text", "score", "meta"}
    assert [h["text"] for h in hits] == ["x-axis", "xy"]  # orthogonal chunks (score 0) dropped
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_threshold_and_limit(mstore):
    _seed(mstore)
    q = [1, 0.1, 0, 0]
    hits = mstore.search("c", q, limit=10, threshold=-1.0, mode=mstore.mode)
    assert len(hits) == 4
    hits = mstore.search("c", q, limit=10, threshold=0.5, mode=mstore.mode)
    assert [h["text"] for h in hits] == ["x-axis", "xy"]
    assert all(h["score"] >= 0.5 for h in hits)
    assert len(mstore.search("c", q, limit=1, threshold=-1.0, mode=mstore.mode)) == 1
    assert mstore.search("c", q, threshold=0.9999, mode=mstore.mode) == []


def test_search_file_ids_filter(mstore):
    fa, fb = _seed(mstore)
    hits = mstore.search("c", [1, 0, 0, 0], limit=5, threshold=-1.0, mode=mstore.mode, file_ids=[fb])
    assert {h["file_id"] for h in hits} == {fb}
    assert hits[0]["text"] == "xy"
    assert mstore.search("c", [1, 0, 0, 0], mode=mstore.mode, file_ids=[]) == []


def test_ann_and_linear_agree(tmp_path):
    if not _probe_ann(tmp_path):
        pytest.skip("sqlite-vec not available")
    s = VectorStore(str(tmp_path / "s.db"))
    _seed(s)
    for q in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.6, 0.8, 0, 0]):
        a = s.search("c", q, limit=4, threshold=-1, mode="ann")
        l = s.search("c", q, limit=4, threshold=-1, mode="linear")
        assert a[0]["chunk_id"] == l[0]["chunk_id"]
        assert a[0]["score"] == pytest.approx(l[0]["score"], abs=1e-5)
    s.close()


def test_ann_mode_unavailable_raises(tmp_path):
    s = VectorStore(str(tmp_path / "s.db"), enable_ann=False)
    _seed(s)
    with pytest.raises(RuntimeError):
        s.search("c", [1, 0, 0, 0], mode="ann")
    assert s.search("c", [1, 0, 0, 0], mode="auto")[0]["text"] == "x-axis"
    with pytest.raises(ValueError):
        s.search("c", [1, 0, 0, 0], mode="weird")
    s.close()


def test_search_missing_or_inline_collection_returns_empty(store):
    assert store.search("nope", [1, 0]) == []
    store.add_file("t", name="n", path="/n", type="t", size=1, mode="inline", inline_text="x")
    assert store.search("t", [1, 0]) == []


def test_vec_backfill_after_linear_inserts(tmp_path):
    if not _probe_ann(tmp_path):
        pytest.skip("sqlite-vec not available")
    db = str(tmp_path / "s.db")
    s = VectorStore(db, enable_ann=False)
    _seed(s)
    s.close()
    s = VectorStore(db)
    assert s.search("c", [0, 0, 1, 0], mode="ann")[0]["text"] == "z-axis"
    s.close()


# ----------------------------------------------------------- delete / status
def test_delete_collection_cleans_everything(store):
    _seed(store)
    table = _vec_table("c")
    if store.ann_available:
        assert store._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone()
    store.delete_collection("c")
    assert store.get_collection("c") is None
    assert store.list_files("c") == []
    assert store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert store._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone() is None
    store.delete_collection("c")  # idempotent
    # recreate with a different dim works after deletion
    store.create_collection("c", 2, "m2")
    assert store.get_collection("c")["dim"] == 2


def test_delete_file_removes_vec_rows(store):
    if not store.ann_available:
        pytest.skip("sqlite-vec not available")
    fa, fb = _seed(store)
    store.delete_file("c", fa)
    n = store._conn.execute(f"SELECT COUNT(*) FROM {_vec_table('c')}").fetchone()[0]
    assert n == 2
    hits = store.search("c", [1, 0, 0, 0], threshold=-1, mode="ann")
    assert {h["file_id"] for h in hits} == {fb}


def test_status(store):
    assert store.status() == {"ann_available": store.ann_available, "collections": 0}
    store.create_collection("a", 2, "m")
    store.create_collection("b", 3, "m")
    assert store.status()["collections"] == 2


def test_concurrent_inserts(store):
    store.create_collection("c", 2, "m")
    fid = store.add_file("c", name="a", path="/a", type="t", size=1, mode="embeddings")

    def work():
        for _ in range(20):
            store.insert_chunks("c", fid, ["t"], [[1, 0]])

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.get_file("c", fid)["chunk_count"] == 80
    assert [c["chunk_index"] for c in store.get_chunks("c", fid)] == list(range(80))


# ----------------------------------------------------------------------- meta
def test_meta_round_trip(mstore):
    mstore.create_collection("c", 2, "m")
    fid = mstore.add_file("c", name="a.pdf", path="/a", type="application/pdf", size=1, mode="embeddings")
    mstore.insert_chunks("c", fid, ["p1", "p2", "none"], [[1, 0], [0, 1], [1, 1]],
                         metas=[{"page": 1}, {"page": 2, "title": "Prüfung"}, {}])
    assert [c["meta"] for c in mstore.get_chunks("c", fid)] == [{"page": 1}, {"page": 2, "title": "Prüfung"}, {}]
    hits = mstore.search("c", [0, 1], limit=3, threshold=-1, mode=mstore.mode)
    assert hits[0]["text"] == "p2" and hits[0]["meta"] == {"page": 2, "title": "Prüfung"}
    assert {h["text"]: h["meta"] for h in hits}["none"] == {}
    with pytest.raises(ValueError):
        mstore.insert_chunks("c", fid, ["x"], [[1, 0]], metas=[{}, {}])


def test_meta_column_added_to_old_db(tmp_path):
    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE collections (name TEXT PRIMARY KEY, dim INTEGER NOT NULL,
            embed_model TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL);
        CREATE TABLE files (id TEXT PRIMARY KEY, collection TEXT NOT NULL, name TEXT NOT NULL,
            path TEXT NOT NULL, type TEXT NOT NULL DEFAULT '', size INTEGER NOT NULL DEFAULT 0,
            mode TEXT NOT NULL, chunk_count INTEGER NOT NULL DEFAULT 0, inline_text TEXT,
            created_at REAL NOT NULL, UNIQUE(collection, name, path));
        CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, collection TEXT NOT NULL,
            file_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL);
    """)
    conn.execute("INSERT INTO collections VALUES ('c', 2, 'm', 0)")
    conn.execute("INSERT INTO files VALUES ('f1', 'c', 'a', '/a', 't', 1, 'embeddings', 1, NULL, 0)")
    conn.execute("INSERT INTO chunks VALUES (1, 'k1', 'c', 'f1', 0, 'old', ?)",
                 (np.array([1, 0], dtype=np.float32).tobytes(),))
    conn.commit()
    conn.close()

    s = VectorStore(db, enable_ann=False)
    cols = {r[1] for r in s._conn.execute("PRAGMA table_info(chunks)").fetchall()}
    assert "meta" in cols
    assert s.get_chunks("c", "f1")[0]["meta"] == {}
    assert s.search("c", [1, 0], threshold=-1)[0]["meta"] == {}
    s.insert_chunks("c", "f1", ["new"], [[0, 1]], metas=[{"page": 7}])
    assert s.get_chunks("c", "f1")[1]["meta"] == {"page": 7}
    s.close()
    VectorStore(db, enable_ann=False).close()  # reopening a migrated DB is a no-op
