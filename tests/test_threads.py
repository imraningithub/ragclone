import pytest

from app.threads import ThreadStore


def test_thread_crud_and_messages(tmp_path):
    ts = ThreadStore(str(tmp_path / "t.db"))
    a = ts.create()
    b = ts.create("Second")
    assert a["title"] == "New chat"

    ts.add_message(a["id"], "user", "hi")
    ts.add_message(a["id"], "assistant", "hello", [{"n": 1, "file_name": "x.txt"}])
    assert [t["id"] for t in ts.list()][0] == a["id"]  # most recently updated first

    got = ts.get(a["id"])
    assert [m["role"] for m in got["messages"]] == ["user", "assistant"]
    assert got["messages"][1]["citations"][0]["file_name"] == "x.txt"

    assert ts.rename(b["id"], "Renamed")["title"] == "Renamed"
    with pytest.raises(ValueError):
        ts.rename(b["id"], "  ")
    with pytest.raises(ValueError):
        ts.add_message(a["id"], "robot", "x")

    ts.delete(a["id"])
    with pytest.raises(KeyError):
        ts.get(a["id"])
    with pytest.raises(KeyError):
        ts.add_message(a["id"], "user", "x")
    ts.close()


def test_threads_scoped_by_owner(tmp_path):
    ts = ThreadStore(str(tmp_path / "t.db"))
    mine = ts.create(owner="alice")
    ts.create(owner="bob")
    assert [t["id"] for t in ts.list(owner="alice")] == [mine["id"]]
    assert len(ts.list()) == 2  # no owner = unscoped (chat mode)
    assert ts.get(mine["id"], owner="alice")["id"] == mine["id"]
    with pytest.raises(KeyError):
        ts.get(mine["id"], owner="bob")


def test_old_threads_db_gets_owner_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE threads (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL,"
                       " updated_at REAL NOT NULL); INSERT INTO threads VALUES ('t1', 'Old', 1, 1);")
    conn.close()
    ts = ThreadStore(str(path))
    assert ts.list(owner="alice") == []  # legacy threads belong to nobody
    assert [t["id"] for t in ts.list()] == ["t1"]
    assert ts.create(owner="alice")["id"] in [t["id"] for t in ts.list(owner="alice")]
