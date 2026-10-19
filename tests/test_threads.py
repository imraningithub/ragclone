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
