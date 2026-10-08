"""Worker process for tests/test_chroma_multiprocess.py.

Usage: python chroma_mp_worker.py <role> <store_dir> <mode>
  seed    create the collection (small HNSW thresholds so saves happen fast)
  stale   load the index, wait for <store_dir>/../go.flag, then write late docs
  writer  write fresh docs, then exit
  check   print JSON: unreadable fresh ids and self-retrieval hits
mode: "session" writes through chroma_client.write_session, "naive" through
a long-held client (the pre-fix behaviour).
"""
import json
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

role, store, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3]

import config  # noqa: E402

config.CHROMA_DB_DIR = store
config.CHROMA_LOCKS_DIR = store / ".locks"
import chroma_lock  # noqa: E402

chroma_lock.CHROMA_LOCKS_DIR = store / ".locks"
import chroma_client  # noqa: E402

NAME = "mp_test"
META = {"hnsw:space": "cosine", "hnsw:batch_size": 5, "hnsw:sync_threshold": 10}
GO = store.parent / "go.flag"
FRESH = [f"fresh-{i}" for i in range(20)]


def collection(client):
    return client.get_or_create_collection(NAME, metadata=META)


def write(client_or_none, ids, texts):
    if mode == "session":
        with chroma_client.write_session(NAME) as client:
            collection(client).add(ids=ids, documents=texts)
    else:
        collection(client_or_none or chroma_client.get_client()).add(ids=ids, documents=texts)


if role == "seed":
    collection(chroma_client.get_client()).add(
        ids=[f"seed-{i}" for i in range(50)],
        documents=[f"seed document {i} about topic {i % 7}" for i in range(50)],
    )
elif role == "stale":
    held = chroma_client.get_client()
    collection(held).query(query_texts=["topic"], n_results=3)
    print("loaded", flush=True)
    while not GO.exists():
        time.sleep(0.1)
    write(held, [f"late-{i}" for i in range(15)], [f"late write {i} from a long-lived process" for i in range(15)])
elif role == "reader":
    # Long-lived reader (like the MCP server): loads, then searches again
    # after another process wrote. Prints how many fresh docs it can find.
    col = collection(chroma_client.get_client())
    col.query(query_texts=["topic"], n_results=3)
    print("loaded", flush=True)
    while not GO.exists():
        time.sleep(0.1)
    col = collection(chroma_client.get_client())
    found = sum(
        f"fresh-{i}" in col.query(query_texts=[f"fresh note {i} about zebra{i}"], n_results=3)["ids"][0]
        for i in range(20)
    )
    print(json.dumps({"found": found}), flush=True)
elif role == "writer":
    write(None, FRESH, [f"fresh note {i} about zebra{i}" for i in range(20)])
elif role == "check":
    col = collection(chroma_client.get_client())
    unreadable = []
    for i in FRESH + [f"late-{i}" for i in range(15)]:
        try:
            col.get(ids=[i], include=["embeddings"])
        except Exception:
            unreadable.append(i)
    got = col.get(ids=FRESH, include=["documents"])
    hits = sum(i in col.query(query_texts=[d], n_results=3)["ids"][0] for i, d in zip(got["ids"], got["documents"]))
    print(json.dumps({"unreadable": unreadable, "self_hits": hits, "count": col.count()}))
