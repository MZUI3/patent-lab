# src/vectorstore/snapshot.py — 색인 스냅샷: 다른 PC에서 임베딩 호출 없이 바로 색인을 복원한다
# Chroma 저장소(chroma_db/)는 수백 MB의 바이너리라 git에 올리기 어렵고, 버전·OS·경로에 묶인다.
# 대신 컬렉션의 ID·본문·메타데이터(json.gz)와 임베딩(float16 npz)을 data/index/에 저장해 git으로 공유하고,
# 색인이 비어 있으면 이것으로 다시 채운다. 임베딩을 그대로 쓰므로 API 호출이 없다.
#
# 실행 (프로젝트 루트에서)
#   python -m src.vectorstore.snapshot export   현재 색인을 data/index/에 저장 (커밋 전에)
#   python -m src.vectorstore.snapshot import   data/index/로 색인을 채운다 (앱은 색인이 비면 자동으로 한다)
import gzip
import json
import sys

import chromadb
import numpy as np

from src.config import DOC_COLLECTION, META_COLLECTION, ROOT
from src.vectorstore.store import chroma_path

SNAPSHOT_DIR = ROOT / "data" / "index"
COLLECTIONS = (META_COLLECTION, DOC_COLLECTION)
BATCH = 1000


def _paths(name: str):
    return SNAPSHOT_DIR / f"{name}.json.gz", SNAPSHOT_DIR / f"{name}.npz"


def exists() -> bool:
    return all(p.exists() for name in COLLECTIONS for p in _paths(name))


def export() -> dict:
    client = chromadb.PersistentClient(path=chroma_path())
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in COLLECTIONS:
        got = client.get_collection(name).get(include=["documents", "metadatas", "embeddings"])
        meta_path, emb_path = _paths(name)
        with gzip.open(meta_path, "wt", encoding="utf-8") as f:
            json.dump({"ids": got["ids"], "documents": got["documents"], "metadatas": got["metadatas"]},
                      f, ensure_ascii=False)
        np.savez_compressed(emb_path, embeddings=np.asarray(got["embeddings"], dtype=np.float16))
        counts[name] = len(got["ids"])
    return counts


def restore(progress=lambda msg: None) -> dict:
    """스냅샷으로 컬렉션을 다시 만든다. 이미 있는 컬렉션은 지우고 채운다."""
    client = chromadb.PersistentClient(path=chroma_path())
    existing = {c.name for c in client.list_collections()}
    counts = {}
    for name in COLLECTIONS:
        meta_path, emb_path = _paths(name)
        with gzip.open(meta_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        emb = np.load(emb_path)["embeddings"].astype(np.float32)
        if name in existing:
            client.delete_collection(name)
        col = client.create_collection(name, metadata={"hnsw:space": "cosine"})   # store.get_store와 같은 설정
        n = len(data["ids"])
        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            col.add(ids=data["ids"][start:end], documents=data["documents"][start:end],
                    metadatas=data["metadatas"][start:end], embeddings=emb[start:end].tolist())
            progress(f"{name} {end:,}/{n:,}")
        counts[name] = n
    return counts


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "export":
        print("스냅샷 저장:", export(), "→", SNAPSHOT_DIR)
    elif cmd == "import":
        if not exists():
            raise SystemExit(f"{SNAPSHOT_DIR}에 스냅샷이 없습니다.")
        print("색인 복원:", restore(progress=print))
    else:
        raise SystemExit("사용법: python -m src.vectorstore.snapshot export|import")


if __name__ == "__main__":
    main()
