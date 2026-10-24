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
from datetime import date

import chromadb
import numpy as np

from src.config import DOC_COLLECTION, EMBEDDING_MODEL, EMBEDDING_PROVIDER, META_COLLECTION, ROOT
from src.vectorstore.store import chroma_path

SNAPSHOT_DIR = ROOT / "data" / "index"
MANIFEST = SNAPSHOT_DIR / "manifest.json"   # 어떤 임베딩 모델로 만든 벡터인지 — 다른 모델 설정에서 복원하면 검색이 틀어진다
LEGACY_EMBEDDING_MODEL = "text-embedding-3-small"   # manifest가 없던 시절의 스냅샷은 모두 이 모델
COLLECTIONS = (META_COLLECTION, DOC_COLLECTION)
BATCH = 1000


def _paths(name: str):
    return SNAPSHOT_DIR / f"{name}.json.gz", SNAPSHOT_DIR / f"{name}.npz"


def exists() -> bool:
    return all(p.exists() for name in COLLECTIONS for p in _paths(name))


def manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"embedding_provider": "openai", "embedding_model": LEGACY_EMBEDDING_MODEL}


def embedding_model() -> str:
    return manifest()["embedding_model"]


def compatible() -> bool:
    """스냅샷의 임베딩 모델이 현재 설정(config.EMBEDDING_MODEL)과 같은가. 다르면 복원하지 말고 다시 색인해야 한다."""
    return embedding_model() == EMBEDDING_MODEL


def export() -> dict:
    client = chromadb.PersistentClient(path=chroma_path())
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    counts, dim = {}, None
    for name in COLLECTIONS:
        got = client.get_collection(name).get(include=["documents", "metadatas", "embeddings"])
        meta_path, emb_path = _paths(name)
        with gzip.open(meta_path, "wt", encoding="utf-8") as f:
            json.dump({"ids": got["ids"], "documents": got["documents"], "metadatas": got["metadatas"]},
                      f, ensure_ascii=False)
        emb = np.asarray(got["embeddings"], dtype=np.float16)
        np.savez_compressed(emb_path, embeddings=emb)
        counts[name] = len(got["ids"])
        dim = dim or (int(emb.shape[1]) if emb.ndim == 2 and len(emb) else None)
    MANIFEST.write_text(json.dumps({
        "embedding_provider": EMBEDDING_PROVIDER, "embedding_model": EMBEDDING_MODEL, "dim": dim,
        "counts": counts, "exported_at": date.today().isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


def restore(progress=lambda msg: None) -> dict:
    """스냅샷으로 컬렉션을 다시 만든다. 이미 있는 컬렉션은 지우고 채운다."""
    if not compatible():
        raise RuntimeError(
            f"스냅샷은 임베딩 모델 '{embedding_model()}'로 만든 것인데 현재 설정은 '{EMBEDDING_MODEL}'입니다. "
            "복원하지 않습니다. 색인을 다시 만드세요: python -m src.vectorstore.index_meta --reset → "
            "python -m src.vectorstore.index_docs --rebuild → python -m src.vectorstore.snapshot export")
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
        print("스냅샷 저장:", export(), f"(임베딩 {EMBEDDING_MODEL}) →", SNAPSHOT_DIR)
    elif cmd == "import":
        if not exists():
            raise SystemExit(f"{SNAPSHOT_DIR}에 스냅샷이 없습니다.")
        try:
            print("색인 복원:", restore(progress=print))
        except RuntimeError as e:
            raise SystemExit(f"[중단] {e}") from e
    else:
        raise SystemExit("사용법: python -m src.vectorstore.snapshot export|import")


if __name__ == "__main__":
    main()
