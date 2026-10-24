# src/vectorstore/index_meta.py
# 메타데이터 색인 (RAG 1): 특허 1건 = 청크 1개, 청크 ID = 출원번호.
# 실행: python -m src.vectorstore.index_meta [--reset] (프로젝트 루트에서)
import argparse

from langchain_chroma import Chroma

from src.config import META_COLLECTION
from src.data_prep.items import Item, load_items
from src.env import MissingAPIKeyError
from src.vectorstore.store import get_store

BATCH = 100   # 임베딩 요청 1회에 보낼 항목 수


def index_items(items: list[Item], store: Chroma, prune: bool = True) -> dict:
    """항목을 출원번호 ID로 upsert한다. 몇 번을 다시 실행해도 청크 수는 항목 수와 같다.

    - 텍스트·메타데이터가 그대로인 항목은 다시 임베딩하지 않는다.
    - prune=True면 원본에서 사라진 항목의 청크는 지운다. 검색 중 몇 건만 더할 때는 False.
    """
    existing = store.get(ids=None if prune else [it.id for it in items], include=["documents", "metadatas"])
    current = {i: (doc, meta) for i, doc, meta in zip(existing["ids"], existing["documents"], existing["metadatas"])}

    changed = [it for it in items if current.get(it.id) != (it.text, it.metadata)]
    for start in range(0, len(changed), BATCH):
        batch = changed[start:start + BATCH]
        store.add_texts(   # langchain-chroma의 add_texts는 같은 ID를 덮어쓴다 (upsert)
            texts=[it.text for it in batch],
            metadatas=[it.metadata for it in batch],
            ids=[it.id for it in batch],
        )
        print(f"  임베딩·저장 {start + len(batch)}/{len(changed)}")

    stale = sorted(set(current) - {it.id for it in items}) if prune else []
    if stale:
        store.delete(ids=stale)

    return {
        "loaded": len(items),
        "upserted": len(changed),
        "unchanged": len(items) - len(changed),
        "deleted": len(stale),
        "count": len(store.get(include=[])["ids"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="특허 메타데이터 색인")
    ap.add_argument("--reset", action="store_true",
                    help="컬렉션을 지우고 처음부터 색인한다 (인덱스 파일이 손상돼 열리지 않을 때)")
    args = ap.parse_args()
    try:
        items = load_items()
        store = get_store(META_COLLECTION)
        if args.reset:
            store.delete_collection()
            store = get_store(META_COLLECTION)
    except (FileNotFoundError, MissingAPIKeyError) as e:
        raise SystemExit(f"[중단] {e}") from e

    print(f"'{META_COLLECTION}' 컬렉션에 항목 {len(items)}건 색인 시작")
    s = index_items(items, store)
    print(f"새로 저장·갱신 {s['upserted']}건 / 변경 없음 {s['unchanged']}건 / 삭제 {s['deleted']}건")
    print(f"저장된 청크 수: {s['count']}건 (로드한 항목 {s['loaded']}건과 "
          f"{'일치' if s['count'] == s['loaded'] else '불일치'})")


if __name__ == "__main__":
    main()
