# src/vectorstore/index_docs.py
# 외부 문서 색인 (RAG 2): 수집한 문서를 청킹해 patent_docs 컬렉션에 넣는다.
# 청크 ID = "항목ID:문서ID:순번" — 여러 특허가 같은 위키 문서를 수집해도 서로 덮어쓰지 않는다.
#
# 실행 (프로젝트 루트에서)
#   python -m src.vectorstore.index_docs --item 1020230136220 --question "지터는 왜 측정하나?"   수집+색인
#   python -m src.vectorstore.index_docs --rebuild   청크 설정을 바꾼 뒤 캐시로 전체 재색인 (외부 호출 없음)
#   python -m src.vectorstore.index_docs             컬렉션 현황
import argparse
import time
from collections import Counter

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DOC_COLLECTION
from src.data_prep import kipris, wiki
from src.data_prep.collect_docs import CollectResult, apply_length_rule, collect, load_log, mark_collected
from src.data_prep.docs import SOURCE_LABELS, SourceDoc
from src.data_prep.items import Item, load_items
from src.env import MissingAPIKeyError
from src.vectorstore.store import get_store


def get_splitter() -> RecursiveCharacterTextSplitter:
    # 문단(빈 줄) → 줄 → 문장 → 공백 순으로 자른다. 청구항은 빈 줄로 나뉘어 있어 청구항 단위로 잘린다.
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""])


def index_documents(item: Item, docs: list[SourceDoc], store: Chroma) -> int:
    """문서별로 이전 청크를 지우고 새로 넣는다. 넣은 청크 수를 돌려준다."""
    splitter = get_splitter()
    today = time.strftime("%Y-%m-%d")
    total = 0
    for doc in docs:
        store.delete(where={"$and": [{"item_id": item.id}, {"doc_id": doc.doc_id}]})
        chunks = splitter.split_text(doc.text)
        store.add_texts(
            # 청크 앞에 문서 제목을 붙여, 본문만으로는 주제가 드러나지 않는 청크도 검색에 걸리게 한다.
            texts=[f"[{doc.title}]\n{c}" for c in chunks],
            ids=[f"{item.id}:{doc.doc_id}:{i}" for i in range(len(chunks))],
            metadatas=[{
                "item_id": item.id,
                "item_title": item.metadata["title"],
                "doc_id": doc.doc_id,
                "doc_title": doc.title,
                "source_url": doc.url,
                "source_type": doc.source_type,
                "search_term": doc.search_term,
                "chunk_index": i,
                "collected_at": today,
            } for i in range(len(chunks))],
        )
        total += len(chunks)
    return total


def ensure_documents(item: Item, question: str, llm: BaseChatModel | None = None,
                     store: Chroma | None = None) -> tuple[CollectResult, int]:
    """질문에 필요한 문서를 수집·색인한다. 이미 수집한 검색어·문서는 건너뛴다."""
    result = collect(item, question, llm)
    added = index_documents(item, result.docs, store or get_store(DOC_COLLECTION)) if result.docs else 0
    mark_collected(item.id, result)
    return result, added


def rebuild() -> dict:
    """수집 기록과 캐시만으로 컬렉션을 다시 만든다. 청크 크기·중복 길이를 바꾼 뒤 쓴다."""
    items = {it.id: it for it in load_items()}
    get_store(DOC_COLLECTION).delete_collection()
    store = get_store(DOC_COLLECTION)
    per_item = {}
    for item_id, entry in load_log().items():
        docs = []
        for doc_id, info in entry["docs"].items():
            if doc_id.startswith("kipris-"):
                docs.append(kipris.patent_document(item_id))
            else:
                doc = apply_length_rule(wiki.page(int(doc_id.removeprefix("wiki-")), info["search_term"]))
                if doc:
                    docs.append(doc)
        per_item[item_id] = index_documents(items[item_id], docs, store)
    return per_item


def stats(store: Chroma) -> Counter:
    metas = store.get(include=["metadatas"])["metadatas"]
    return Counter(SOURCE_LABELS.get(m["source_type"], m["source_type"]) for m in metas)


def print_result(result: CollectResult, added: int) -> None:
    print(f"검색어(프롬프트1): {result.terms}")
    if result.skipped_terms:
        print(f"  이미 수집해 건너뜀: {result.skipped_terms}")
    print(f"선별(프롬프트2) 색인: {result.selected}")
    print(f"선별(프롬프트2) 버림: {result.rejected}")
    if result.too_short:
        print(f"본문이 짧아 제외: {result.too_short}")
    print("이번에 색인한 문서:")
    for d in result.docs:
        print(f"  - [{SOURCE_LABELS[d.source_type]}] {d.title} ({len(d.text)}자) {d.url}")
    if not result.docs:
        print("  (새로 수집한 문서 없음)")
    print(f"추가된 청크: {added}개")


def main() -> None:
    ap = argparse.ArgumentParser(description="외부 문서 수집·색인")
    ap.add_argument("--item", help="출원번호")
    ap.add_argument("--question", help="질문")
    ap.add_argument("--rebuild", action="store_true", help="캐시로 전체 재색인")
    args = ap.parse_args()

    try:
        if args.rebuild:
            per_item = rebuild()
            print(f"청크 {CHUNK_SIZE}자 / 중복 {CHUNK_OVERLAP}자로 재색인: 항목 {len(per_item)}건, "
                  f"청크 {sum(per_item.values())}개")
        elif args.item:
            if not args.question:
                raise SystemExit("--question을 함께 주세요.")
            item = next((it for it in load_items() if it.id == args.item), None)
            if item is None:
                raise SystemExit(f"출원번호 {args.item}인 특허가 항목 목록에 없습니다.")
            print(f"[{item.id}] {item.metadata['title']}\n질문: {args.question}\n")
            print_result(*ensure_documents(item, args.question))
        store = get_store(DOC_COLLECTION)
        print(f"\n'{DOC_COLLECTION}' 청크 수: {len(store.get(include=[])['ids'])}개 {dict(stats(store))}")
    except (FileNotFoundError, MissingAPIKeyError) as e:
        raise SystemExit(f"[중단] {e}")


if __name__ == "__main__":
    main()
