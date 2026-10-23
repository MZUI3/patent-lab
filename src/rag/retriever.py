# src/rag/retriever.py — 근거 검색 및 병합 (F2006)
# RAG 1(patent_meta)에서 선택한 특허 청크를 검색 없이 항상 넣고,
# RAG 2(patent_docs)에서 그 특허로 범위를 좁혀 질문과 비슷한 청크를 찾은 뒤 합친다.
import re
from dataclasses import dataclass

from langchain_chroma import Chroma

from src.config import DOC_COLLECTION, META_COLLECTION, MMR_FETCH_K, MMR_LAMBDA, SEARCH_TYPE, TOP_K
from src.data_prep.docs import SOURCE_LABELS
from src.data_prep.items import Item
from src.vectorstore.store import get_store


@dataclass
class Evidence:
    n: int              # 근거 번호 — 답변의 [근거 n]과 화면 목록을 잇는다
    chunk_id: str
    source_type: str    # item_meta | patent_claims | wikipedia
    title: str
    url: str
    text: str

    @property
    def label(self) -> str:
        return SOURCE_LABELS.get(self.source_type, self.source_type)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def search_docs(item: Item, question: str, store: Chroma, top_k: int, search_type: str):
    """선택한 특허의 외부 문서 청크만 검색한다. item_id 필터가 빠지면 다른 특허의 문서가 섞인다."""
    where = {"item_id": item.id}
    if search_type == "mmr":   # 넉넉히 가져와 내용이 겹치지 않게 고른다 (F2014)
        return store.max_marginal_relevance_search(question, k=top_k, fetch_k=max(MMR_FETCH_K, top_k),
                                                   lambda_mult=MMR_LAMBDA, filter=where)
    return store.similarity_search(question, k=top_k, filter=where)


def retrieve_evidence(item: Item, question: str, top_k: int = TOP_K, search_type: str = SEARCH_TYPE,
                      meta_store: Chroma | None = None, doc_store: Chroma | None = None) -> list[Evidence]:
    """근거 목록 = 특허 메타데이터 1건 + 외부 문서 청크 최대 top_k개. 같은 청크·같은 내용은 한 번만 넣는다."""
    meta_store = meta_store or get_store(META_COLLECTION)
    doc_store = doc_store or get_store(DOC_COLLECTION)

    meta = meta_store.get(ids=[item.id], include=["documents", "metadatas"])
    found = [(item.id, meta["documents"][0], meta["metadatas"][0])] if meta["ids"] else \
            [(item.id, item.text, item.metadata)]   # 색인 전이어도 선택한 특허 정보는 근거에 넣는다
    for d in search_docs(item, question, doc_store, top_k, search_type):
        m = d.metadata
        found.append((f"{m['item_id']}:{m['doc_id']}:{m['chunk_index']}", d.page_content, m))

    evidence, seen_ids, seen_text = [], set(), set()
    for chunk_id, text, m in found:
        if chunk_id in seen_ids or _norm(text) in seen_text:
            continue
        seen_ids.add(chunk_id)
        seen_text.add(_norm(text))
        is_meta = m.get("source_type") == "item_meta"
        evidence.append(Evidence(
            n=len(evidence) + 1,
            chunk_id=chunk_id,
            source_type=m["source_type"],
            title=m["title"] if is_meta else m["doc_title"],
            url=m["source_url"],
            text=text,
        ))
    return evidence


def format_evidence(evidence: list[Evidence]) -> str:
    """프롬프트3에 넣는 근거 블록. 번호·출처 유형·제목을 머리에 단다."""
    return "\n\n".join(f"[근거 {e.n}] ({e.label}) {e.title}\n{e.text}" for e in evidence)
