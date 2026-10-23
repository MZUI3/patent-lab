# src/service/explore.py — 조건 기반 특허 탐색 (F2011)
# 자연어 조건으로 메타데이터 색인(RAG 1)에서 후보를 찾고, LLM이 조건을 모두 만족하는 특허만 고른다.
import re
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel

from src.config import EXPLORE_CANDIDATES, EXPLORE_MAX_RESULTS, META_COLLECTION
from src.rag.llm import get_chat, parse_json_array
from src.rag.prompts import CONDITION_FILTER
from src.vectorstore.store import get_store


@dataclass
class Match:
    item_id: str
    title: str
    applicant: str
    application_date: str
    status: str
    reason: str


def year_filter(condition: str) -> dict | None:
    """'2020년 이후', '2010년까지' 같은 연도 조건을 메타데이터 필터로 바꾼다.

    의미 유사도 검색은 연도를 구분하지 못하므로, 후보를 가져오기 전에 범위를 좁힌다.
    """
    clauses = []
    if m := re.search(r"(\d{4})\s*년?\s*(?:이후|부터|이상)", condition):
        clauses.append({"application_year": {"$gte": int(m.group(1))}})
    if m := re.search(r"(\d{4})\s*년?\s*(?:이전|까지|이하)", condition):
        clauses.append({"application_year": {"$lte": int(m.group(1))}})
    if len(clauses) > 1:
        return {"$and": clauses}
    return clauses[0] if clauses else None


def find_by_text(query: str, items: list, limit: int = 30) -> list[Match]:
    """특허 목록에서 명칭·출원인·출원번호에 입력한 단어가 모두 들어간 특허를 최근 출원 순으로 고른다 (F2009 항목 선택).
    LLM·임베딩을 쓰지 않는 단순 목록 검색이라 즉시 나온다."""
    words = [w.lower() for w in query.split() if w.strip()]
    if not words:
        return []
    hits = []
    for it in items:
        m = it.metadata
        haystack = f"{m['title']} {m['applicant']} {m['item_id']}".lower()
        if all(w in haystack for w in words):
            hits.append(Match(m["item_id"], m["title"], m["applicant"], m["application_date"], m["status"], m.get("field", "")))
    hits.sort(key=lambda x: x.application_date, reverse=True)
    return hits[:limit]


def _abstract(text: str) -> str:
    return text.split("요약: ", 1)[1][:200] if "요약: " in text else ""


def explore(condition: str, llm: BaseChatModel | None = None, store: Chroma | None = None) -> list[Match]:
    store = store or get_store(META_COLLECTION)
    docs = store.similarity_search(condition, k=EXPLORE_CANDIDATES, filter=year_filter(condition))
    if not docs:
        return []

    listing = "\n".join(
        f"{i}. {d.metadata['title']} | 출원인: {d.metadata['applicant']} | 출원일: {d.metadata['application_date']}"
        f" | 상태: {d.metadata['status']}\n   요약: {_abstract(d.page_content)}"
        for i, d in enumerate(docs, 1))
    raw = ((CONDITION_FILTER | (llm or get_chat()))
           .invoke({"condition": condition, "candidates": listing, "max_results": EXPLORE_MAX_RESULTS}).content)

    matches, seen = [], set()
    for pick in parse_json_array(raw):
        n = pick.get("n") if isinstance(pick, dict) else None
        if not isinstance(n, int) or not 1 <= n <= len(docs) or n in seen:
            continue
        seen.add(n)
        m = docs[n - 1].metadata
        matches.append(Match(m["item_id"], m["title"], m["applicant"], m["application_date"],
                             m["status"], str(pick.get("reason", ""))))
    return matches[:EXPLORE_MAX_RESULTS]
