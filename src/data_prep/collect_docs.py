# src/data_prep/collect_docs.py
# 외부 문서 수집: 프롬프트1로 검색어를 뽑고, 위키백과 후보를 프롬프트2로 골라 본문을 받는다.
# 선택한 특허 자체의 공보(요약·청구항)는 선별·길이 규칙과 상관없이 항상 수집한다.
import json
from dataclasses import dataclass, field, replace

from langchain_core.language_models import BaseChatModel

from src.config import COLLECT_LOG, DOC_MAX_CHARS, DOC_MIN_CHARS, MAX_SELECTED_DOCS, MAX_TERMS
from src.data_prep import kipris, wiki
from src.data_prep.docs import SourceDoc
from src.data_prep.items import Item
from src.rag.llm import get_chat, parse_json_array
from src.rag.prompts import CANDIDATE_SELECTION, QUERY_EXTRACTION


@dataclass
class CollectResult:
    terms: list[str]                                        # 프롬프트1이 뽑은 검색어
    new_terms: list[str]                                    # 이번에 처음 검색한 검색어
    skipped_terms: list[str]                                # 이미 수집해서 건너뛴 검색어
    question: str = ""                                      # 검색어를 뽑은 질문 (같은 질문이면 검색어를 재사용)
    selected: list[str] = field(default_factory=list)       # 프롬프트2가 고른 후보 제목
    rejected: list[str] = field(default_factory=list)       # 프롬프트2가 버린 후보 제목
    too_short: list[str] = field(default_factory=list)      # 골랐지만 본문이 짧아 뺀 문서
    docs: list[SourceDoc] = field(default_factory=list)     # 이번에 새로 색인할 문서


# ── 수집 기록: 항목별로 처리한 검색어와 색인한 문서 ─────────────────────────────

def load_log() -> dict:
    return json.loads(COLLECT_LOG.read_text(encoding="utf-8")) if COLLECT_LOG.exists() else {}


def mark_collected(item_id: str, result: CollectResult) -> None:
    """색인까지 끝난 뒤에 기록한다. 중간에 실패하면 다음 질문에서 다시 수집된다."""
    log = load_log()
    entry = log.setdefault(item_id, {"terms": [], "docs": {}})
    entry["terms"] += [t for t in result.new_terms if t not in entry["terms"]]
    if result.question:
        entry.setdefault("questions", {})[result.question] = result.terms
    for d in result.docs:
        entry["docs"][d.doc_id] = {"title": d.title, "source_type": d.source_type, "search_term": d.search_term}
    COLLECT_LOG.parent.mkdir(parents=True, exist_ok=True)
    COLLECT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 프롬프트 호출 ─────────────────────────────────────────────────────────────

def extract_terms(item: Item, question: str, llm: BaseChatModel) -> list[str]:
    """프롬프트1."""
    raw = (QUERY_EXTRACTION | llm).invoke(
        {"item_text": item.text, "question": question, "max_terms": MAX_TERMS}).content
    terms = []
    for t in parse_json_array(raw):
        if isinstance(t, str) and t.strip() and t.strip() not in terms:
            terms.append(t.strip())
    return terms[:MAX_TERMS]


def select_candidates(item: Item, question: str, candidates: list[dict], llm: BaseChatModel) -> list[int]:
    """프롬프트2. 고른 후보의 0부터 시작하는 위치를 돌려준다."""
    listing = "\n".join(f"{i}. {c['title']} — {c['snippet']}" for i, c in enumerate(candidates, 1))
    raw = (CANDIDATE_SELECTION | llm).invoke({
        "item_title": item.metadata["title"], "question": question,
        "candidates": listing, "max_docs": MAX_SELECTED_DOCS}).content
    picked = []
    for n in parse_json_array(raw):
        if isinstance(n, int) and 1 <= n <= len(candidates) and n - 1 not in picked:
            picked.append(n - 1)
    return picked[:MAX_SELECTED_DOCS]


# ── 길이 규칙 (위키 문서에만 적용) ────────────────────────────────────────────

def apply_length_rule(doc: SourceDoc) -> SourceDoc | None:
    if len(doc.text) < DOC_MIN_CHARS:
        return None
    if len(doc.text) > DOC_MAX_CHARS:
        return replace(doc, text=doc.text[:DOC_MAX_CHARS])
    return doc


# ── 수집 ─────────────────────────────────────────────────────────────────────

def collect(item: Item, question: str, llm: BaseChatModel | None = None) -> CollectResult:
    entry = load_log().get(item.id, {"terms": [], "docs": {}})
    llm = llm or get_chat()

    docs = []
    own = f"kipris-{item.id}"
    if own not in entry["docs"]:              # 특허 자체 문서는 어떤 규칙보다 우선해 항상 수집
        docs.append(kipris.patent_document(item.id))

    # 같은 질문은 지난번 검색어를 그대로 쓴다. 프롬프트1은 온도 0이어도 가끔 다른 검색어를 내서,
    # 그대로 두면 같은 질문에 수집이 다시 일어난다 (같은 질문에 수집이 반복되면 안 된다).
    question_key = " ".join(question.split())
    terms = entry.get("questions", {}).get(question_key) or extract_terms(item, question, llm)
    new_terms = [t for t in terms if t not in entry["terms"]]
    result = CollectResult(terms=terms, new_terms=new_terms,
                           skipped_terms=[t for t in terms if t in entry["terms"]], docs=docs,
                           question=question_key)

    pool: dict[int, dict] = {}                 # pageid → 후보 (여러 검색어에 걸린 문서는 한 번만)
    for term in new_terms:
        for c in wiki.search(term):
            if f"wiki-{c['pageid']}" not in entry["docs"] and c["pageid"] not in pool:
                pool[c["pageid"]] = {**c, "term": term}
    candidates = list(pool.values())
    picked = select_candidates(item, question, candidates, llm) if candidates else []

    for i, c in enumerate(candidates):
        if i not in picked:
            result.rejected.append(c["title"])
            continue
        doc = apply_length_rule(wiki.page(c["pageid"], search_term=c["term"]))
        if doc is None:
            result.too_short.append(c["title"])
        else:
            result.selected.append(c["title"])
            result.docs.append(doc)
    return result
