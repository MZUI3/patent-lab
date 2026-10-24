# src/service/claim_brief.py — 도메인 특화 기능: 청구항 해설
# 특허 청구항은 법률 문장이라 읽기 어렵다. 청구항(KIPRIS)과 개념 문서(위키백과)를 근거로
# 한 줄 요약·해결 과제·핵심 구성·용어 풀이를 만들고, 문장마다 근거 번호를 단다.
from dataclasses import dataclass
from typing import Callable

from langchain_core.language_models import BaseChatModel

from src.config import BRIEF_TOP_K
from src.data_prep.collect_docs import CollectResult
from src.data_prep.items import Item
from src.rag.citations import cited_evidence
from src.rag.llm import get_chat
from src.rag.prompts import CLAIM_BRIEF, NOT_FOUND
from src.rag.retriever import Evidence, format_evidence, retrieve_evidence
from src.vectorstore.index_docs import ensure_documents


@dataclass
class BriefResult:
    text: str
    evidence: list[Evidence]
    cited: list[Evidence]
    collect_result: CollectResult
    added_chunks: int


def claim_brief(item: Item, llm: BaseChatModel | None = None,
                progress: Callable[[str], None] = lambda msg: None) -> BriefResult:
    llm = llm or get_chat()
    query = f"{item.metadata['title']}의 청구항 핵심 구성, 해결하려는 문제, 기술 용어"

    progress("검색어 추출 및 문서 수집 중…")
    result, added = ensure_documents(item, query, llm)
    progress("근거 검색 중…")
    evidence = retrieve_evidence(item, query, top_k=BRIEF_TOP_K)
    progress("해설 생성 중…")
    text = (CLAIM_BRIEF | llm).invoke({
        "evidence": format_evidence(evidence),
        "item_title": item.metadata["title"],
        "not_found": NOT_FOUND,
    }).content
    return BriefResult(text, evidence, cited_evidence(text, evidence), result, added)
