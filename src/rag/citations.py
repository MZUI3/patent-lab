# src/rag/citations.py — 답변 본문의 근거 번호를 찾아 실제로 인용된 근거만 남긴다 (F2008)
import re

from src.rag.retriever import Evidence

CITATION = re.compile(r"\[근거\s*([\d\s,]+)\]")   # [근거 3], [근거 1, 3] 모두 읽는다


def cited_numbers(answer: str) -> list[int]:
    nums = []
    for group in CITATION.findall(answer):
        for n in re.findall(r"\d+", group):
            if int(n) not in nums:
                nums.append(int(n))
    return nums


def cited_evidence(answer: str, evidence: list[Evidence]) -> list[Evidence]:
    """답변에 등장한 순서가 아니라 근거 번호 순으로, 실제로 있는 번호만 돌려준다."""
    nums = set(cited_numbers(answer))
    return [e for e in evidence if e.n in nums]


def strip_citations(text: str) -> str:
    """이전 대화를 다음 질문의 맥락으로 넘길 때 근거 번호를 지운다. 질문마다 번호가 다시 매겨지기 때문이다."""
    return re.sub(r"\s*" + CITATION.pattern, "", text)
