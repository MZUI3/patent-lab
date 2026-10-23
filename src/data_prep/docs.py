# src/data_prep/docs.py — 외부 문서 1건의 공통 형태와 출처 유형 이름
from dataclasses import dataclass

# 출처 유형 → 화면 표시 이름 (F2007, F2016)
SOURCE_LABELS = {
    "item_meta": "특허 메타데이터",
    "patent_claims": "KIPRIS 특허공보(청구항)",
    "wikipedia": "한국어 위키백과",
}


@dataclass
class SourceDoc:
    doc_id: str        # "kipris-<출원번호>" | "wiki-<pageid>"
    title: str
    url: str           # 원문 출처
    source_type: str   # SOURCE_LABELS의 키
    text: str
    search_term: str = ""   # 이 문서를 찾은 검색어 (특허 자체 문서는 빈 값)
