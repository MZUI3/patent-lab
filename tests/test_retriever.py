# 근거 검색·병합 — 선택한 특허의 메타데이터는 항상 1번, 문서 청크는 특허 범위로만, 중복은 한 번만.
from langchain_core.documents import Document

from src.data_prep.items import Item
from src.rag.retriever import format_evidence, retrieve_evidence

ITEM = Item(id="1020230136220", text="발명의 명칭: 지터 측정 회로\n요약: 지터를 측정한다",
            metadata={"item_id": "1020230136220", "title": "지터 측정 회로", "source_type": "item_meta",
                      "source_url": "https://doi.org/10.8080/1020230136220"})


class FakeMetaStore:
    def __init__(self, found: bool = True):
        self.found = found

    def get(self, ids, include):
        if not self.found:
            return {"ids": [], "documents": [], "metadatas": []}
        return {"ids": ids, "documents": [ITEM.text], "metadatas": [ITEM.metadata]}


class FakeDocStore:
    def __init__(self):
        self.filters = []

    def similarity_search(self, question, k, filter=None):
        self.filters.append(filter)
        meta = {"item_id": ITEM.id, "doc_id": "wiki-388277", "source_type": "wikipedia",
                "doc_title": "지터", "source_url": "https://ko.wikipedia.org/wiki/지터"}
        return [
            Document(page_content="지터는 신호의 시간 오차다.", metadata={**meta, "chunk_index": 0}),
            Document(page_content="지터는  신호의 시간   오차다.", metadata={**meta, "chunk_index": 1}),   # 공백만 다른 중복
            Document(page_content="지터는 신호의 시간 오차다.", metadata={**meta, "chunk_index": 0}),      # 같은 청크 ID
            Document(page_content="위상 고정 루프는 …", metadata={**meta, "chunk_index": 2}),
        ][:k]


def test_meta_first_then_docs_with_dedupe_and_item_filter():
    docs = FakeDocStore()
    ev = retrieve_evidence(ITEM, "지터란?", top_k=4, search_type="similarity",
                           meta_store=FakeMetaStore(), doc_store=docs)
    assert docs.filters == [{"item_id": ITEM.id}]                 # 다른 특허 문서가 섞이지 않게 필터
    assert [e.n for e in ev] == [1, 2, 3]                          # 중복 2개 제거, 번호는 연속
    assert ev[0].source_type == "item_meta" and ev[0].title == "지터 측정 회로"
    assert ev[1].chunk_id == "1020230136220:wiki-388277:0" and ev[1].title == "지터"
    assert ev[2].text.startswith("위상 고정 루프")


def test_item_not_indexed_yet_still_appears_as_evidence_1():
    ev = retrieve_evidence(ITEM, "q", top_k=1, search_type="similarity",
                           meta_store=FakeMetaStore(found=False), doc_store=FakeDocStore())
    assert ev[0].chunk_id == ITEM.id and ev[0].text == ITEM.text


def test_format_evidence_labels_source_type():
    ev = retrieve_evidence(ITEM, "q", top_k=1, search_type="similarity",
                           meta_store=FakeMetaStore(), doc_store=FakeDocStore())
    block = format_evidence(ev)
    assert block.startswith("[근거 1] (특허 메타데이터) 지터 측정 회로")
    assert "[근거 2] (한국어 위키백과) 지터" in block
