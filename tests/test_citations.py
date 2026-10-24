from src.rag.citations import cited_evidence, cited_numbers, strip_citations
from src.rag.retriever import Evidence


def ev(n: int) -> Evidence:
    return Evidence(n, f"c{n}", "wikipedia", f"문서 {n}", "https://ko.wikipedia.org", "본문")


def test_cited_numbers_reads_both_forms_in_order_without_duplicates():
    assert cited_numbers("답 [근거 1][근거 3, 2] 다시 [근거 1] [근거 9]") == [1, 3, 2, 9]


def test_cited_evidence_keeps_only_existing_numbers_in_list_order():
    evidence = [ev(1), ev(2), ev(3)]
    assert [e.n for e in cited_evidence("[근거 3] 그리고 [근거 1] [근거 7]", evidence)] == [1, 3]


def test_no_citation_gives_empty_list():
    assert cited_evidence("근거 없는 답", [ev(1)]) == []


def test_strip_citations_removes_markers_and_preceding_space():
    assert strip_citations("지터는 시간 오차다 [근거 1]. 측정은 [근거 2, 3] 비교기로.") == "지터는 시간 오차다. 측정은 비교기로."
