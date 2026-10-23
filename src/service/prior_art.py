# src/service/prior_art.py — 도메인 특화 기능: 선행기술 조사 (F2010)
# 특허를 낼 때 가장 손이 많이 가는 일은 "내 아이디어와 비슷한 특허를 찾아, 청구항의 어느 구성이 겹치는지
# 하나씩 대조하는 것"이다. 이 기능은 그 과정을 RAG로 줄인다.
#   1) 아이디어 → 구성요소 (프롬프트)          2) 메타데이터 색인(RAG 1)에서 후보 검색
#   3) 명칭·요약으로 가까운 특허 선별 (프롬프트) 4) 선별 특허의 청구항 수집 (KIPRIS, 캐시)
#   5) 특허별로 구성요소 ↔ 청구항 대비 (프롬프트, 판정마다 근거 번호)
#
# 실행: python -m src.service.prior_art "아이디어 설명"
import difflib
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.language_models import BaseChatModel

from src.config import META_COLLECTION, MMR_FETCH_K, PRIOR_ART_CANDIDATES, PRIOR_ART_MAX, PRIOR_ART_MODEL
from src.data_prep import kipris
from src.data_prep.items import load_items
from src.data_prep.patent_search import add_patents, search_kipris
from src.env import MissingAPIKeyError
from src.rag.llm import get_chat, parse_json_array, parse_json_object
from src.rag.prompts import ELEMENT_COMPARE, IDEA_ELEMENTS, PRIOR_ART_SELECT, SEARCH_KEYWORDS
from src.rag.retriever import Evidence, format_evidence
from src.vectorstore.index_docs import get_splitter
from src.vectorstore.index_meta import index_items
from src.vectorstore.store import get_store

STATUSES = ("개시", "일부", "미개시")
WEIGHT = {"개시": 1.0, "일부": 0.5, "미개시": 0.0}


@dataclass
class ElementRow:
    no: int
    element: str                # 대비에 쓴 구성요소 문장
    name: str                   # 화면에 보일 짧은 이름
    status: str                 # 개시 | 일부 | 미개시
    evidence: list[Evidence]    # 판정을 뒷받침하는 청구항 근거
    note: str
    quote: str = ""             # 근거에서 그대로 옮긴 구절 (원문에 있는지 코드로 확인함)


@dataclass
class Comparison:
    item_id: str
    title: str
    applicant: str
    application_date: str
    status: str                 # 특허의 등록 상태
    reason: str                 # 후보로 고른 이유
    summary: str
    rows: list[ElementRow]
    url: str

    @property
    def overlap(self) -> float:
        """구성요소가 겹치는 정도. 개시 1, 일부 0.5로 센다."""
        return sum(WEIGHT[r.status] for r in self.rows) / len(self.rows) if self.rows else 0.0

    @property
    def cited(self) -> list[Evidence]:
        seen = {}
        for r in self.rows:
            for e in r.evidence:
                seen.setdefault(e.n, e)
        return [seen[n] for n in sorted(seen)]


@dataclass
class KiprisSearch:
    query: str
    total: int        # KIPRIS 검색 결과 전체 건수
    received: int     # 받은 건수 (최대 KIPRIS_ROWS)
    added: int        # 처음 보는 특허라 색인에 더한 건수


@dataclass
class PriorArtResult:
    idea: str
    elements: list[str]         # 구성요소 문장 (청구항 대비용)
    candidates: int
    comparisons: list[Comparison] = field(default_factory=list)
    searches: list[KiprisSearch] = field(default_factory=list)
    search_error: str = ""     # KIPRIS 검색을 못 했으면 그 이유 (이미 색인된 특허로만 조사)
    element_names: list[str] = field(default_factory=list)   # 구성요소 짧은 이름 (화면용)


def split_elements(idea: str, llm: BaseChatModel) -> list[tuple[str, str]]:
    """[(짧은 이름, 대비용 문장)]. 이름이 없으면 문장 앞부분을 이름으로 쓴다."""
    out = []
    for e in parse_json_array((IDEA_ELEMENTS | llm).invoke({"idea": idea}).content):
        if isinstance(e, dict) and str(e.get("detail", "")).strip():
            detail = str(e["detail"]).strip()
            out.append((str(e.get("name", "")).strip() or detail[:12], detail))
        elif isinstance(e, str) and e.strip():
            out.append((e.strip()[:12], e.strip()))
    return out[:6]


def expand_index(idea: str, llm: BaseChatModel) -> tuple[list[KiprisSearch], str]:
    """아이디어로 KIPRIS 전 분야를 검색해, 처음 보는 특허를 메타데이터 색인에 더한다."""
    queries = [q.strip() for q in parse_json_array((SEARCH_KEYWORDS | llm).invoke({"idea": idea}).content)
               if isinstance(q, str) and q.strip()][:2]
    searches = []
    try:
        for q in queries:
            total, rows = search_kipris(q)
            add_patents(rows, q)   # 처음 보는 특허를 patents_ondemand.json에 기록
            # 색인 여부는 파일이 아니라 컬렉션 기준으로 본다 — 이전 실행이 기록 후 색인 전에 멈췄어도 여기서 채워진다
            known = {it.id: it for it in load_items()}
            hits = [known[r["ApplicationNumber"]] for r in rows if r["ApplicationNumber"] in known]
            stats = index_items(hits, get_store(META_COLLECTION), prune=False) if hits else {"upserted": 0}
            searches.append(KiprisSearch(q, total, len(rows), stats["upserted"]))
    except (MissingAPIKeyError, RuntimeError):   # 화면에는 이유만 짧게 (오류 원문에는 요청 URL이 섞일 수 있다)
        return searches, "KIPRIS 검색을 하지 못해 이미 색인된 특허로만 조사"
    return searches, ""


def select_candidates(idea: str, docs: list, llm: BaseChatModel) -> list[tuple]:
    listing = "\n".join(
        f"{i}. {d.metadata['title']} | 출원인: {d.metadata['applicant']} | 출원일: {d.metadata['application_date']}"
        f"\n   요약: {d.page_content.split('요약: ', 1)[-1][:250]}"
        for i, d in enumerate(docs, 1))
    raw = (PRIOR_ART_SELECT | llm).invoke({"idea": idea, "candidates": listing,
                                           "max_results": PRIOR_ART_MAX}).content
    picked, seen = [], set()
    for p in parse_json_array(raw):
        n = p.get("n") if isinstance(p, dict) else None
        if isinstance(n, int) and 1 <= n <= len(docs) and n not in seen:
            seen.add(n)
            picked.append((docs[n - 1], str(p.get("reason", ""))))
    return picked[:PRIOR_ART_MAX]


def claim_evidence(item_id: str) -> list[Evidence]:
    """선행 특허 자체 문서(요약·청구항)를 청크로 나눠 근거 번호를 붙인다."""
    doc = kipris.patent_document(item_id)
    chunks = get_splitter().split_text(doc.text)
    return [Evidence(i, f"{item_id}:{doc.doc_id}:{i - 1}", doc.source_type, doc.title, doc.url, c)
            for i, c in enumerate(chunks, 1)]


def _squash(text: str) -> str:
    """공백과 괄호 속 병기(예: "스캔 체인(scan chain)")를 지워, LLM이 병기를 빼고 옮긴 구절도 맞춰 본다."""
    return re.sub(r"\s+", "", re.sub(r"\([^)]*\)", "", text))


def quote_found(quote: str, evidence: list[Evidence]) -> bool:
    """인용 구절이 근거 원문에 실제로 있는지 본다. 공백 차이는 무시하고, 구절의 90% 이상이
    한 덩어리로 일치하면 있다고 본다 (LLM이 조사 하나를 바꿔 옮기는 정도는 허용)."""
    q = _squash(quote)
    if len(q) < 4:
        return False
    for e in evidence:
        t = _squash(e.text)
        if q in t:
            return True
        m = difflib.SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
        if m.size >= 0.9 * len(q):
            return True
    return False


def compare(elements: list[str], names: list[str], meta: dict, reason: str, evidence: list[Evidence],
            llm: BaseChatModel) -> Comparison:
    listing = "\n".join(f"{i}. {e}" for i, e in enumerate(elements, 1))
    raw = (ELEMENT_COMPARE | llm).invoke({"elements": listing, "title": meta["title"],
                                          "evidence": format_evidence(evidence)}).content
    data = parse_json_object(raw)
    by_no = {r.get("no"): r for r in data.get("elements", []) if isinstance(r, dict)}
    by_n = {e.n: e for e in evidence}

    rows = []
    for i, element in enumerate(elements, 1):
        r = by_no.get(i, {})
        status = r.get("status") if r.get("status") in STATUSES else "미개시"
        ev = [by_n[n] for n in r.get("evidence", []) if isinstance(n, int) and n in by_n]
        note = str(r.get("note", "")) or "판정 없음"
        quote = str(r.get("quote", "")).strip()
        # 겹친다는 판정은 근거 원문으로 확인될 때만 인정한다 — 거짓 "개시"가 선행조사에서 가장 위험하다
        if status != "미개시" and not ev:
            status, note, ev = "미개시", note + " (근거 번호가 없어 미개시로 처리)", []
        elif status != "미개시" and not quote_found(quote, ev):
            status, note, ev = "미개시", note + " (인용 구절을 근거 원문에서 찾지 못해 미개시로 처리)", []
        elif status == "미개시":
            ev = []   # 겹치지 않는 구성요소에는 근거 문서를 붙이지 않는다
        rows.append(ElementRow(i, element, names[i - 1], status, ev, note, quote if status != "미개시" else ""))
    return Comparison(meta["item_id"], meta["title"], meta["applicant"], meta["application_date"],
                      meta["status"], reason, str(data.get("summary", "")), rows, meta["source_url"])


def prior_art_search(idea: str, llm: BaseChatModel | None = None,
                     progress: Callable[[str], None] = lambda msg: None) -> PriorArtResult:
    # 구성요소 대비는 "인공지능 모델 ↔ 신경망"처럼 미세한 판정이라 기본 모델로는 판정이 흔들렸다 → 상위 모델
    llm = llm or get_chat(model=PRIOR_ART_MODEL)

    progress("아이디어를 구성요소로 나누는 중…")
    pairs = split_elements(idea, llm)
    names, elements = [n for n, _ in pairs], [d for _, d in pairs]
    if not elements:
        return PriorArtResult(idea, [], 0)

    progress("KIPRIS에서 관련 특허 검색 중…")
    searches, search_error = expand_index(idea, llm)

    progress("비슷한 특허 후보 검색 중…")
    docs = get_store(META_COLLECTION).max_marginal_relevance_search(
        idea, k=PRIOR_ART_CANDIDATES, fetch_k=max(MMR_FETCH_K, 2 * PRIOR_ART_CANDIDATES))
    picked = select_candidates(idea, docs, llm)

    progress(f"선행 특허 {len(picked)}건의 청구항 수집 중…")
    usable, evidence, skipped = [], [], []
    for d, reason in picked:   # KIPRIS는 순차 호출 (간격 규칙). 한 건이 실패해도 나머지로 조사를 이어간다
        try:
            evidence.append(claim_evidence(d.metadata["item_id"]))
            usable.append((d, reason))
        except (MissingAPIKeyError, RuntimeError):
            skipped.append(d.metadata["title"])
    picked = usable
    if skipped:
        search_error = (search_error + " / " if search_error else "") + \
            f"청구항을 받지 못해 대비에서 뺀 특허 {len(skipped)}건"

    progress("구성요소별 대비 중…")
    with ThreadPoolExecutor(max_workers=max(1, len(picked))) as pool:
        comparisons = list(pool.map(lambda args: compare(elements, names, args[0][0].metadata, args[0][1], args[1], llm),
                                    zip(picked, evidence)))
    comparisons.sort(key=lambda c: c.overlap, reverse=True)
    return PriorArtResult(idea, elements, len(docs), comparisons, searches, search_error, names)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    idea = " ".join(sys.argv[1:]) or input("아이디어: ")
    result = prior_art_search(idea, progress=lambda m: print(f"… {m}"))
    print("\n구성요소:")
    for i, e in enumerate(result.elements, 1):
        print(f"  {i}. {e}")
    for s in result.searches:
        print(f"KIPRIS 검색 \"{s.query}\": 전체 {s.total:,}건 중 {s.received}건 수신, 새로 색인 {s.added}건")
    if result.search_error:
        print(f"KIPRIS 검색 생략: {result.search_error}")
    print(f"\n후보 {result.candidates}건 → 대비 {len(result.comparisons)}건")
    for c in result.comparisons:
        print(f"\n■ {c.title} ({c.item_id}, {c.applicant.split(',')[0]}, {c.application_date}, {c.status}) "
              f"— 겹침 {c.overlap:.0%}\n  고른 이유: {c.reason}\n  요약: {c.summary}")
        for r in c.rows:
            print(f"   {r.no}. [{r.status}] {r.element} — {r.note} {[f'근거 {e.n}' for e in r.evidence]}"
                  + (f"\n        인용: \"{r.quote}\"" if r.quote else ""))


if __name__ == "__main__":
    main()
