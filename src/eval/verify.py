# src/eval/verify.py — 통합 점검: 실제 색인·파이프라인·LLM·화면(AppTest)으로 핵심 동작이 유지되는지 확인한다
# 실행: python -m src.eval.verify [점검 이름 ...]   결과: docs/eval/verification_run.md
#   이름을 주면 그 점검만 다시 실행해 기존 결과표의 해당 행만 바꾼다 (예: module-layout http-etiquette)
# 외부 호출: LLM과, 아직 수집하지 않은 특허 1건의 KIPRIS·위키 수집이 일어난다.
# 청크 크기 점검은 patent_docs를 다른 설정으로 재색인했다가 원래 설정으로 되돌린다.
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from unittest import mock

sys.stdout.reconfigure(encoding="utf-8")

import requests
import streamlit as st
from dotenv import dotenv_values
from streamlit.testing.v1 import AppTest

import src.config as config
from src.config import API_CALL_LOG, DOC_COLLECTION, ENV_PATH, META_COLLECTION, ROOT
from src.data_prep import http
from src.data_prep.collect_docs import load_log
from src.data_prep.items import build_item_text, load_items, load_patents
from src.rag.citations import cited_numbers
from src.rag.graph import build_graph
from src.rag.prompts import NOT_FOUND
from src.rag.retriever import Evidence, retrieve_evidence
from src.service.claim_brief import claim_brief
from src.service.explore import explore
from src.vectorstore import index_docs, store as store_mod
from src.vectorstore.index_docs import ensure_documents
from src.vectorstore.index_meta import index_items
from src.vectorstore.store import count_chunks, get_store

OUT = ROOT / "docs" / "eval" / "verification_run.md"
APP = str(ROOT / "src" / "ui" / "app.py")
CHECKS = []
ITEMS = {it.id: it for it in load_items()}
JITTER = "1020230136220"


def check(code: str, desc: str):
    def deco(fn):
        CHECKS.append((code, desc, fn))
        return fn
    return deco


def api_calls() -> int:
    return len(API_CALL_LOG.read_text(encoding="utf-8").splitlines()) if API_CALL_LOG.exists() else 0


# ── AppTest 도우미: 실제 LLM 대신 정해진 답을 내는 그래프 ──────────────────────

class FakeGraph:
    def __init__(self, answer: str = "", cited: tuple = (), error: Exception | None = None):
        self.answer, self.cited, self.error = answer, cited, error

    def stream(self, state, stream_mode=None):
        if self.error:
            raise self.error
        ev = [Evidence(1, f"{state['item'].id}", "item_meta", state["item"].metadata["title"], "https://x", "본문")]
        yield {"retrieve": {"collect_result": None, "added_chunks": 0, "evidence": ev}}
        yield {"generate": {"answer": self.answer, "cited": [e for e in ev if e.n in self.cited]}}


def run_app(graph: FakeGraph | None = None, open_item: str | None = JITTER) -> AppTest:
    """앱을 띄운다. open_item을 주면 결과 카드에서 그 특허를 고른 상태(오른쪽 패널 = 질문하기)로 시작한다."""
    st.cache_resource.clear()
    with mock.patch("src.rag.graph.build_graph", return_value=graph or FakeGraph("답 [근거 1]", (1,))):
        at = AppTest.from_file(APP, default_timeout=120)
        if open_item:
            at.session_state["sel_search"] = open_item
        at.run()
    at._graph = graph
    return at


def ask(at: AppTest, question: str) -> AppTest:
    with mock.patch("src.rag.graph.build_graph", return_value=at._graph or FakeGraph("답 [근거 1]", (1,))):
        at.chat_input[0].set_value(question).run()
    return at


def page_text(at: AppTest) -> str:
    parts = [e.value for kind in ("markdown", "error", "warning", "info", "caption") for e in getattr(at, kind)]
    parts += [e.value for e in at.sidebar.info]
    return "\n".join(str(p) for p in parts)


# ── 데이터와 색인 ─────────────────────────────────────────────────────────

@check("index-count", "로드한 항목 수와 색인된 청크 수가 일치한다")
def _():
    n_raw, n_items, n_chunks = len(load_patents()), len(ITEMS), count_chunks(META_COLLECTION)
    return n_items == n_chunks, f"원본 {n_raw}건 → 항목 {n_items}건, patent_meta 청크 {n_chunks}개"


@check("index-idempotent", "색인 작업을 두 번 실행해도 청크 수가 늘지 않는다")
def _():
    items = list(ITEMS.values())
    s1 = index_items(items, get_store(META_COLLECTION))
    s2 = index_items(items, get_store(META_COLLECTION))
    ok = s1["count"] == s2["count"] == len(items) and s2["upserted"] == 0
    return ok, (f"1회차: 갱신 {s1['upserted']}·변경 없음 {s1['unchanged']}, 청크 {s1['count']} / "
                f"2회차: 갱신 {s2['upserted']}·변경 없음 {s2['unchanged']}, 청크 {s2['count']}")


@check("index-text-fields", "항목 1건의 색인 텍스트에 의도한 속성이 모두 들어간다")
def _():
    p = next(p for p in load_patents() if p["RegistrationNumber"] and p["Abstract"])
    text = build_item_text(p)
    labels = ["발명의 명칭", "출원인", "출원번호", "출원일", "IPC 분류", "등록 상태", "요약"]
    missing = [l for l in labels if f"{l}: " not in text]
    first = text.split("\n요약: ")[0].replace("\n", " / ")
    return not missing, f"속성 {len(labels) - len(missing)}/{len(labels)} — {first} / 요약: …" + \
        (f" (누락 {missing})" if missing else "")


@check("index-vs-bruteforce", "색인 검색 결과가 전수 계산(brute force)과 일치한다 (인덱스 손상 점검)")
def _():
    import numpy as np
    from src.vectorstore.store import get_embeddings
    s = get_store(META_COLLECTION)
    query = "스캔 체인 고장 진단 기계 학습"
    q = np.array(get_embeddings().embed_query(query))
    g = s.get(include=["embeddings"])
    E = np.array(g["embeddings"])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    brute = [g["ids"][i] for i in np.argsort(1 - E @ (q / np.linalg.norm(q)))[:10]]
    hnsw = s._collection.query(query_embeddings=[q.tolist()], n_results=10, include=[])["ids"][0]
    same = len(set(brute) & set(hnsw))
    return same >= 9, f"\"{query}\" 상위 10건: 색인 검색과 전수 계산 공통 {same}/10 (전체 {len(g['ids']):,}건)"


@check("snapshot-restore", "저장소의 색인 스냅샷만으로 다른 PC에서 같은 색인을 만들 수 있다")
def _():
    import tempfile
    from src.vectorstore import snapshot
    from src.vectorstore.store import get_embeddings
    if not snapshot.exists():
        return None, "data/index/에 스냅샷이 없습니다 (python -m src.vectorstore.snapshot export)"
    q = get_embeddings().embed_query("배터리 열폭주 감지 가스 센서")
    live = get_store(META_COLLECTION)._collection.query(query_embeddings=[q], n_results=10, include=[])["ids"][0]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:   # Windows에서 Chroma가 파일을 잡고 있을 수 있다
        t = time.time()
        with mock.patch.object(store_mod, "CHROMA_DIR", Path(tmp)):
            counts = snapshot.restore()
            import chromadb
            restored = chromadb.PersistentClient(path=store_mod.chroma_path()).get_collection(META_COLLECTION)
            got = restored.query(query_embeddings=[q], n_results=10, include=[])["ids"][0]
            del restored
        elapsed = time.time() - t
        import chromadb.api.client as cl
        cl.SharedSystemClient.clear_system_cache()   # 임시 폴더를 지울 수 있게 연결을 닫는다
    ok = counts.get(META_COLLECTION) == count_chunks(META_COLLECTION) and got == live
    return ok, (f"빈 폴더에 복원 {elapsed:.0f}초 (API 호출 없음): {counts} / "
                f"복원본 검색 상위 10건이 원본과 {'순서까지 같음' if got == live else f'{len(set(got) & set(live))}/10 일치'}")


# ── 문서 수집과 색인 ──────────────────────────────────────────────────────

NEW_CASE = {}


@check("collect-terms", "하나의 질문에 대해 프롬프트1이 뽑은 검색어 목록을 확인한다")
def _():
    log = load_log()
    # 실행할 때마다 아직 수집하지 않은 메모리 테스트 특허 1건을 새로 고른다
    item = next(it for it in ITEMS.values() if it.id not in log and "메모리" in it.metadata["title"])
    question = "이 특허가 자동으로 생성하는 회로는 무엇이고, 메모리 자체 시험은 왜 필요한가?"
    before = api_calls()
    result, added = ensure_documents(item, question)
    NEW_CASE.update(item=item, question=question, result=result, added=added, calls=api_calls() - before)
    return bool(result.terms), (f"[{item.id}] {item.metadata['title']} / 질문 \"{question}\" → "
                                f"검색어 {result.terms}")


@check("collect-select", "후보 중 프롬프트2로 색인된 문서와 버려진 문서를 구분해 확인한다")
def _():
    r = NEW_CASE["result"]
    return None if not (r.selected or r.rejected) else True, \
        (f"색인: {r.selected or '없음'} / 버림: {r.rejected or '없음'}" +
         (f" / 짧아서 제외: {r.too_short}" if r.too_short else "") +
         f" / 새 문서 {len(r.docs)}건·청크 {NEW_CASE['added']}개, 외부 호출 {NEW_CASE['calls']}회")


@check("collect-no-repeat", "같은 질문을 다시 물으면 수집이 다시 일어나지 않는다")
def _():
    before = api_calls()
    result, added = ensure_documents(NEW_CASE["item"], NEW_CASE["question"])
    calls = api_calls() - before
    ok = calls == 0 and not result.docs and added == 0
    return ok, (f"재질문: 검색어 {result.terms}, 건너뜀 {result.skipped_terms}, 새 문서 {len(result.docs)}건, "
                f"추가 청크 {added}개, 외부 호출 {calls}회")


@check("collect-own-document", "선택한 항목 자체를 다룬 문서(특허 공보)가 빠지지 않았다")
def _():
    s = get_store(DOC_COLLECTION)
    rows = []
    for item_id, entry in load_log().items():
        doc_id = f"kipris-{item_id}"
        n = len(s.get(where={"$and": [{"item_id": item_id}, {"doc_id": doc_id}]}, include=[])["ids"])
        rows.append((item_id, doc_id in entry["docs"], n))
    bad = [r for r in rows if not r[1] or r[2] == 0]
    return not bad, f"수집한 특허 {len(rows)}건 모두 공보 청크 보유: " + \
        ", ".join(f"{i}({n}청크)" for i, _, n in rows) + (f" / 누락 {bad}" if bad else "")


@check("chunk-size-effect", "청크 크기·중복 길이를 설정에서 바꿔 색인하면 청크 수가 달라진다")
def _():
    original = (config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    counts = {}
    try:
        for size, overlap in [(500, 100), original]:
            index_docs.CHUNK_SIZE, index_docs.CHUNK_OVERLAP = size, overlap
            counts[(size, overlap)] = sum(index_docs.rebuild().values())
    finally:
        index_docs.CHUNK_SIZE, index_docs.CHUNK_OVERLAP = original
    a, b = counts[(500, 100)], counts[original]
    return a != b and count_chunks(DOC_COLLECTION) == b, \
        f"500/100 → {a}청크, {original[0]}/{original[1]} → {b}청크 (원래 설정으로 복원, 현재 {count_chunks(DOC_COLLECTION)}청크)"


# ── 답변과 근거 ───────────────────────────────────────────────────────────

ANSWER = {}


def app_graph():
    """항목을 하나만 실행해도 되도록, 그래프와 대표 답변은 처음 필요할 때 만든다."""
    if "graph" not in ANSWER:
        ANSWER["graph"] = build_graph()
    return ANSWER["graph"]


@check("topk-effect", "Top-k를 바꾸면 같은 질문의 근거 목록이 달라진다")
def _():
    q = "적분 회로와 비교기는 이 회로에서 어떤 역할을 하나?"
    lists = {k: retrieve_evidence(ITEMS[JITTER], q, top_k=k) for k in (3, 10)}
    ids = {k: [e.chunk_id.split(":", 1)[-1] for e in v] for k, v in lists.items()}
    return len(lists[3]) != len(lists[10]), \
        f"k=3 → 근거 {len(lists[3])}개 {ids[3]} / k=10 → 근거 {len(lists[10])}개 (추가 {len(lists[10]) - len(lists[3])}개)"


@check("graph-nodes", "LangGraph가 retrieve와 generate 두 노드로 구성된다")
def _():
    graph = app_graph()
    nodes = set(graph.get_graph().nodes)
    edges = sorted((e.source, e.target) for e in graph.get_graph().edges)
    ok = nodes == {"__start__", "retrieve", "generate", "__end__"}
    return ok, f"노드 {sorted(nodes)}, 간선 {edges}"


@check("citations-match", "답변 본문의 근거 번호와 표시되는 근거 목록이 일치한다 (응답 시간 포함)")
def _():
    t = time.time()
    out = app_graph().invoke({"item": ITEMS[JITTER], "question": "이 특허가 지터를 측정하는 원리는 무엇인가?"})
    elapsed = time.time() - t
    nums = set(cited_numbers(out["answer"]))
    ev_nums = {e.n for e in out["evidence"]}
    shown = {e.n for e in out["cited"]}
    ANSWER["normal"] = out
    ok = bool(nums) and nums <= ev_nums and shown == nums
    return ok, f"본문 번호 {sorted(nums)} / 표시 근거 {sorted(shown)} / 검색된 근거 {len(ev_nums)}개 / 처리 {elapsed:.1f}초"


@check("source-types", "근거 목록에 항목 메타데이터와 외부 문서가 구분되어 표시된다")
def _():
    evidence = ANSWER["normal"]["evidence"] if "normal" in ANSWER else         retrieve_evidence(ITEMS[JITTER], "이 특허가 지터를 측정하는 원리는 무엇인가?")
    labels = [e.label for e in evidence]
    kinds = sorted(set(labels))
    return "특허 메타데이터" in kinds and len(kinds) >= 2, f"근거 출처 유형: {kinds}"


@check("not-found-answer", "문서에 없는 것을 물으면 추측하지 않고 확인되지 않는다고 답한다")
def _():
    out = app_graph().invoke({"item": ITEMS[JITTER], "collect": False,
                                  "question": "이 회로의 양산 단가와 삼성전자 제품에 적용된 시점은?"})
    ANSWER["missing"] = out
    return NOT_FOUND in out["answer"], f"답변: \"{out['answer'][:80]}\" / 인용 {len(out['cited'])}개"


@check("no-citation-notice", "답변에 인용된 근거가 없으면 화면에 안내된다")
def _():
    at = ask(run_app(FakeGraph(f"{NOT_FOUND}.", ())), "단가는?")
    text = page_text(at)
    real = f" (실제 파이프라인 not-found-answer 답변도 인용 {len(ANSWER['missing']['cited'])}개)" if "missing" in ANSWER else ""
    return "답변에 인용된 근거가 없습니다" in text, \
        f"인용 0개 답변 → 화면 info: \"{next((i.value for i in at.info if '근거' in i.value), '없음')[:40]}…\"{real}"


# ── 서비스 기능과 화면 ────────────────────────────────────────────────────

@check("chat-isolation", "항목을 바꿨을 때 이전 대화가 섞이지 않는다")
def _():
    other = "1020210190872"
    at = ask(run_app(), "첫 특허 질문")
    n_first = len(at.chat_message)
    at.session_state["sel_search"] = other   # 다른 결과 카드를 고른 것과 같다
    at.run()
    n_other = len(at.chat_message)
    at.session_state["sel_search"] = JITTER
    at.run()
    n_back = len(at.chat_message)
    return n_first == 2 and n_other == 0 and n_back == 2, \
        f"지터 특허 질문 후 메시지 {n_first}개 → 다른 특허로 전환 {n_other}개 → 다시 지터 특허 {n_back}개"


@check("explore-conditions", "조건 기반 탐색 결과에 조건과 맞지 않는 항목이 섞이지 않는다")
def _():
    cases = [
        ("2015년 이후 출원된 스캔 체인 고장 진단 관련 등록 특허",
         lambda m: int(m.application_date[:4]) >= 2015 and m.status == "등록"),
        ("2005년 이전에 출원된 메모리 자체 시험(BIST) 관련 특허", lambda m: int(m.application_date[:4]) <= 2005),
        ("삼성전자가 출원한 테스트 패턴 생성 관련 특허", lambda m: "삼성전자" in m.applicant),
    ]
    lines, ok = [], True
    for cond, rule in cases:
        matches = explore(cond)
        bad = [m.title for m in matches if not rule(m)]
        ok &= not bad
        lines.append(f"\"{cond}\" → {len(matches)}건, 구조 조건 위반 {len(bad)}건 "
                     f"({'; '.join(f'{m.application_date[:4]} {m.status} {m.applicant.split(",")[0]} {m.title[:20]}' for m in matches[:4])})")
    return ok, " / ".join(lines)


@check("brief-evidence", "도메인 특화 기능(청구항 해설) 결과에도 근거가 함께 제시된다")
def _():
    b = claim_brief(ITEMS[JITTER])
    lines = [l for l in b.text.splitlines() if l.strip() and not l.startswith("####")]
    with_cite = sum("[근거" in l for l in lines)
    return with_cite == len(lines) and bool(b.cited), \
        f"해설 {len(lines)}줄 중 근거 번호 {with_cite}줄, 표시 근거 {[f'{e.n}:{e.label}' for e in b.cited]}"


@check("prior-art-quotes", "도메인 특화 기능(선행기술 조사)의 겹침 판정이 모두 청구항 원문으로 뒷받침된다")
def _():
    from src.service.prior_art import prior_art_search, quote_found
    idea = ("스캔 체인에 테스트 패턴을 넣고 나온 고장 로그로 학습 데이터를 만들어, 신경망 모델이 고장 난 스캔 셀의 "
            "위치를 예측하는 진단 방법. 고장 유형별로 별도 모델을 두고, 예측 결과를 다시 테스트해 검증한다.")
    t = time.time()
    r = prior_art_search(idea)
    elapsed = time.time() - t
    claimed = [(c, row) for c in r.comparisons for row in c.rows if row.status != "미개시"]
    unsupported = [row.element for c, row in claimed if not (row.evidence and quote_found(row.quote, row.evidence))]
    top = r.comparisons[0] if r.comparisons else None
    ok = bool(r.elements) and bool(r.comparisons) and not unsupported
    return ok, (f"구성요소 {len(r.elements)}개, 후보 {r.candidates}건 → 대비 {len(r.comparisons)}건, {elapsed:.0f}초 / "
                f"1위: {top.title if top else '-'} (겹침 {top.overlap:.0%}) / "
                f"개시·일부 판정 {len(claimed)}건 중 원문 인용 미확인 {len(unsupported)}건")


@check("missing-index-and-key", "색인이 없는 상태와 API 키가 없는 상태에서 안내가 표시된다")
def _():
    empty = ROOT / "data" / "_verify_empty_chroma"
    # 스냅샷이 있으면 앱이 스스로 색인을 복원하므로, "색인도 스냅샷도 없는" 상태를 만든다
    with mock.patch.object(store_mod, "CHROMA_DIR", empty), mock.patch("src.vectorstore.snapshot.exists", return_value=False):
        no_index = run_app(open_item=None)
    saved = {k: os.environ.pop(k) for k in ("GMS_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY") if k in os.environ}
    try:
        no_key = run_app(open_item=None)
    finally:
        os.environ.update(saved)
    w = [x.value for x in no_index.warning]
    e = [x.value for x in no_key.error]
    ok = any("색인이 비어" in x for x in w) and any("API 키" in x for x in e) \
        and not no_index.chat_input and not no_key.chat_input and not no_index.exception and not no_key.exception
    return ok, f"색인 없음 → \"{w[0][:45]}…\" / 키 없음 → \"{e[0][:45]}…\" / 두 경우 모두 입력창 숨김·예외 없음"


@check("disclaimer", "판단이 위험한 도메인의 면책 안내가 표시된다")
def _():
    at = run_app()
    info = [x.value for x in at.sidebar.info]
    return any("법률 자문" in x for x in info), f"사이드바: \"{info[0][:60]}…\""


# ── 운영 품질: 키·오류 노출·모듈 구조·외부 호출 ───────────────────────────────────────────────────────

@check("no-keys-in-repo", "소스코드와 저장소 이력에 API 키가 남아 있지 않다")
def _():
    env = {k: v for k, v in dotenv_values(ENV_PATH).items() if v and k in ("GMS_KEY", "KIPRIS_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")}
    found = []
    for k, v in env.items():
        tracked = subprocess.run(["git", "grep", "-l", "-F", v], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        history = subprocess.run(["git", "log", "--all", "--oneline", f"-S{v}"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
        if tracked or history:
            found.append(k)
    # 수집 데이터·스냅샷(data/)은 다른 PC용으로 일부러 저장소에 둔다 — 위의 키 값 검색이 그 파일들까지 훑는다.
    # 여기서는 .env와 Chroma 저장소가 추적되지 않는지만 본다.
    tracked = subprocess.run(["git", "ls-files", ".env", "chroma_db"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    n_data = len(subprocess.run(["git", "ls-files", "data"], cwd=ROOT, capture_output=True, text=True).stdout.split())
    return not found and not tracked, \
        (f"키 {len(env)}종 값 검색: 추적 파일(수집 데이터 {n_data}개 포함)·전체 커밋 이력 "
         f"{'0건' if not found else '발견 ' + str(found)} / git 추적 중인 .env·chroma_db {len(tracked)}개")


@check("no-internal-leak", "화면에 API 키·프롬프트 전문·오류 추적 정보가 노출되지 않는다")
def _():
    secret = "sk-VERIFY-SECRET-123"
    err = RuntimeError(f"Traceback (most recent call last): url?ServiceKey={secret}")
    at = ask(run_app(FakeGraph(error=err)), "오류를 일으키는 질문")
    text = page_text(at) + "\n".join(x.value for x in at.error)
    leaked = [w for w in (secret, "Traceback", "ServiceKey", "규칙\n- [근거]") if w in text]
    shown = [x.value for x in at.error]
    return not leaked and not at.exception, f"내부 오류 발생 시 화면: {shown} / 노출된 문자열 {leaked or '없음'}"


@check("module-layout", "수집·색인·검색·생성·UI가 모듈로 분리되어 있다")
def _():
    src = ROOT / "src"
    hits = {}
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"plus\.kipris\.or\.kr|wikipedia\.org/w/api", text):
            hits[str(path.relative_to(src))] = True
    ui_imports = [str(p.relative_to(src)) for p in src.rglob("*.py")
                  if "ui" not in p.parts and "eval" not in p.parts and "from src.ui" in p.read_text(encoding="utf-8")]
    outside = [h for h in hits if not h.startswith("data_prep")]
    return not outside and not ui_imports, \
        f"외부 API 주소가 있는 파일: {sorted(hits)} (모두 data_prep) / UI를 가져다 쓰는 하위 모듈 {ui_imports or '없음'} / " \
        f"폴더: {sorted(p.name for p in src.iterdir() if p.is_dir() and not p.name.startswith('_'))}"


@check("http-etiquette", "외부 호출이 재시도·호출 간격 규칙을 지키고 식별 정보를 보낸다")
def _():
    calls = []

    def flaky(url, params=None, headers=None, timeout=None):
        calls.append((time.time(), headers))
        if len(calls) < 3:
            raise requests.ConnectionError(f"fail {url}?key=SECRET")
        r = requests.Response()
        r.status_code = 200
        return r

    log = ROOT / "data" / "_verify_calls.log"
    with mock.patch.object(http.requests, "get", side_effect=flaky), mock.patch.object(http, "API_CALL_LOG", log):
        t = time.time()
        http.get("https://example.test", params={}, tag="verify", detail="retry", interval=0.5, secret="SECRET")
        total = time.time() - t
        with mock.patch.object(http.requests, "get", side_effect=requests.ConnectionError("x?key=SECRET")):
            try:
                http.get("https://example.test", params={}, tag="verify", detail="fail", interval=0.5, secret="SECRET")
                masked = "예외 없음"
            except RuntimeError as e:
                masked = str(e)
    logged = len(log.read_text(encoding="utf-8").splitlines())
    log.unlink()
    from src.env import wiki_user_agent
    ua = wiki_user_agent()
    ok = len(calls) == 3 and logged == 6 and total >= 1 + 2 + 0.5 and "SECRET" not in masked and os.environ["WIKI_CONTACT"] in ua
    return ok, (f"2회 실패 후 3회째 성공: 시도 3회, 대기 포함 {total:.1f}초(재시도 1·2초 + 간격 0.5초), 시도 기록 {logged}줄 / "
                f"3회 모두 실패 시 메시지 \"{masked[:50]}…\"(키 가림) / 위키 User-Agent에 연락처 포함")


def main() -> None:
    only = set(sys.argv[1:])
    rows = []
    for code, desc, fn in CHECKS:
        if only and code not in only:
            continue
        t = time.time()
        try:
            ok, evidence = fn()
        except Exception as e:
            ok, evidence = False, f"실행 오류: {e.__class__.__name__}: {e}"
            traceback.print_exc()
        status = {True: "통과", False: "실패", None: "미확인"}[ok]
        rows.append((code, desc, status, evidence))
        print(f"[{status}] {code} {desc} ({time.time() - t:.1f}s)\n    {evidence}")

    lines = ["# 통합 점검 결과\n", f"- 실행: {time.strftime('%Y-%m-%d %H:%M')} / `python -m src.eval.verify`\n",
             "| 점검 | 확인 내용 | 결과 | 근거 (실행 결과) |", "|---|---|---|---|"]
    new = {c: f"| {c} | {d} | {s} | {e.replace('|', '/')} |" for c, d, s, e in rows}
    if only and OUT.exists():   # 일부만 다시 실행했으면 기존 표의 해당 행만 바꾼다
        old = [l for l in OUT.read_text(encoding="utf-8").splitlines() if l.startswith("| ") and not l.startswith("| 점검")]
        rows_by_code = {l.split(" | ")[0][2:]: l for l in old} | new
        order = [c for c, _, _ in CHECKS]
        lines += sorted(rows_by_code.values(), key=lambda l: order.index(l.split(" | ")[0][2:])
                        if l.split(" | ")[0][2:] in order else len(order))
    else:
        lines += list(new.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n통과 {sum(r[2] == '통과' for r in rows)} / 실패 {sum(r[2] == '실패' for r in rows)} / "
          f"미확인 {sum(r[2] == '미확인' for r in rows)} → {OUT}")


if __name__ == "__main__":
    main()
