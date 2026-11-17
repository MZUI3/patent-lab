# 사내 문서 — 카탈로그 검사, 색인 동기화(버전·권한 변경 반영), 권한 필터 검색, 전송 정책, 출력 점검
# 실제 Chroma(임시 폴더)에 가짜 임베딩으로 색인해, where 필터가 실제로 권한 밖 청크를 막는지 본다.
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.data_prep.internal import load_catalog
from src.security import policy
from src.security.acl import User
from src.service import internal_qa
from src.service.internal_qa import ONLY_WITHHELD, build_internal_graph, search
from src.vectorstore.index_internal import indexed, sync

KIM = User("kim", "김", "researcher", ("dft",))
LEE = User("lee", "이", "researcher", ("battery",))
PARK = User("park", "박", "ip", ("ip",))
GUEST = User("guest", "방문자", "guest")

CATALOG = """
teams: {dft: 테스트팀, battery: 배터리팀, ip: 특허팀}
documents:
  - {id: INV-1, file: inv.md, title: 스캔 체인 진단 발명 신고서, kind: 발명 신고서, level: 기밀,
     allow: [team:dft, role:ip], owner: kim, version: "1.0", updated: "2026-09-01"}
  - {id: RPT-1, file: rpt.md, title: 메모리 BIST 실험 보고서, kind: 실험 보고서, level: 내부,
     allow: [team:dft], owner: kim, version: "1.0", updated: "2026-09-01"}
  - {id: PUB-1, file: pub.md, title: 스캔 체인 진단 학회 초록, kind: 공개 자료, level: 공개,
     allow: [all], owner: kim, version: "1.0", updated: "2026-06-01"}
  - {id: TBL-1, file: tbl.csv, title: 셀 열폭주 시험, kind: 실험 데이터, level: 내부,
     allow: [team:battery], owner: lee, version: "1.0", updated: "2026-09-01"}
"""
FILES = {
    "inv.md": "스캔 체인 고장 진단 발명. 상위 3개 후보 적중률 92.4%. 과제 코드 DFT-7Q3.",
    "rpt.md": "메모리 BIST March SS 불량 검출률 99.3%, March C- 96.1%.",
    "pub.md": "스캔 체인 고장 진단 학회 발표. 상위 3개 후보 적중률 81.0%.",
    "tbl.csv": "셀,가스 감지(초),열폭주(초)\nC-01,412,540\nC-02,388,539\n",
}


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "internal"
    root.mkdir()
    (root / "catalog.yaml").write_text(CATALOG, encoding="utf-8")
    for name, text in FILES.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def load(root):
    return load_catalog(root / "catalog.yaml", root)


@pytest.fixture
def gemini(monkeypatch):
    """기본 경로 = Gemini, 사내 모델 없음. 상한은 테스트마다 정한다."""
    monkeypatch.setattr(policy, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(policy, "LOCAL_LLM_BASE_URL", "")

    def set_ceiling(level):
        monkeypatch.setattr(policy, "LLM_MAX_LEVEL", level)
    set_ceiling("")
    return set_ceiling


def fake_chat(answer="답 [근거 1]"):
    prompts, routes = [], []

    def chat(route):
        routes.append(route)
        return RunnableLambda(lambda p: (prompts.append(p.to_string()), AIMessage(content=answer))[1])
    return chat, prompts, routes


# ── 카탈로그·색인 ─────────────────────────────────────────────────────────

def test_repo_catalog_is_clean():
    cat = load_catalog()
    assert len(cat.docs) == 11 and cat.rejected == [] and cat.unregistered == []


def test_catalog_rejects_broken_acl_and_reports_unregistered(corpus):
    (corpus / "catalog.yaml").write_text(CATALOG + """
  - {id: BAD-1, file: pub.md, title: 잘못된 권한, kind: x, level: 기밀, allow: [employee], owner: kim, version: "1", updated: "x"}
""", encoding="utf-8")
    (corpus / "stray.md").write_text("카탈로그에 없는 파일", encoding="utf-8")
    cat = load(corpus)
    assert [d for d, _ in cat.rejected] == ["BAD-1"]
    assert cat.unregistered == ["stray.md"]


def test_sync_is_idempotent_and_tracks_versions(corpus, chroma_store, tmp_path):
    store = chroma_store()
    rep = sync(store, load(corpus), manifest=tmp_path / "m.json", record=False)
    assert len(rep.added) == 4 and rep.chunks == 4
    assert sync(store, load(corpus), manifest=None, record=False).unchanged == ["INV-1", "RPT-1", "PUB-1", "TBL-1"]

    (corpus / "rpt.md").write_text(FILES["rpt.md"] + " 시험 시간 2.2배.", encoding="utf-8")   # 버전을 안 올림
    rep = sync(store, load(corpus), manifest=None, record=False)
    assert rep.updated == ["RPT-1 (내용 변경)"] and "버전(v1.0)이 그대로" in rep.warnings[0]


def test_permission_change_is_applied_to_index(corpus, chroma_store):
    store = chroma_store()
    sync(store, load(corpus), manifest=None, record=False)
    assert "RPT-1" in {e.doc_id for e in search(KIM, "메모리 BIST 검출률", store)}

    catalog = (corpus / "catalog.yaml").read_text(encoding="utf-8")
    (corpus / "catalog.yaml").write_text(catalog.replace("allow: [team:dft], owner", "allow: [role:ip], owner"),
                                         encoding="utf-8")
    rep = sync(store, load(corpus), manifest=None, record=False)
    assert rep.updated and "등급·권한" in rep.updated[0]
    assert "RPT-1" not in {e.doc_id for e in search(KIM, "메모리 BIST 검출률", store)}   # 권한을 거두면 바로 안 보인다
    assert "RPT-1" in {e.doc_id for e in search(PARK, "메모리 BIST 검출률", store)}


def test_removed_from_catalog_is_deleted(corpus, chroma_store):
    store = chroma_store()
    sync(store, load(corpus), manifest=None, record=False)
    (corpus / "pub.md").unlink()   # 파일이 없어지면 카탈로그 검사에서 빠지고 → 색인에서 삭제
    rep = sync(store, load(corpus), manifest=None, record=False)
    assert rep.removed == ["PUB-1"] and "PUB-1" not in indexed(store)


# ── 권한 필터 검색 ────────────────────────────────────────────────────────

@pytest.mark.parametrize("user, visible", [
    (GUEST, {"PUB-1"}),
    (KIM, {"INV-1", "RPT-1", "PUB-1"}),
    (LEE, {"TBL-1", "PUB-1"}),
    (PARK, {"INV-1", "PUB-1"}),
])
def test_search_returns_only_readable_docs(corpus, chroma_store, user, visible):
    store = chroma_store()
    sync(store, load(corpus), manifest=None, record=False)
    found = {e.doc_id for e in search(user, "스캔 체인 고장 진단 적중률 열폭주 BIST", store, k=10)}
    assert found == visible


def test_double_check_drops_rows_if_filter_is_missing(corpus, chroma_store):
    """필터를 빠뜨린 저장소(버그 상황)에서도 can_read가 권한 밖 청크를 버린다."""
    store = chroma_store()
    sync(store, load(corpus), manifest=None, record=False)

    class NoFilter:
        def similarity_search(self, q, k, filter=None):
            return store.similarity_search(q, k=k)
    assert {e.doc_id for e in search(GUEST, "스캔 체인 적중률", NoFilter(), k=10)} == {"PUB-1"}


# ── 전송 정책과 답변 ──────────────────────────────────────────────────────

def test_confidential_text_never_reaches_external_prompt(corpus, chroma_store, gemini):
    gemini("내부")
    store = chroma_store()
    sync(store, load(corpus), manifest=None, record=False)
    chat, prompts, routes = fake_chat()
    out = build_internal_graph(store, chat).invoke({"user": KIM, "question": "스캔 체인 진단 적중률은?"})
    assert "INV-1" in {e.doc_id for e in out["withheld"]}               # 열람 권한은 있지만 기밀
    assert all(e.level != "기밀" for e in out["evidence"])
    assert "92.4%" not in prompts[0] and "DFT-7Q3" not in prompts[0]   # 기밀 본문은 외부 프롬프트에 없다
    assert routes[0].name == "gemini"


def test_local_model_receives_confidential_when_configured(corpus, chroma_store, gemini, monkeypatch):
    gemini("내부")
    monkeypatch.setattr(policy, "LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    store = chroma_store()
    sync(store, load(corpus), manifest=None, record=False)
    chat, prompts, routes = fake_chat()
    out = build_internal_graph(store, chat).invoke({"user": KIM, "question": "스캔 체인 진단 적중률은?"})
    assert out["withheld"] == [] and routes[0].name == "local" and "92.4%" in prompts[0]


def test_only_withheld_evidence_means_no_llm_call(corpus, chroma_store, gemini):
    gemini("")   # Gemini 무료 등급 기본값: 공개까지
    store = chroma_store()
    sync(store, load(corpus), manifest=None, record=False)
    chat, prompts, _ = fake_chat()
    out = build_internal_graph(store, chat).invoke({"user": LEE, "question": "셀 열폭주 시험 가스 감지 시간"})
    if out["evidence"]:                      # 공개 초록이 함께 걸리면 그것만 보낸다
        assert all(e.level == "공개" for e in out["evidence"])
    else:
        assert out["answer"] == ONLY_WITHHELD and prompts == []
    assert "TBL-1" in {e.doc_id for e in out["withheld"]}


def test_history_is_filtered_by_route_ceiling():
    external = policy.Route("gemini", "m", "내부", external=True)
    local = policy.Route("local", "m", "기밀", external=False)
    history = [("user", "q1", "기밀"), ("assistant", "92.4%", "기밀"), ("user", "q2", "내부"), ("assistant", "99.3%", "내부"),
               ("assistant", "등급 표시가 없는 옛 기록")]   # 등급을 모르면 기밀로 본다
    assert internal_qa.sendable_history(history, external) == ([("user", "q2"), ("assistant", "99.3%")], 3)
    assert internal_qa.sendable_history(history, local)[1] == 0


def test_confidential_history_is_not_sent_to_external_route(corpus, chroma_store, gemini, monkeypatch):
    """사내 모델로 기밀 근거를 써서 답한 뒤, 다음 질문이 외부 경로로 가면 그 대화는 맥락으로 보내지 않는다."""
    gemini("내부")
    monkeypatch.setattr(policy, "LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    store = chroma_store()
    sync(store, load(corpus), manifest=None, record=False)
    chat, prompts, routes = fake_chat()
    history = [("user", "열폭주 발명 경보 기준은?", "기밀"), ("assistant", "80°C 위험 단계 (BAT-2K9)", "기밀"),
               ("user", "C-01 가스 감지는?", "내부"), ("assistant", "C-01은 412초", "내부")]
    out = build_internal_graph(store, chat).invoke({"user": LEE, "question": "셀 열폭주 시험 가스 감지", "history": history})
    assert routes[0].name == "gemini"                                  # 이번 근거는 내부까지 → 외부 경로
    assert "BAT-2K9" not in prompts[0] and "80°C" not in prompts[0]    # 기밀을 담은 대화는 보내지 않는다
    assert "C-01은 412초" in prompts[0] and out["history_dropped"] == 2


def test_answer_links_not_in_sources_are_removed(corpus, chroma_store, gemini):
    gemini("내부")
    store = chroma_store()
    sync(store, load(corpus), manifest=None, record=False)
    chat, _, _ = fake_chat("검출률은 99.3%입니다 [근거 1]. 자세한 내용은 http://vendor-portal.example/login")
    out = build_internal_graph(store, chat).invoke({"user": KIM, "question": "March SS 불량 검출률"})
    assert "vendor-portal" not in out["answer"] and out["removed_urls"] == ["http://vendor-portal.example/login"]
    assert [e.n for e in out["cited"]] == [1]
