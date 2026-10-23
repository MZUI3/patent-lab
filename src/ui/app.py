# src/ui/app.py — Patent Lab 화면 (F2009~F2012)
# 실행: streamlit run src/ui/app.py (프로젝트 루트에서)
#
# 중심 기능은 선행기술 조사다. 아이디어를 넣으면 가까운 선행 특허가 왼쪽에 나오고, 특허를 고르면 오른쪽 패널에서
# 구성요소 대비 · 질문하기 · 청구항 해설로 그 특허를 확인한다. 조건 탐색도 같은 패널로 이어진다.
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # streamlit은 스크립트 폴더만 path에 넣는다

import streamlit as st

from src.config import DOC_COLLECTION, META_COLLECTION, PATENTS_JSON
from src.data_prep.docs import SOURCE_LABELS
from src.data_prep.items import load_items
from src.env import MissingAPIKeyError, kipris_key, openai_kwargs, wiki_user_agent
from src.rag.prompts import DISCLAIMER, PRIOR_ART_DISCLAIMER
from src.vectorstore import snapshot
from src.vectorstore.store import count_chunks

log = logging.getLogger("patent_lab")
STATUS_BADGE = {"개시": ":red-badge[개시]", "일부": ":orange-badge[일부]", "미개시": ":gray-badge[미개시]"}
STATUS_DOT = {"개시": "🟥", "일부": "🟨", "미개시": "⬜"}
HISTORY_TURNS = 6   # 다음 질문에 맥락으로 넘길 이전 메시지 수
EXAMPLES = {
    "스캔 체인 고장 진단": "스캔 체인에 테스트 패턴을 넣고 나온 고장 로그로 학습 데이터를 만들어, 신경망 모델이 고장 난 "
                     "스캔 셀의 위치를 예측하는 진단 방법. 고장 유형별로 별도 모델을 두고, 예측 결과를 다시 테스트해 검증한다.",
    "배터리 열폭주 감지": "전기차 배터리 팩 안에 셀마다 온도 센서와 가스 센서를 두고, 온도 상승률과 가스 농도를 함께 보고 "
                     "열폭주 징후를 조기에 감지해 냉각 장치를 가동하는 시스템.",
    "메모리 BIST": "메모리 셀 어레이를 시험하는 내장 자체 시험(BIST) 회로가 마치(March) 알고리즘 테스트 패턴을 스스로 "
                 "생성하고, 불량으로 판정된 셀의 주소를 저장해 리페어 회로에 전달한다.",
}

st.set_page_config(page_title="Patent Lab", page_icon="🔬", layout="wide")


# ── 리소스 ───────────────────────────────────────────────────────────────────

@st.cache_resource
def get_items() -> dict:
    return {it.id: it for it in load_items()}


@st.cache_resource
def get_graph():
    from src.rag.graph import build_graph
    return build_graph()


def find_item(item_id: str):
    """선행기술 조사 중 KIPRIS에서 새로 받은 특허도 찾을 수 있게, 없으면 목록을 다시 읽는다."""
    items = get_items()
    if item_id not in items:
        get_items.clear()
        items = get_items()
    return items.get(item_id)


def missing_keys() -> dict[str, str]:
    out = {}
    for name, check in (("openai", openai_kwargs), ("kipris", kipris_key), ("wiki", wiki_user_agent)):
        try:
            check()
        except MissingAPIKeyError as e:
            out[name] = str(e)
    return out


def show_error(e: Exception) -> None:
    """화면에는 안내만 보여주고, 추적 정보는 서버 로그에만 남긴다 (NF2004)."""
    log.exception("요청 처리 실패")
    if isinstance(e, MissingAPIKeyError):
        st.error(f"API 키 설정이 필요합니다. {e}")
    else:
        st.error("처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")


def select(state_key: str, item_id: str) -> None:
    st.session_state[state_key] = item_id


def set_idea(text: str) -> None:
    st.session_state.idea = text


def clear_result() -> None:
    st.session_state.prior_art_result = None
    st.session_state.sel_search = None


# ── 공통 표시 ─────────────────────────────────────────────────────────────────

def render_evidence(cited: list) -> None:
    """답변에 실제로 인용된 근거만 제목·출처 유형·원문 링크와 함께 보여준다 (F2008)."""
    if not cited:
        st.info("답변에 인용된 근거가 없습니다. 문서에서 확인된 내용이 아닐 수 있습니다.")
        return
    by_type = " · ".join(f"{label} {n}" for label, n in Counter(e.label for e in cited).items())
    with st.expander(f"근거 문서 {len(cited)}개 ({by_type})", expanded=False):   # 출처 종류 구분 (F2016)
        for e in cited:
            st.markdown(f"**[근거 {e.n}]** `{e.label}` [{e.title}]({e.url})")
            excerpt = " ".join(e.text.split())   # 줄바꿈 뒤 "3. "이 번호 목록으로 바뀌지 않게 한 줄로
            st.caption(excerpt[:300] + ("…" if len(excerpt) > 300 else ""))


def render_collect(result, added: int) -> None:
    if result is None:
        return
    st.write(f"검색어: {', '.join(result.terms) or '-'}")
    if result.skipped_terms:
        st.write(f"이미 수집한 검색어: {', '.join(result.skipped_terms)}")
    if result.docs:
        st.write("새로 수집한 문서: " + ", ".join(f"{d.title} ({SOURCE_LABELS[d.source_type]})" for d in result.docs))
        st.write(f"추가된 청크 {added}개")
    else:
        st.write("새로 수집한 문서 없음")


def render_comparison(c) -> None:
    """선행 특허 1건과 아이디어의 구성요소 대비표."""
    st.progress(c.overlap, text=f"구성요소 겹침 {c.overlap:.0%}")
    if c.summary:
        st.caption(c.summary)
    # 한눈에: 구성요소마다 한 줄 (판정 배지 + 짧은 이름). 원문 구절·이유는 펼쳐서 본다.
    st.markdown("  \n".join(f"{STATUS_BADGE[r.status]} {r.name}" for r in c.rows))
    with st.expander("판정 근거 보기 (청구항 원문 구절)"):
        for r in c.rows:
            st.markdown(f"{STATUS_BADGE[r.status]} **{r.name}** — {r.element}")
            if r.quote:
                st.markdown(f"> “{r.quote}” " + " ".join(f"[근거 {e.n}]" for e in r.evidence))
            st.caption(r.note)
    if c.cited:   # 겹치는 구성요소가 없으면 "미개시" 배지로 충분하다
        render_evidence(c.cited)


def render_chat(item, keys: dict) -> None:
    """선택한 특허에 대한 질의응답 (F2007~F2009). 대화 기록은 특허별로 따로 둔다."""
    history = st.session_state.histories.setdefault(item.id, [])
    box = st.container(height=440 if history else "content", border=False)
    with box:
        if not history:
            st.caption("이 특허에 대해 물어보세요. 예) 이 특허가 해결하려는 문제는? / 청구항 1의 구성은? / 현재 등록 상태는?")
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    render_evidence(msg["cited"])

    if question := st.chat_input("이 특허에 대해 질문하세요", key=f"chat-{item.id}"):
        prev = [(h["role"], h["content"]) for h in history[-HISTORY_TURNS:]]
        history.append({"role": "user", "content": question})
        with box:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    out = {}
                    with st.status("검색어 추출 · 문서 수집 · 근거 검색 중…", expanded=False) as status:
                        state = {"item": item, "question": question, "history": prev,
                                 "collect": "kipris" not in keys and "wiki" not in keys}
                        for update in get_graph().stream(state, stream_mode="updates"):
                            for node, values in update.items():
                                out.update(values)
                                if node == "retrieve":
                                    render_collect(out.get("collect_result"), out.get("added_chunks", 0))
                                    st.write(f"근거 후보 {len(out['evidence'])}개 검색")
                                    status.update(label="답변 생성 중…")
                        status.update(label="완료", state="complete")
                    st.markdown(out["answer"])
                    render_evidence(out["cited"])
                    history.append({"role": "assistant", "content": out["answer"], "cited": out["cited"]})
                except Exception as e:
                    history.pop()   # 실패한 질문은 기록에 남기지 않는다
                    show_error(e)


def render_brief(item) -> None:
    """청구항 해설 (F2010 ②)."""
    brief = st.session_state.briefs.get(item.id)
    st.caption("청구항과 관련 개념 문서를 근거로, 이 특허를 쉬운 말로 풀어 줍니다.")
    if st.button("청구항 해설 만들기" if brief is None else "다시 만들기", key=f"brief-{item.id}"):
        from src.service.claim_brief import claim_brief
        try:
            with st.status("준비 중…", expanded=False) as status:
                brief = claim_brief(item, progress=lambda msg: status.update(label=msg))
                render_collect(brief.collect_result, brief.added_chunks)
                status.update(label="완료", state="complete")
            st.session_state.briefs[item.id] = brief
        except Exception as e:
            show_error(e)
    if brief is not None:
        st.markdown(brief.text)
        render_evidence(brief.cited)


def render_patent_panel(item_id: str, keys: dict, comparison=None) -> None:
    """오른쪽 패널: 고른 특허를 대비표 · 질문 · 청구항 해설로 확인한다."""
    item = find_item(item_id)
    if item is None:
        st.warning("이 특허의 정보를 찾지 못했습니다.")
        return
    m = item.metadata
    with st.container(border=True):
        st.markdown(f"### {m['title']}")
        st.caption(f"출원번호 {m['item_id']} · {m['applicant']} · 출원일 {m['application_date']} · {m['status']}"
                   + (f" · {m['field']}" if m.get("field") else "") + f" · [KIPRIS 원문]({m['source_url']})")
        views = (["구성요소 대비"] if comparison else []) + ["질문하기", "청구항 해설"]
        view = st.segmented_control("보기", views, default=views[0], key=f"view-{'cmp' if comparison else 'qa'}",
                                    label_visibility="collapsed") or views[0]
        if view == "구성요소 대비":
            render_comparison(comparison)
        elif view == "질문하기":
            render_chat(item, keys)
        else:
            render_brief(item)


# ── 페이지: 선행기술 조사 (F2010 ①) ─────────────────────────────────────────

def page_prior_art() -> None:
    keys = st.session_state.missing_keys
    st.title("🧭 선행기술 조사")
    st.caption("출원하려는 아이디어를 적으면, 가까운 선행 특허를 찾아 청구항과 구성요소별로 대비합니다. "
               "겹친다는 판정에는 청구항 원문 구절이 함께 나옵니다.")

    with st.form("idea_form"):
        st.text_area("발명 아이디어", key="idea", height=110, label_visibility="collapsed",
                     placeholder="무엇이 무엇을 하는지 몇 문장으로 적어 주세요. "
                                 "예) 배터리 셀마다 온도·가스 센서를 두고 열폭주 징후를 조기에 감지해 냉각 장치를 가동한다.")
        go = st.form_submit_button("선행 특허 찾기", type="primary")
    if st.session_state.get("prior_art_result") is None and not go:   # 예시는 조사 전에만 보인다
        ex = st.columns(len(EXAMPLES) + 1)
        ex[0].caption("예시 아이디어")
        for col, (label, text) in zip(ex[1:], EXAMPLES.items()):
            col.button(label, key=f"ex-{label}", on_click=set_idea, args=(text,), use_container_width=True)

    idea = st.session_state.get("idea", "").strip()
    if go and not idea:
        st.warning("아이디어를 입력해 주세요. 위의 예시 아이디어를 눌러 채울 수도 있습니다.")
    if go and idea:
        from src.service.prior_art import prior_art_search
        try:
            with st.status("준비 중…", expanded=False) as status:
                result = prior_art_search(idea, progress=lambda msg: status.update(label=msg))
                status.update(label="완료", state="complete")
            st.session_state.prior_art_result = result
            st.session_state.sel_search = result.comparisons[0].item_id if result.comparisons else None
            get_items.clear()   # KIPRIS에서 새로 받은 특허가 있으면 목록을 다시 읽는다
        except Exception as e:
            show_error(e)

    result = st.session_state.get("prior_art_result")
    selected = st.session_state.get("sel_search")
    if result is None:
        if selected:   # 결과 없이 특허만 고른 상태 (다른 화면에서 넘어온 경우)
            render_patent_panel(selected, keys)
        return

    st.caption(f"⚖️ {PRIOR_ART_DISCLAIMER}")
    if not result.elements:
        st.info("아이디어에서 구성요소를 뽑지 못했습니다. 무엇이 무엇을 하는지 조금 더 구체적으로 적어 주세요.")
        return
    searched = " · ".join(f"KIPRIS \"{s.query}\" {s.total:,}건 중 {s.received}건 확인(새로 색인 {s.added})"
                          for s in result.searches)
    st.caption(f"구성요소 {len(result.elements)}개 · 후보 {result.candidates}건 → 대비 {len(result.comparisons)}건"
               + (f" · {searched}" if searched else "")
               + (f" · ⚠️ {result.search_error}" if result.search_error else ""))
    names = getattr(result, "element_names", None) or result.elements
    head, reset = st.columns([5, 1])
    head.markdown("**구성요소** " + " · ".join(f"`{i}` {n}" for i, n in enumerate(names, 1)))
    reset.button("↺ 새 조사", on_click=clear_result, use_container_width=True,
                 help="결과를 지우고 예시 아이디어를 다시 봅니다")
    if not result.comparisons:
        st.info("가까운 선행 특허를 찾지 못했습니다. 아이디어를 다르게 표현해 보세요.")
        return

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown(f"**가까운 선행 특허 {len(result.comparisons)}건**")
        for c in result.comparisons:
            with st.container(border=True):
                st.markdown(f"**{c.title}**")
                st.caption(f"{c.applicant.split(',')[0]} · {c.application_date[:4]} · {c.status}")
                st.progress(c.overlap, text=f"겹침 {c.overlap:.0%}  {''.join(STATUS_DOT[r.status] for r in c.rows)}")
                st.button("선택됨" if c.item_id == selected else "자세히 보기", key=f"pick-{c.item_id}",
                          type="primary" if c.item_id == selected else "secondary", use_container_width=True,
                          on_click=select, args=("sel_search", c.item_id))
    with right:
        comparison = next((c for c in result.comparisons if c.item_id == selected), None)
        if selected:
            render_patent_panel(selected, keys, comparison)
        else:
            st.caption("왼쪽에서 특허를 고르면 대비표·질문·청구항 해설을 볼 수 있습니다.")


# ── 페이지: 조건 탐색 (F2011) ────────────────────────────────────────────────

def page_explore() -> None:
    keys = st.session_state.missing_keys
    st.title("🔎 특허 찾기")
    mode = st.segmented_control("찾는 방법", ["조건으로 찾기", "이름·번호로 찾기"], default="조건으로 찾기",
                                key="find_mode", label_visibility="collapsed") or "조건으로 찾기"
    if mode == "조건으로 찾기":   # F2011: 자연어 조건 → 후보 검색 → LLM 조건 선별
        st.caption("기술·연도·출원인·등록 상태 같은 조건을 자연어로 적으면, 조건을 모두 만족하는 특허만 보여줍니다.")
        with st.form("explore_form"):
            condition = st.text_input("조건", label_visibility="collapsed",
                                      placeholder="예) 2020년 이후 출원된 메모리 BIST 관련 등록 특허")
            submitted = st.form_submit_button("찾기", type="primary")
        if submitted and condition.strip():
            from src.service.explore import explore
            try:
                with st.spinner("후보 검색 및 조건 확인 중…"):
                    matches = explore(condition)
                st.session_state.explore_result = (condition, matches)
                st.session_state.sel_explore = matches[0].item_id if matches else None
            except Exception as e:
                show_error(e)
    else:   # F2009: 목록에서 질문할 특허를 바로 고른다
        st.caption("특허 명칭·출원인·출원번호의 일부를 적으면 색인된 특허 목록에서 바로 찾습니다. 예) 스캔 체인 연세대 / 1020230136220")
        query = st.text_input("이름·번호", key="find_text", label_visibility="collapsed",
                              placeholder="예) 스캔 체인 연세대")
        if query.strip() and st.session_state.get("find_text_last") != query:
            from src.service.explore import find_by_text
            matches = find_by_text(query, list(get_items().values()))
            st.session_state.explore_result = (query, matches)
            st.session_state.sel_explore = matches[0].item_id if matches else None
            st.session_state.find_text_last = query

    if "explore_result" not in st.session_state:
        return
    cond, matches = st.session_state.explore_result
    selected = st.session_state.get("sel_explore")
    if not matches:
        st.info(f"\"{cond}\" 조건에 맞는 특허를 찾지 못했습니다. 조건을 넓히거나 다르게 표현해 보세요.")
        return
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown(f"**\"{cond}\" — {len(matches)}건**")
        for mt in matches:
            with st.container(border=True):
                st.markdown(f"**{mt.title}**")
                st.caption(f"{mt.applicant.split(',')[0]} · {mt.application_date[:4]} · {mt.status}")
                if mt.reason:
                    st.write(mt.reason)
                st.button("선택됨" if mt.item_id == selected else "자세히 보기", key=f"pick-{mt.item_id}",
                          type="primary" if mt.item_id == selected else "secondary", use_container_width=True,
                          on_click=select, args=("sel_explore", mt.item_id))
    with right:
        if selected:
            render_patent_panel(selected, keys)


# ── 진입 ─────────────────────────────────────────────────────────────────────

if not PATENTS_JSON.exists():
    st.warning("특허 메타데이터가 없습니다. 터미널에서 `python -m src.data_prep.collect_meta`를 먼저 실행하세요.")
    st.stop()

meta_count, doc_count = count_chunks(META_COLLECTION), count_chunks(DOC_COLLECTION)
if meta_count == 0 and snapshot.exists():   # 다른 PC에서 처음 실행: 저장소의 스냅샷으로 색인을 만든다 (API 호출 없음)
    with st.spinner("저장소에 포함된 색인 스냅샷으로 색인을 만드는 중입니다 (처음 한 번, 약 20초)…"):
        snapshot.restore()
    st.rerun()
st.session_state.missing_keys = missing_keys()
st.session_state.setdefault("histories", {})
st.session_state.setdefault("briefs", {})

with st.sidebar:
    st.title("🔬 Patent Lab")
    st.caption("아이디어로 선행 특허를 찾고, 청구항 원문으로 확인합니다.")
    fields = Counter(it.metadata["field"] for it in get_items().values())
    c1, c2 = st.columns(2)
    c1.metric("색인된 특허", f"{meta_count:,}")
    c2.metric("문서 청크", f"{doc_count:,}")
    with st.expander(f"수집 분야 {len(fields)}개"):
        st.markdown("\n".join(f"- {name} {n:,}건" for name, n in fields.most_common()))
        st.caption("선행기술 조사 때 KIPRIS에서 관련 특허를 더 받아 '검색 수집'으로 쌓습니다.")
    st.info(DISCLAIMER, icon="⚖️")

# 진행 불가 상황 안내 (F2012)
if meta_count == 0:
    st.warning("특허 색인이 비어 있습니다. 터미널에서 `python -m src.vectorstore.index_meta`를 실행한 뒤 새로고침하세요.")
    st.stop()
if "openai" in st.session_state.missing_keys:
    st.error(f"API 키 설정이 필요합니다. {st.session_state.missing_keys['openai']}")
    st.stop()
if st.session_state.missing_keys:
    st.warning("외부 문서 수집에 필요한 설정이 없습니다. 이미 색인된 특허·문서로만 조사하고 답합니다. "
               + " ".join(st.session_state.missing_keys.values()))

st.navigation([
    st.Page(page_prior_art, title="선행기술 조사", icon="🧭", url_path="prior-art", default=True),
    st.Page(page_explore, title="특허 찾기", icon="🔎", url_path="explore"),
], position="top").run()
