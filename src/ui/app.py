# src/ui/app.py — Patent Lab 화면
# 실행: streamlit run src/ui/app.py (프로젝트 루트에서)
#
# 로그인한 사용자의 권한 안에서 동작한다. 화면은 넷이다:
#   선행기술 조사(기본) · 특허 찾기 — 공개 특허(KIPRIS)가 대상. 아이디어를 넣으면 가까운 선행 특허가 왼쪽에 나오고,
#     특허를 고르면 오른쪽 패널에서 구성요소 대비 · 질문하기 · 청구항 해설로 확인한다
#   사내 문서 — 열람 권한이 있는 사내 문서만 검색하고, 문서 등급에 따라 보낼 LLM 경로를 고른다
#   운영 현황 — 시스템 관리자만: 감사 로그 집계, 사내 문서 색인 현황, 전송 정책
# LLM을 부르는 요청은 모두 감사 로그에 남기고(입력은 원문 대신 해시), 사용자별 일일 한도를 넘으면 막는다.
import logging
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # streamlit은 스크립트 폴더만 path에 넣는다

import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback

from src.config import APP_ENV, DOC_COLLECTION, EMBEDDING_MODEL, META_COLLECTION, PATENTS_JSON, PRIOR_ART_MODEL
from src.data_prep.docs import SOURCE_LABELS
from src.data_prep.internal import load_catalog
from src.data_prep.items import load_items
from src.env import MissingAPIKeyError, kipris_key, openai_kwargs, wiki_user_agent
from src.rag.prompts import DISCLAIMER, PRIOR_ART_DISCLAIMER
from src.security import acl, audit, policy
from src.security.auth import AccountsError, authenticate, load_accounts, startup_problem
from src.security.policy import SendBlocked
from src.vectorstore import snapshot
from src.vectorstore.store import count_chunks

log = logging.getLogger("patent_lab")
STATUS_BADGE = {"개시": ":red-badge[개시]", "일부": ":orange-badge[일부]", "미개시": ":gray-badge[미개시]"}
STATUS_DOT = {"개시": "🟥", "일부": "🟨", "미개시": "⬜"}
LEVEL_BADGE = {"공개": ":green-badge[공개]", "내부": ":blue-badge[내부]", "기밀": ":red-badge[기밀]"}
HISTORY_TURNS = 6   # 다음 질문에 맥락으로 넘길 이전 메시지 수
EXAMPLES = {
    "스캔 체인 고장 진단": "스캔 체인에 테스트 패턴을 넣고 나온 고장 로그로 학습 데이터를 만들어, 신경망 모델이 고장 난 "
                     "스캔 셀의 위치를 예측하는 진단 방법. 고장 유형별로 별도 모델을 두고, 예측 결과를 다시 테스트해 검증한다.",
    "배터리 열폭주 감지": "전기차 배터리 팩 안에 셀마다 온도 센서와 가스 센서를 두고, 온도 상승률과 가스 농도를 함께 보고 "
                     "열폭주 징후를 조기에 감지해 냉각 장치를 가동하는 시스템.",
    "메모리 BIST": "메모리 셀 어레이를 시험하는 내장 자체 시험(BIST) 회로가 마치(March) 알고리즘 테스트 패턴을 스스로 "
                 "생성하고, 불량으로 판정된 셀의 주소를 저장해 리페어 회로에 전달한다.",
}
# 아이디어의 공개 여부 → 민감도 등급. 미공개 발명이 기본값이다 (모르면 막는 쪽)
IDEA_LEVELS = {"미공개 발명 (기밀)": "기밀", "공개된 내용 (이미 출원·발표함)": "공개"}
PUBLIC_IDEA = "공개된 내용 (이미 출원·발표함)"
INTERNAL_EXAMPLES = ("March SS와 March C-의 불량 검출률 차이는?", "공급사 A 고온용 프로브 카드의 납기는?",
                     "미공개 발명을 외부 AI에 넣어도 되나요?", "가스 감지 선행 시간이 가장 짧은 셀은?")

st.set_page_config(page_title="Patent Lab", page_icon="🔬", layout="wide")


# ── 리소스 ───────────────────────────────────────────────────────────────────

@st.cache_resource
def get_items() -> dict:
    return {it.id: it for it in load_items()}


@st.cache_resource
def get_graph():
    from src.rag.graph import build_graph
    return build_graph()


@st.cache_resource
def get_internal_graph():
    from src.service.internal_qa import build_internal_graph
    return build_internal_graph()


def answer_level(evidence: list) -> str:
    from src.service.internal_qa import answer_level as level_of
    return level_of(evidence)


@st.cache_resource
def get_accounts():
    return load_accounts()


@st.cache_resource
def get_catalog():
    return load_catalog()


@st.cache_resource(show_spinner="사내 문서 색인을 카탈로그와 맞추는 중… (처음 한 번은 임베딩 때문에 1분 안팎)")
def ensure_internal():
    """프로세스마다 한 번: 카탈로그와 색인을 맞춘다. 바뀐 문서(내용·등급·권한)만 다시 색인한다."""
    from src.vectorstore import index_internal
    return index_internal.sync()


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
    """화면에는 안내만 보여주고, 추적 정보는 서버 로그에만 남긴다."""
    log.exception("요청 처리 실패")
    if isinstance(e, MissingAPIKeyError):
        st.error(f"API 키 설정이 필요합니다. {e}")
    else:
        st.error("처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")


def select(state_key: str, item_id: str) -> None:
    st.session_state[state_key] = item_id


def set_idea(text: str) -> None:
    st.session_state.idea = text
    st.session_state.idea_level = PUBLIC_IDEA   # 예시는 공개된 내용이다


def clear_result() -> None:
    st.session_state.prior_art_result = None
    st.session_state.sel_search = None


def logout() -> None:
    audit.record("logout", st.session_state.get("user"))
    st.session_state.clear()   # 다음 사용자에게 이전 대화·조사 결과가 남지 않게 세션을 통째로 비운다


# ── 감사 로그와 사용 한도 ─────────────────────────────────────────────────────

def over_limit(event: str) -> bool:
    user = st.session_state.user
    if audit.count_today(user.username) < user.daily_limit:
        return False
    audit.record(event, user, outcome="limit")
    st.warning(f"오늘 LLM 요청 한도({user.daily_limit}회)를 다 썼습니다. 내일 다시 쓰거나 관리자에게 한도 조정을 요청하세요.")
    return True


@contextmanager
def audited(event: str, **fields):
    """LLM을 부르는 요청 하나를 감사 로그 한 줄로 남긴다. 블록 안에서 fields를 채우면(경로·보낸 문서 등) 함께 기록된다."""
    user, t0 = st.session_state.user, time.perf_counter()
    with get_openai_callback() as cb:
        try:
            yield fields
        except SendBlocked as e:
            audit.record(event, user, outcome="blocked", reason=f"'{e.level}' 등급을 받을 LLM 경로 없음", **fields)
            raise
        except Exception as e:
            audit.record(event, user, outcome="error", error=type(e).__name__,
                         seconds=round(time.perf_counter() - t0, 1), tokens=cb.total_tokens, **fields)
            raise
    audit.record(event, user, outcome="ok", seconds=round(time.perf_counter() - t0, 1), tokens=cb.total_tokens, **fields)


def public_fields(text: str) -> dict:
    """공개 특허 기능의 감사 로그 기본값: 보낸 자료는 공개 등급, 경로는 기본 경로."""
    return {"level": "공개", "route": policy.primary_route().label, **audit.digest(text)}


# ── 공통 표시 ─────────────────────────────────────────────────────────────────

def render_evidence(cited: list) -> None:
    """답변에 실제로 인용된 근거만 제목·출처 유형·원문 링크와 함께 보여준다. 사내 문서는 문서 ID·버전·등급."""
    if not cited:
        st.info("답변에 인용된 근거가 없습니다. 문서에서 확인된 내용이 아닐 수 있습니다.")
        return
    by_type = " · ".join(f"{label} {n}" for label, n in Counter(e.label for e in cited).items())
    with st.expander(f"근거 문서 {len(cited)}개 ({by_type})", expanded=False):   # 출처 종류 구분
        for e in cited:
            if e.url:
                st.markdown(f"**[근거 {e.n}]** `{e.label}` [{e.title}]({e.url})")
            else:
                st.markdown(f"**[근거 {e.n}]** `{e.label}` {LEVEL_BADGE.get(e.level, e.level)} {e.title} · "
                            f"`{e.doc_id}` v{e.version}" + (" ⚠️" if e.flagged else ""))
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
    """선택한 특허에 대한 질의응답. 대화 기록은 특허별로 따로 둔다."""
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
                if over_limit("qa"):
                    history.pop()
                    return
                try:
                    out = {}
                    with audited("qa", item=item.id, **public_fields(question)) as info, \
                            st.status("검색어 추출 · 문서 수집 · 근거 검색 중…", expanded=False) as status:
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
                        info["sent"] = sorted({e.chunk_id.split(":")[0] for e in out["evidence"]})
                    st.markdown(out["answer"])
                    render_evidence(out["cited"])
                    history.append({"role": "assistant", "content": out["answer"], "cited": out["cited"]})
                except Exception as e:
                    history.pop()   # 실패한 질문은 기록에 남기지 않는다
                    show_error(e)


def render_brief(item) -> None:
    """청구항 해설."""
    brief = st.session_state.briefs.get(item.id)
    st.caption("청구항과 관련 개념 문서를 근거로, 이 특허를 쉬운 말로 풀어 줍니다.")
    if st.button("청구항 해설 만들기" if brief is None else "다시 만들기", key=f"brief-{item.id}") \
            and not over_limit("brief"):
        from src.service.claim_brief import claim_brief
        try:
            with audited("brief", item=item.id, **public_fields(item.id)), \
                    st.status("준비 중…", expanded=False) as status:
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


# ── 페이지: 선행기술 조사 ─────────────────────────────────────────

def page_prior_art() -> None:
    keys = st.session_state.missing_keys
    st.title("🧭 선행기술 조사")
    st.caption("출원하려는 아이디어를 적으면, 가까운 선행 특허를 찾아 청구항과 구성요소별로 대비합니다. "
               "겹친다는 판정에는 청구항 원문 구절이 함께 나옵니다.")

    with st.form("idea_form"):
        st.text_area("발명 아이디어", key="idea", height=110, label_visibility="collapsed",
                     placeholder="무엇이 무엇을 하는지 몇 문장으로 적어 주세요. "
                                 "예) 배터리 셀마다 온도·가스 센서를 두고 열폭주 징후를 조기에 감지해 냉각 장치를 가동한다.")
        st.radio("아이디어 공개 여부", list(IDEA_LEVELS), key="idea_level", horizontal=True,
                 help="미공개 발명은 기밀로 다룹니다. 기밀을 받을 수 있는 LLM 경로(사내 모델)가 없으면 아무것도 보내지 않고, "
                      "KIPRIS 키워드 검색도 하지 않습니다.")
        go = st.form_submit_button("선행 특허 찾기", type="primary")
    if st.session_state.get("prior_art_result") is None and not go:   # 예시는 조사 전에만 보인다
        ex = st.columns(len(EXAMPLES) + 1)
        ex[0].caption("예시 아이디어")
        for col, (label, text) in zip(ex[1:], EXAMPLES.items()):
            col.button(label, key=f"ex-{label}", on_click=set_idea, args=(text,), use_container_width=True)

    idea = st.session_state.get("idea", "").strip()
    if go and not idea:
        st.warning("아이디어를 입력해 주세요. 위의 예시 아이디어를 눌러 채울 수도 있습니다.")
    if go and idea and not over_limit("prior_art"):
        from src.service.prior_art import prior_art_search
        level = IDEA_LEVELS[st.session_state.get("idea_level") or next(iter(IDEA_LEVELS))]
        try:
            with audited("prior_art", level=level, **audit.digest(idea)) as info:
                policy.require_route(level, PRIOR_ART_MODEL)   # 보낼 경로가 없으면 진행 표시를 띄우기 전에 멈춘다
                with st.status("준비 중…", expanded=False) as status:
                    result = prior_art_search(idea, progress=lambda msg: status.update(label=msg), level=level)
                    status.update(label="완료", state="complete")
                info.update(route=result.route, compared=[c.item_id for c in result.comparisons])
            st.session_state.prior_art_result = result
            st.session_state.sel_search = result.comparisons[0].item_id if result.comparisons else None
            get_items.clear()   # KIPRIS에서 새로 받은 특허가 있으면 목록을 다시 읽는다
        except SendBlocked as e:
            st.error(f"🔒 {e}")
            st.caption("공개해도 되는 내용이면 '공개된 내용'을 고르세요. 미공개 발명은 사내 모델이 설정된 환경에서 조사합니다. "
                       "아이디어는 어디에도 보내지 않았고, 감사 로그에는 해시만 남았습니다.")
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
               + (f" · ⚠️ {result.search_error}" if result.search_error else "")
               + (f" · 아이디어 등급 {result.level} → {result.route}" if getattr(result, "route", "") else ""))
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


# ── 페이지: 조건 탐색 ────────────────────────────────────────────────

def page_explore() -> None:
    keys = st.session_state.missing_keys
    st.title("🔎 특허 찾기")
    mode = st.segmented_control("찾는 방법", ["조건으로 찾기", "이름·번호로 찾기"], default="조건으로 찾기",
                                key="find_mode", label_visibility="collapsed") or "조건으로 찾기"
    if mode == "조건으로 찾기":   # 자연어 조건 → 후보 검색 → LLM 조건 선별
        st.caption("기술·연도·출원인·등록 상태 같은 조건을 자연어로 적으면, 조건을 모두 만족하는 특허만 보여줍니다.")
        with st.form("explore_form"):
            condition = st.text_input("조건", label_visibility="collapsed",
                                      placeholder="예) 2020년 이후 출원된 메모리 BIST 관련 등록 특허")
            submitted = st.form_submit_button("찾기", type="primary")
        if submitted and condition.strip() and not over_limit("explore"):
            from src.service.explore import explore
            try:
                with audited("explore", **public_fields(condition)), st.spinner("후보 검색 및 조건 확인 중…"):
                    matches = explore(condition)
                st.session_state.explore_result = (condition, matches)
                st.session_state.sel_explore = matches[0].item_id if matches else None
            except Exception as e:
                show_error(e)
    else:   # 목록에서 질문할 특허를 바로 고른다
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


# ── 페이지: 사내 문서 ────────────────────────────────────────────────

def route_text(route) -> str:
    return ("🏢 사내 모델" if not route.external else "🌐 외부 API") + f" {route.label} ({route.ceiling}까지 전송 허용)"


def render_internal_answer(out: dict) -> None:
    st.markdown(out["answer"])
    if out.get("route"):
        st.caption(f"{route_text(out['route'])} · 보낸 근거 {len(out['evidence'])}개 "
                   f"(최고 등급 {acl.max_level(e.level for e in out['evidence'])})")
    withheld = {e.doc_id: e for e in out.get("withheld", [])}
    if withheld:
        ceiling = out["route"].ceiling if out.get("route") else policy.primary_route().ceiling
        st.warning(f"🔒 검색 상위 근거 중 등급이 LLM 경로 상한({ceiling})을 넘어 보내지 않은 문서: "
                   + " · ".join(f"`{d}` v{e.version} {e.title} ({e.level})" for d, e in withheld.items())
                   + " — 열람 권한이 있으니 필요하면 원문을 직접 확인하세요.")
    flagged = sorted({e.doc_id for e in out.get("evidence", []) if e.flagged})
    if flagged:
        st.caption(f"⚠️ {', '.join(flagged)}에 AI에게 내리는 지시로 보이는 문장이 있어, 자료로만 읽도록 표시해 보냈습니다.")
    if out.get("removed_urls"):
        st.caption(f"🔗 근거 출처에 없는 링크 {len(out['removed_urls'])}개를 답변에서 지웠습니다.")
    if out.get("history_dropped"):
        st.caption(f"🔒 이전 대화 {out['history_dropped']}개는 이번 경로의 상한을 넘는 자료를 담고 있어 맥락으로 보내지 않았습니다.")
    render_evidence(out.get("cited", []))


def ask_internal(question: str) -> None:
    st.session_state.internal_q = question


def page_internal() -> None:
    user = st.session_state.user
    st.title("📁 사내 문서")
    st.caption("열람 권한이 있는 문서만 검색합니다. 문서 등급이 LLM 경로의 상한을 넘으면 그 문서는 보내지 않고 제목만 알려 드립니다. "
               "(데모용 가상 문서)")
    catalog = get_catalog()
    visible = [d for d in catalog.docs if acl.can_read(user, acl.acl_metadata(d.allow))]
    paths = " · ".join(route_text(r) for r in policy.routes())
    st.caption(f"LLM 경로: {paths}" + ("" if policy.local_route() else " · 사내 모델 없음 → 기밀 문서는 보내지 않음"))
    with st.expander(f"내가 볼 수 있는 문서 {len(visible)}건"):
        st.dataframe([{"문서": d.id, "종류": d.kind, "제목": d.title, "등급": d.level, "버전": f"v{d.version}",
                       "수정일": d.updated} for d in visible], hide_index=True)

    history = st.session_state.histories.setdefault("__internal__", [])
    box = st.container(height=460 if history else "content", border=False)
    with box:
        if not history:
            cols = st.columns(len(INTERNAL_EXAMPLES) + 1)
            cols[0].caption("예시 질문")
            for col, q in zip(cols[1:], INTERNAL_EXAMPLES):
                col.button(q, key=f"iq-{q}", on_click=ask_internal, args=(q,), use_container_width=True)
        for msg in history:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    render_internal_answer(msg["out"])
                else:
                    st.markdown(msg["content"])

    question = st.chat_input("사내 문서에 대해 질문하세요", key="chat-internal") or st.session_state.pop("internal_q", None)
    if not question:
        return
    # 대화마다 담긴 자료의 등급을 함께 넘긴다 — 그래프가 고른 경로의 상한을 넘는 기록은 보내지 않는다
    prev = [(h["role"], h["content"], h.get("level", "기밀")) for h in history[-HISTORY_TURNS:]]
    history.append({"role": "user", "content": question})
    with box:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            if over_limit("internal_qa"):
                history.pop()
                return
            try:
                with audited("internal_qa", **audit.digest(question)) as info, \
                        st.status("권한 필터 검색 · 전송 정책 확인 · 답변 생성 중…", expanded=False) as status:
                    out = get_internal_graph().invoke({"user": user, "question": question, "history": prev})
                    status.update(label="완료", state="complete")
                    info.update(route=out["route"].label if out.get("route") else "-",
                                level=acl.max_level(e.level for e in out["evidence"]),
                                sent=sorted({e.doc_id for e in out["evidence"]}),
                                withheld=sorted({e.doc_id for e in out["withheld"]}),
                                flagged=sorted({e.doc_id for e in out["evidence"] if e.flagged}),
                                removed_urls=len(out.get("removed_urls", [])),
                                history_dropped=out.get("history_dropped", 0))
                render_internal_answer(out)
                level = answer_level(out["evidence"])
                history[-1]["level"] = level   # 질문도 그 답과 같은 등급으로 본다 (답을 이어 묻는 질문이므로)
                history.append({"role": "assistant", "content": out["answer"], "out": out, "level": level})
            except Exception as e:
                history.pop()
                show_error(e)


# ── 페이지: 운영 현황 (시스템 관리자) ─────────────────────────────────

def page_ops() -> None:
    import pandas as pd
    st.title("📊 운영 현황")
    events = audit.read()
    day, total = audit.summarize(events, audit.today()), audit.summarize(events)
    c = st.columns(5)
    c[0].metric("오늘 LLM 요청", day["requests"], help=f"누적 {total['requests']}")
    c[1].metric("오늘 차단", day["blocked"], help="전송 정책으로 아무것도 보내지 않은 요청")
    c[2].metric("오늘 오류", day["errors"])
    c[3].metric("오늘 로그인 실패", day["login_fail"])
    c[4].metric("오늘 토큰", f"{day['tokens']:,}")
    p50 = f"{day['p50']:.0f}초" if day["p50"] is not None else "-"
    p95 = f"{day['p95']:.0f}초" if day["p95"] is not None else "-"
    st.caption(f"응답 시간(오늘 성공 요청) p50 {p50} · p95 {p95} · 보내지 않은 문서(누적) {total['withheld']}건 · "
               f"비용: 무료 등급·로컬 임베딩이라 0원 · 감사 로그 `{audit.AUDIT_LOG.name}` {len(events)}줄")

    t_user, t_recent, t_index, t_policy = st.tabs(["사용자별", "최근 기록", "사내 문서 색인", "전송 정책"])
    with t_user:
        accounts = get_accounts().users
        rows = [{"아이디": u, "역할": accounts[u].role_label if u in accounts else "-",
                 "일일 한도": accounts[u].daily_limit if u in accounts else "-", **c}
                for u, c in total["by_user"].items()]
        st.dataframe(rows or [{"안내": "아직 기록이 없습니다"}], hide_index=True)
        daily = Counter(str(e["ts"])[:10] for e in events if e.get("event") in audit.LLM_EVENTS)
        if daily:
            st.bar_chart(pd.Series(daily, name="LLM 요청").sort_index())
    with t_recent:
        cols = ("ts", "user", "event", "outcome", "level", "route", "sent", "withheld", "seconds", "tokens")
        recent = [{k: (", ".join(v) if isinstance(v, list) else v) for k, v in e.items() if k in cols}
                  for e in reversed(events[-50:])]
        st.dataframe(recent, hide_index=True)
        st.caption("입력 원문은 기록하지 않습니다 (SHA-256 앞 12자와 글자 수만).")
    with t_index:
        try:
            rep = ensure_internal()
        except Exception:
            log.exception("사내 문서 색인 동기화 실패")
            rep = None
            st.error("사내 문서 색인을 확인하지 못했습니다. 서버 로그를 확인하세요.")
        if rep is not None:
            st.dataframe([{"문서": d.id, "등급": d.level, "열람 주체": ", ".join(d.allow), "버전": f"v{d.version}",
                           "수정일": d.updated, "검토 필요": ", ".join(rep.flagged.get(d.id, []))}
                          for d in get_catalog().docs], hide_index=True)
            st.caption(f"색인 청크 {rep.chunks}개 · 이번 실행에서 새로 색인 {len(rep.added)} · 다시 색인 {len(rep.updated)} · "
                       f"삭제 {len(rep.removed)}")
            for doc_id, reason in rep.rejected:
                st.error(f"색인 안 함 — {doc_id}: {reason}")
            for f in rep.unregistered:
                st.warning(f"카탈로그에 없는 파일(색인 안 함): {f}")
            for w in rep.warnings:
                st.warning(w)
        if st.button("카탈로그 다시 읽고 동기화"):
            get_catalog.clear()
            ensure_internal.clear()
            audit.record("ingest_request", st.session_state.user)
            st.rerun()
    with t_policy:
        st.dataframe([{"경로": r.label, "보낼 수 있는 최고 등급": r.ceiling, "외부 전송": "예" if r.external else "아니오"}
                      for r in policy.routes()], hide_index=True)
        st.caption("기본값은 제공자 약관 기준(Gemini 무료 등급 → 공개, OpenAI API → 내부, 사내 모델 → 기밀). "
                   "`LLM_MAX_LEVEL`·`LOCAL_LLM_BASE_URL`로 바꾼다. 근거: docs/decisions/009-send-policy.md")


# ── 로그인 ───────────────────────────────────────────────────────────────────

def render_login(accounts) -> None:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.title("🔬 Patent Lab")
        st.caption("선행기술 조사와 사내 연구 문서 질의응답. 로그인한 계정의 열람 권한 안에서만 검색하고, "
                   "자료 등급에 따라 보낼 수 있는 LLM만 씁니다.")
        with st.form("login_form"):
            username = st.text_input("아이디")
            password = st.text_input("비밀번호", type="password")
            submitted = st.form_submit_button("로그인", type="primary", use_container_width=True)
        if submitted:
            res = authenticate(username, password, accounts)
            name = username.strip().lower()   # 비밀번호를 아이디 칸에 잘못 넣은 경우가 로그에 남지 않게, 없는 아이디는 이름을 남기지 않는다
            audit.record("login", res.user, outcome=res.status, attempted=name if name in accounts.users else "(없는 아이디)")
            if res.status == "ok":
                st.session_state.user = res.user
                st.rerun()
            elif res.status == "locked":
                st.error(f"로그인 실패가 반복되어 이 아이디를 {res.wait // 60 + 1}분 동안 잠갔습니다.")
            else:
                st.error("아이디 또는 비밀번호가 맞지 않습니다.")
        if accounts.demo and APP_ENV != "production":
            docs = get_catalog().docs
            with st.expander("데모 계정 (가상 인물 · 로컬 데모 전용)", expanded=True):
                st.dataframe([{"아이디": u.username, "이름": u.name, "역할": u.role_label,
                               "볼 수 있는 사내 문서": f"{sum(acl.can_read(u, acl.acl_metadata(d.allow)) for d in docs)}/{len(docs)}건"}
                              for u in accounts.users.values()], hide_index=True)
                st.caption("비밀번호는 모두 `patentlab-demo`입니다. 운영 환경(APP_ENV=production)에서는 이 계정 파일로 시작하지 않습니다.")


# ── 진입 ─────────────────────────────────────────────────────────────────────

if not PATENTS_JSON.exists():
    st.warning("특허 메타데이터가 없습니다. 터미널에서 `python -m src.data_prep.collect_meta`를 먼저 실행하세요.")
    st.stop()

meta_count, doc_count = count_chunks(META_COLLECTION), count_chunks(DOC_COLLECTION)
if meta_count == 0 and snapshot.exists():   # 다른 PC에서 처음 실행: 저장소의 스냅샷으로 색인을 만든다 (API 호출 없음)
    if not snapshot.compatible():   # 임베딩 모델이 다른 스냅샷을 복원하면 질문 벡터와 색인 벡터가 맞지 않아 검색이 틀어진다
        st.warning(f"저장소의 색인 스냅샷은 임베딩 모델 `{snapshot.embedding_model()}`로 만든 것이라 현재 설정 "
                   f"`{EMBEDDING_MODEL}`에서는 쓸 수 없습니다. 터미널에서 색인을 다시 만드세요:\n\n"
                   "`python -m src.vectorstore.index_meta --reset` → `python -m src.vectorstore.index_docs --rebuild` "
                   "→ `python -m src.vectorstore.snapshot export`")
        st.stop()
    with st.spinner("저장소에 포함된 색인 스냅샷으로 색인을 만드는 중입니다 (처음 한 번, 약 20초)…"):
        snapshot.restore()
    st.rerun()

try:
    accounts = get_accounts()
    policy.routes()   # 전송 상한 설정값 검사 — 틀리면 어떤 경로로도 보내지 않고 멈춘다
except AccountsError as e:
    st.error(f"계정 설정을 읽지 못했습니다. {e}")
    st.stop()
except ValueError as e:
    st.error(f"전송 정책 설정(LLM_MAX_LEVEL·LOCAL_LLM_MAX_LEVEL)이 잘못되었습니다. {e}")
    st.stop()
if problem := startup_problem(accounts):
    st.error(problem)
    st.stop()
if "user" not in st.session_state:
    # 로그인 전에는 로그인 페이지 하나만 등록한다 — 이전 세션의 메뉴(운영 현황 등)가 남아 보이지 않게
    st.navigation([st.Page(lambda: render_login(accounts), title="로그인", icon="🔐", url_path="login")],
                  position="hidden").run()
    st.stop()

user = st.session_state.user
st.session_state.missing_keys = missing_keys()
st.session_state.setdefault("histories", {})
st.session_state.setdefault("briefs", {})

with st.sidebar:
    st.title("🔬 Patent Lab")
    st.markdown(f"**{user.name}**  \n{user.role_label}"
                + (f" · {', '.join(get_catalog().teams.get(t, t) for t in user.teams)}" if user.teams else ""))
    used = audit.count_today(user.username)
    st.progress(min(1.0, used / max(1, user.daily_limit)), text=f"오늘 LLM 요청 {used}/{user.daily_limit}회")
    st.button("로그아웃", on_click=logout, use_container_width=True)
    fields = Counter(it.metadata["field"] for it in get_items().values())
    c1, c2 = st.columns(2)
    c1.metric("색인된 특허", f"{meta_count:,}")
    c2.metric("문서 청크", f"{doc_count:,}")
    with st.expander(f"수집 분야 {len(fields)}개"):
        st.markdown("\n".join(f"- {name} {n:,}건" for name, n in fields.most_common()))
        st.caption("선행기술 조사 때 KIPRIS에서 관련 특허를 더 받아 '검색 수집'으로 쌓습니다.")
    with st.expander("외부 전송 정책"):
        st.markdown("\n".join(f"- {r.label}: **{r.ceiling}**까지" for r in policy.routes())
                    + ("" if policy.local_route() else "\n- 사내 모델: 없음 (기밀 자료는 보내지 않음)"))
    st.info(DISCLAIMER, icon="⚖️")

# 진행 불가 상황 안내
if meta_count == 0:
    st.warning("특허 색인이 비어 있습니다. 터미널에서 `python -m src.vectorstore.index_meta`를 실행한 뒤 새로고침하세요.")
    st.stop()
try:
    ensure_internal()
except Exception:   # 사내 문서 색인이 실패해도 공개 특허 기능은 쓸 수 있게 한다
    log.exception("사내 문서 색인 동기화 실패")
    st.warning("사내 문서 색인을 확인하지 못했습니다. 사내 문서 검색 결과가 비어 있거나 오래되었을 수 있습니다.")
if "openai" in st.session_state.missing_keys:
    st.error(f"API 키 설정이 필요합니다. {st.session_state.missing_keys['openai']}")
    st.stop()
if st.session_state.missing_keys:
    st.warning("외부 문서 수집에 필요한 설정이 없습니다. 이미 색인된 특허·문서로만 조사하고 답합니다. "
               + " ".join(st.session_state.missing_keys.values()))

pages = [
    st.Page(page_prior_art, title="선행기술 조사", icon="🧭", url_path="prior-art", default=True),
    st.Page(page_explore, title="특허 찾기", icon="🔎", url_path="explore"),
    st.Page(page_internal, title="사내 문서", icon="📁", url_path="internal"),
]
if user.role == "admin":   # 운영 현황은 시스템 관리자만 (문서 열람 권한과는 별개)
    pages.append(st.Page(page_ops, title="운영 현황", icon="📊", url_path="ops"))
st.navigation(pages, position="top").run()
