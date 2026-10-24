# src/eval/make_prior_art_cases.py — 선행기술 조사 자동 평가셋 생성
#
# 사람이 라벨링한 "아이디어 ↔ 선행 특허" 쌍이 없으므로, 청구항이 이미 캐시된 특허의 청구항 1을
# LLM이 "엔지니어가 아이디어를 설명하듯" 바꿔 쓴 문장을 입력으로 쓴다. 그러면 정답(그 특허 자신)이
# 자동으로 정해진다. 청구항 문장을 그대로 베끼면 검색이 너무 쉬워지므로, 베끼지 않게 지시하고
# 실제로 얼마나 겹치는지(가장 긴 공통 구절 길이)를 함께 기록한다.
#
# 실행: python -m src.eval.make_prior_art_cases [--min-claim-chars 100] [--limit N] [--overwrite]
# 결과: docs/eval/prior_art_cases.json — 한 번 만들면 고정하고, 실험마다 같은 파일을 쓴다.
# 외부 호출: 사례당 LLM 1회 (CHAT_MODEL). KIPRIS·위키 호출은 없다.
import argparse
import difflib
import json
import re
import sys
from datetime import date

from langchain_core.prompts import ChatPromptTemplate

from src.config import CHAT_MODEL, DETAIL_CACHE_DIR, ROOT
from src.data_prep import kipris
from src.data_prep.items import load_items
from src.rag.llm import get_chat

OUT = ROOT / "docs" / "eval" / "prior_art_cases.json"

PARAPHRASE = ChatPromptTemplate.from_messages([
    ("system", """당신은 엔지니어다. 아래 특허 청구항 1을, 특허를 모르는 동료에게 자기 아이디어를 설명하듯
2~4문장의 한국어로 다시 쓴다.

규칙
- 청구항의 핵심 구성 요소와 각 요소의 역할·연결 관계를 모두 담는다. 빠뜨리거나 새로 지어내지 않는다.
- 청구항 문장을 그대로 옮기지 않는다. 어순과 표현을 바꾸고, 같은 구절을 8자 이상 연속으로 베끼지 않는다.
- "상기", "~를 특징으로 하는", "~에 있어서", "청구항", "~하는 단계" 같은 특허 문체를 쓰지 않는다.
- 발명의 명칭, 출원인, 특허 번호를 언급하지 않는다.
- 기술 용어는 보통 쓰는 표기로 쓴다 (영문 병기 가능).
- 출력은 설명문만. 머리말·따옴표·목록을 붙이지 않는다."""),
    ("human", "[청구항 1]\n{claim}"),
])


def claim_1(app_no: str) -> str:
    """캐시된 서지 상세에서 첫 청구항 본문 ("1. 삭제"처럼 지워진 청구항은 건너뛰므로 2항일 수도 있다). 없으면 빈 문자열."""
    doc = kipris.patent_document(app_no)   # 캐시가 있으면 호출하지 않는다
    m = re.search(r"\[청구항\]\n(.*?)(?:\n\n|\Z)", doc.text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"\([^)]*\)", "", text))


def longest_common_run(a: str, b: str) -> int:
    """공백·괄호 병기를 지운 뒤 가장 긴 공통 구절의 길이. 아이디어가 청구항을 얼마나 베꼈는지 본다."""
    a, b = _squash(a), _squash(b)
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size


def save(cases: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"description": "첫 청구항을 아이디어 문장으로 바꿔 쓴 자동 평가 사례. 정답 = item_id 자신.",
                               "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="선행기술 조사 자동 평가셋 생성")
    ap.add_argument("--min-claim-chars", type=int, default=100, help="이보다 짧은 청구항 1은 사례에서 뺀다")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true", help="기존 파일이 있어도 전부 다시 만든다")
    args = ap.parse_args()

    existing = {}
    if OUT.exists() and not args.overwrite:
        existing = {c["item_id"]: c for c in json.loads(OUT.read_text(encoding="utf-8"))["cases"]}

    items = {it.id: it for it in load_items()}
    cached = sorted(p.stem for p in DETAIL_CACHE_DIR.glob("*.xml") if p.stem in items)
    llm = get_chat()   # CHAT_MODEL, temperature 0
    cases, skipped = [], []
    for app_no in cached:
        if args.limit and len(cases) >= args.limit:
            break
        if app_no in existing:
            cases.append(existing[app_no])
            continue
        claim = claim_1(app_no)
        if len(claim) < args.min_claim_chars:
            skipped.append((app_no, len(claim)))
            continue
        idea = (PARAPHRASE | llm).invoke({"claim": claim}).content.strip()
        run = longest_common_run(idea, claim)
        cases.append({
            "item_id": app_no,
            "title": items[app_no].metadata["title"],
            "field": items[app_no].metadata["field"],
            "claim": claim,          # 첫 청구항 원문 (사례 검토용)
            "idea": idea,
            "max_common_run": run,   # 아이디어와 청구항의 가장 긴 공통 구절(자) — 작을수록 표현 차이가 크다
            "model": CHAT_MODEL,
            "generated_at": date.today().isoformat(),
        })
        print(f"{app_no} 공통 구절 {run:>2}자 | {idea[:70]}…")
        save(cases)   # 요청 한도 등으로 중간에 멈춰도 만든 사례는 남기고, 다시 실행하면 이어서 만든다

    save(cases)
    runs = [c["max_common_run"] for c in cases]
    print(f"\n사례 {len(cases)}건 저장 → {OUT}")
    if runs:
        print(f"아이디어-청구항 공통 구절: 평균 {sum(runs) / len(runs):.1f}자, 최대 {max(runs)}자")
    if skipped:
        print(f"청구항 1이 {args.min_claim_chars}자 미만이라 뺀 특허 {len(skipped)}건: "
              + ", ".join(f"{a}({n}자)" for a, n in skipped))


if __name__ == "__main__":
    main()
