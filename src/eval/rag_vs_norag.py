# src/eval/rag_vs_norag.py — RAG 사용 전후 답변 비교용 원자료 생성
# 같은 질문을 (1) 특허 명칭·출원번호만 주고 LLM에 바로 묻기, (2) RAG 파이프라인으로 묻기 두 번 답하게 한다.
# 실행: python -m src.eval.rag_vs_norag   결과: docs/eval/rag_vs_norag_raw.md
from langchain_core.prompts import ChatPromptTemplate

from src.config import ROOT
from src.data_prep.items import load_items
from src.rag.graph import build_graph
from src.rag.llm import get_chat
from src.rag.prompts import EXPERT_ROLE

# 근거 없이 답하면 틀리기 쉬운 질문(청구항·서지 사실)과, 문서로 답할 수 없는 질문을 섞는다.
QUESTIONS = [
    ("1020230136220", "청구항 1의 회로에 포함된 구성 요소를 모두 나열해 줘."),
    ("1020230136220", "이 특허의 출원인, 출원일, 현재 등록 상태는?"),
    ("1020210190872", "이 특허는 스캔 체인의 다중 고장을 어떤 원리로 진단하나?"),
    ("1020230005129", "이 특허가 판별하는 고장 유형은 무엇이고, 학습 데이터는 무엇으로 만드나?"),
    ("1020230136220", "이 회로의 양산 단가와 삼성전자 제품에 적용된 시점은?"),
    ("1020210190872", "이 특허는 미국에도 출원되었나? 미국 특허번호를 알려줘."),
]
OUT = ROOT / "docs" / "eval" / "rag_vs_norag_raw.md"

NO_RAG = ChatPromptTemplate.from_messages([
    ("system", EXPERT_ROLE + " 질문에 한국어로 간결하게 답한다."),
    ("human", "[특허] {title} (출원번호 {item_id})\n[질문] {question}"),
])


def main() -> None:
    items = {it.id: it for it in load_items()}
    llm, graph = get_chat(), build_graph()
    lines = ["# RAG 사용 전후 답변 (원자료)\n",
             "- RAG 없음: 같은 모델에 특허 명칭·출원번호와 질문만 준다.",
             "- RAG 사용: 수집·검색·근거 기반 답변 파이프라인 (기본 설정).\n"]
    for i, (item_id, question) in enumerate(QUESTIONS, 1):
        item = items[item_id]
        plain = (NO_RAG | llm).invoke({"title": item.metadata["title"], "item_id": item_id,
                                       "question": question}).content
        out = graph.invoke({"item": item, "question": question})
        cited = "\n".join(f"  - [근거 {e.n}] ({e.label}) {e.title} — {e.url}" for e in out["cited"]) \
            or "  - (인용된 근거 없음)"
        lines += [f"## Q{i}. {question}", f"특허: {item.metadata['title']} ({item_id})\n",
                  "### RAG 없음", plain, "", "### RAG 사용", out["answer"], "", "근거:", cited, ""]
        print(f"Q{i} 완료 (인용 {len(out['cited'])}개)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"결과 저장 → {OUT}")


if __name__ == "__main__":
    main()
