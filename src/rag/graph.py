# src/rag/graph.py — RAG 파이프라인: retrieve → generate 두 노드의 LangGraph
#
# 실행 (프로젝트 루트에서)
#   python -m src.rag.graph --item 1020230136220 --question "지터를 측정하는 원리는?"
#   옵션: --top-k 8  --search mmr  --no-collect  --show-graph
import argparse
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from src.config import SEARCH_TYPE, TOP_K
from src.data_prep.collect_docs import CollectResult
from src.data_prep.items import Item, load_items
from src.env import MissingAPIKeyError
from src.rag.citations import cited_evidence, strip_citations
from src.rag.llm import get_chat
from src.rag.prompts import ANSWER_GENERATION, NOT_FOUND
from src.rag.retriever import Evidence, format_evidence, retrieve_evidence
from src.vectorstore.index_docs import ensure_documents, print_result


class RAGState(TypedDict, total=False):
    # 입력
    item: Item
    question: str
    history: list[tuple[str, str]]    # [("user"|"assistant", 내용)] — 선택한 특허의 이전 대화
    top_k: int
    search_type: str                  # "similarity" | "mmr"
    collect: bool                     # False면 이미 색인된 문서로만 답한다
    # retrieve 결과
    collect_result: CollectResult | None
    added_chunks: int
    evidence: list[Evidence]
    # generate 결과
    answer: str
    cited: list[Evidence]             # 답변에 실제로 인용된 근거만


def build_graph(llm: BaseChatModel | None = None):
    llm = llm or get_chat()

    def retrieve(state: RAGState) -> RAGState:
        item, question = state["item"], state["question"]
        result, added = (ensure_documents(item, question, llm) if state.get("collect", True) else (None, 0))
        evidence = retrieve_evidence(item, question,
                                     top_k=state.get("top_k", TOP_K),
                                     search_type=state.get("search_type", SEARCH_TYPE))
        return {"collect_result": result, "added_chunks": added, "evidence": evidence}

    def generate(state: RAGState) -> RAGState:
        history = [(HumanMessage if role == "user" else AIMessage)(strip_citations(text))
                   for role, text in state.get("history", [])]
        answer = (ANSWER_GENERATION | llm).invoke({
            "evidence": format_evidence(state["evidence"]),
            "question": state["question"],
            "history": history,
            "not_found": NOT_FOUND,
        }).content
        return {"answer": answer, "cited": cited_evidence(answer, state["evidence"])}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def main() -> None:
    ap = argparse.ArgumentParser(description="근거 기반 답변")
    ap.add_argument("--item", required=True, help="출원번호")
    ap.add_argument("--question", required=True)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--search", choices=["similarity", "mmr"], default=SEARCH_TYPE)
    ap.add_argument("--no-collect", action="store_true", help="문서 수집 없이 이미 색인된 문서로만 답한다")
    ap.add_argument("--show-graph", action="store_true", help="그래프 구조를 출력한다")
    args = ap.parse_args()

    item = next((it for it in load_items() if it.id == args.item), None)
    if item is None:
        raise SystemExit(f"출원번호 {args.item}인 특허가 항목 목록에 없습니다.")
    try:
        app = build_graph()
    except MissingAPIKeyError as e:
        raise SystemExit(f"[중단] {e}") from e
    if args.show_graph:
        print(app.get_graph().draw_mermaid())

    out = app.invoke({"item": item, "question": args.question, "top_k": args.top_k,
                      "search_type": args.search, "collect": not args.no_collect})

    print(f"[{item.id}] {item.metadata['title']}\n질문: {args.question}  (top_k={args.top_k}, {args.search})\n")
    if out.get("collect_result"):
        print("── 문서 수집")
        print_result(out["collect_result"], out["added_chunks"])
        print()
    cited_ids = {e.chunk_id for e in out["cited"]}
    print(f"── 검색된 근거 {len(out['evidence'])}개 (* = 답변에 인용)")
    for e in out["evidence"]:
        print(f" {'*' if e.chunk_id in cited_ids else ' '} [근거 {e.n}] ({e.label}) {e.title}  {e.chunk_id}")
    print(f"\n── 답변\n{out['answer']}\n")
    print("── 근거 문서")
    if not out["cited"]:
        print("  답변에 인용된 근거가 없습니다.")
    for e in out["cited"]:
        print(f"  [근거 {e.n}] ({e.label}) {e.title} — {e.url}")


if __name__ == "__main__":
    main()
