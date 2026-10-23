# src/eval/compare_search.py — 유사도 검색과 MMR 검색의 근거 차이 비교 (F2014)
# 실행: python -m src.eval.compare_search [--top-k 5]   결과: docs/eval/f2014_mmr_k<k>.md
import argparse
import itertools
from collections import Counter

import numpy as np

from src.config import DOC_COLLECTION, MMR_FETCH_K, MMR_LAMBDA, ROOT, TOP_K
from src.data_prep.docs import SOURCE_LABELS
from src.data_prep.items import load_items
from src.rag.retriever import search_docs
from src.vectorstore.store import get_store

# 문서가 이미 수집된 특허와 대표 질문
CASES = [
    ("1020230136220", "지터는 무엇이고, 이 특허는 지터를 어떻게 측정하나?"),
    ("1020210190872", "스캔 체인 고장을 진단하는 이 특허의 핵심 방법은 무엇이고, 스캔 체인은 왜 테스트에 쓰이나?"),
    ("1020230005129", "이 특허는 기계 학습을 어떻게 쓰고, 기계 학습은 어떤 기술인가?"),
]
OUT_DIR = ROOT / "docs" / "eval"


def chunk_id(doc) -> str:
    m = doc.metadata
    return f"{m['item_id']}:{m['doc_id']}:{m['chunk_index']}"


def redundancy(store, ids: list[str]) -> float:
    """고른 청크끼리의 평균 코사인 유사도. 높을수록 내용이 겹친다."""
    got = store.get(ids=ids, include=["embeddings"])
    vecs = np.array(got["embeddings"], dtype=float)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    pairs = [float(vecs[i] @ vecs[j]) for i, j in itertools.combinations(range(len(vecs)), 2)]
    return sum(pairs) / len(pairs) if pairs else 0.0


def summarize(store, docs) -> dict:
    ids = [chunk_id(d) for d in docs]
    return {
        "ids": ids,
        "docs": Counter(d.metadata["doc_title"] for d in docs),
        "types": Counter(SOURCE_LABELS[d.metadata["source_type"]] for d in docs),
        "redundancy": redundancy(store, ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=TOP_K)
    k = ap.parse_args().top_k

    items = {it.id: it for it in load_items()}
    store = get_store(DOC_COLLECTION)
    lines = ["# F2014 유사도 검색 vs MMR 검색 비교\n",
             f"- 외부 문서 근거 수 k = {k}, MMR 후보 fetch_k = {MMR_FETCH_K}, lambda = {MMR_LAMBDA}",
             "- 겹침 = 고른 청크끼리의 평균 코사인 유사도 (낮을수록 다양)\n"]

    for item_id, question in CASES:
        item = items[item_id]
        sim = summarize(store, search_docs(item, question, store, k, "similarity"))
        mmr = summarize(store, search_docs(item, question, store, k, "mmr"))
        same = len(set(sim["ids"]) & set(mmr["ids"]))

        lines += [f"## {item.metadata['title']} ({item_id})", f"질문: {question}\n",
                  "| | 유사도 검색 | MMR |", "|---|---|---|",
                  f"| 서로 다른 문서 수 | {len(sim['docs'])} | {len(mmr['docs'])} |",
                  f"| 출처 종류 | {dict(sim['types'])} | {dict(mmr['types'])} |",
                  f"| 청크 간 겹침 | {sim['redundancy']:.3f} | {mmr['redundancy']:.3f} |",
                  f"| 두 방식에 공통인 청크 | {same}/{k} | |\n",
                  "| 순위 | 유사도 검색 | MMR |", "|---|---|---|"]
        for r, (a, b) in enumerate(itertools.zip_longest(sim["ids"], mmr["ids"], fillvalue="-"), 1):
            lines.append(f"| {r} | `{a.split(':', 1)[1]}` | `{b.split(':', 1)[1]}` |")
        lines.append("")
        print(f"[{item_id}] 문서 수 {len(sim['docs'])}→{len(mmr['docs'])}, "
              f"겹침 {sim['redundancy']:.3f}→{mmr['redundancy']:.3f}, 공통 {same}/{k}")
        print(f"    유사도: {dict(sim['types'])}\n    MMR  : {dict(mmr['types'])}")

    out = OUT_DIR / f"f2014_mmr_k{k}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n결과 저장 → {out}")


if __name__ == "__main__":
    main()
