# src/eval/prior_art_eval.py — 선행기술 조사 자동 평가
#
# docs/eval/prior_art_cases.json의 사례마다 (청구항 1을 바꿔 쓴 아이디어 → 정답 = 그 특허 자신)
# 선행기술 조사를 돌리고, 정답 특허가 어디까지 올라오는지와 판정이 맞는지를 잰다.
#
#   검색   Recall@20 — 메타데이터 검색 후보 20건에 정답이 있는가
#          Recall@5  — 후보 중 LLM이 고른 대비 대상 5건에 정답이 있는가,  MRR@5 — 그 순위의 역수 평균
#   판정   개시율    — 정답 특허에 대한 구성요소 판정 중 "개시"의 비율 (자기 청구항이므로 전부 개시가 정답)
#          개시+일부 — "개시" 또는 "일부"의 비율
#          겹침 1위율 — 대비한 특허 중 정답의 겹침 비율이 가장 높은 사례의 비율
#   비용   사례당 시간·토큰·추정 비용
#
# 실행: python -m src.eval.prior_art_eval --label baseline [--limit N] [--cases 경로]
# 결과: docs/eval/prior_art_eval/<label>_<날짜>.json (원자료), 같은 이름 .md (사례별 표),
#       docs/eval/prior_art_eval.md (실험 요약표에 한 줄 추가 — 설정을 바꿀 때마다 비교)
# 외부 호출: LLM만 (사례당 구성요소 1 + 선별 1 + 대비 최대 5회). KIPRIS 검색·청구항 수집은 하지 않는다
#            (expand=False, cached_only=True) — 같은 조건으로 반복하고 월 호출 한도를 쓰지 않기 위해서.
import argparse
import json
import sys
import time
from datetime import date

from langchain_community.callbacks.manager import get_openai_callback

from src import config
from src.config import ROOT
from src.service.prior_art import prior_art_search

CASES = ROOT / "docs" / "eval" / "prior_art_cases.json"
OUT_DIR = ROOT / "docs" / "eval" / "prior_art_eval"
SUMMARY = ROOT / "docs" / "eval" / "prior_art_eval.md"

# 추정 단가 (USD / 1M 토큰, 2026-10 기준 공개 가격). 모델이 표에 없으면 비용은 비워 둔다.
PRICE = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def settings() -> dict:
    return {
        "prior_art_model": config.PRIOR_ART_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "candidates": config.PRIOR_ART_CANDIDATES,
        "max_compare": config.PRIOR_ART_MAX,
        "mmr_fetch_k": config.MMR_FETCH_K,
        "expand": False,
        "cached_only": True,
    }


def run_case(case: dict) -> dict:
    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        result = prior_art_search(case["idea"], expand=False, cached_only=True)
    seconds = time.perf_counter() - t0
    gold = case["item_id"]

    cand_rank = result.candidate_ids.index(gold) + 1 if gold in result.candidate_ids else None
    sel_rank = result.selected_ids.index(gold) + 1 if gold in result.selected_ids else None
    source = next((c for c in result.comparisons if c.item_id == gold), None)
    others = [c for c in result.comparisons if c.item_id != gold]
    statuses = {"개시": 0, "일부": 0, "미개시": 0}
    if source:
        for r in source.rows:
            statuses[r.status] += 1
    price = PRICE.get(config.PRIOR_ART_MODEL)
    cost = (cb.prompt_tokens * price[0] + cb.completion_tokens * price[1]) / 1e6 if price else None
    return {
        "item_id": gold,
        "title": case["title"],
        "n_elements": len(result.elements),
        "elements": result.elements,
        "candidate_rank": cand_rank,
        "selected_rank": sel_rank,
        "compared": len(result.comparisons),
        "skipped_note": result.search_error,
        "source_overlap": source.overlap if source else None,
        "source_statuses": statuses if source else None,
        "source_rows": [{"element": r.name, "status": r.status, "quote": r.quote, "note": r.note}
                        for r in source.rows] if source else [],
        "best_other_overlap": max((c.overlap for c in others), default=None),
        "overlap_rank": (1 + sum(1 for c in others if c.overlap > source.overlap)) if source else None,
        "others": [{"item_id": c.item_id, "title": c.title, "overlap": c.overlap} for c in others],
        "seconds": round(seconds, 1),
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "est_cost_usd": round(cost, 4) if cost is not None else None,
    }


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(rows: list[dict]) -> dict:
    compared = [r for r in rows if r["source_statuses"]]
    n_el = sum(sum(r["source_statuses"].values()) for r in compared)
    disclosed = sum(r["source_statuses"]["개시"] for r in compared)
    partial = sum(r["source_statuses"]["일부"] for r in compared)
    return {
        "cases": len(rows),
        "recall_at_20": _mean([r["candidate_rank"] is not None for r in rows]),
        "recall_at_5": _mean([r["selected_rank"] is not None for r in rows]),
        "mrr_at_5": _mean([1 / r["selected_rank"] if r["selected_rank"] else 0.0 for r in rows]),
        "source_compared": len(compared),
        "disclosed_rate": disclosed / n_el if n_el else None,
        "disclosed_or_partial_rate": (disclosed + partial) / n_el if n_el else None,
        "overlap_top1_rate": _mean([r["overlap_rank"] == 1 for r in compared]),
        "mean_seconds": _mean([r["seconds"] for r in rows]),
        "mean_tokens": _mean([r["prompt_tokens"] + r["completion_tokens"] for r in rows]),
        "mean_cost_usd": _mean([r["est_cost_usd"] for r in rows]),
    }


def pct(v) -> str:
    return "-" if v is None else f"{v:.0%}"


def usd(v, digits: int = 3) -> str:
    return "-" if v is None else f"${v:.{digits}f}"


def ratio(s: dict | None) -> str:
    return f"{s['개시']}/{s['일부']}/{s['미개시']}" if s else "-"


def write_detail(path, label: str, agg: dict, rows: list[dict], cfg: dict) -> None:
    lines = [f"# 선행기술 조사 평가 — {label} ({date.today().isoformat()})\n",
             "설정: " + ", ".join(f"{k}={v}" for k, v in cfg.items()), "",
             "| 지표 | 값 |", "|---|---|",
             f"| 사례 수 | {agg['cases']} |",
             f"| Recall@20 (후보) | {pct(agg['recall_at_20'])} |",
             f"| Recall@5 (대비 대상) | {pct(agg['recall_at_5'])} |",
             f"| MRR@5 | {agg['mrr_at_5']:.3f} |" if agg["mrr_at_5"] is not None else "| MRR@5 | - |",
             f"| 개시율 (정답 특허, 대비한 {agg['source_compared']}건) | {pct(agg['disclosed_rate'])} |",
             f"| 개시+일부율 | {pct(agg['disclosed_or_partial_rate'])} |",
             f"| 겹침 1위율 | {pct(agg['overlap_top1_rate'])} |",
             f"| 사례당 평균 시간 | {agg['mean_seconds']:.1f}초 |",
             f"| 사례당 평균 토큰 | {agg['mean_tokens']:,.0f} |",
             f"| 사례당 추정 비용 | {usd(agg['mean_cost_usd'], 4)} |",
             "", "## 사례별", "",
             "| 특허 | 구성요소 | 후보 순위 | 대비 순위 | 정답 겹침 | 개시/일부/미개시 | 다른 특허 최고 겹침 | 시간 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['title'][:30]} ({r['item_id']}) | {r['n_elements']} | {r['candidate_rank'] or '-'} | "
            f"{r['selected_rank'] or '-'} | {pct(r['source_overlap'])} | {ratio(r['source_statuses'])} | "
            f"{pct(r['best_other_overlap'])} | {r['seconds']}초 |")
    lines += ["", "## 정답 특허에 대한 판정 (미개시·일부만 — 검토용)", ""]
    for r in rows:
        bad = [x for x in r["source_rows"] if x["status"] != "개시"]
        if not bad:
            continue
        lines.append(f"**{r['title']} ({r['item_id']})**")
        for x in bad:
            lines.append(f"- [{x['status']}] {x['element']} — {x['note']}" + (f' / 인용 "{x["quote"]}"' if x["quote"] else ""))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def append_summary(label: str, agg: dict, cfg: dict, detail_name: str) -> None:
    header = ["# 선행기술 조사 자동 평가 — 실험 요약", "",
              "사례: [prior_art_cases.json](prior_art_cases.json) (청구항 1을 바꿔 쓴 아이디어, 정답 = 그 특허). "
              "실행: `python -m src.eval.prior_art_eval --label <이름>`. 사례별 결과는 [prior_art_eval/](prior_art_eval/).", "",
              "| 날짜 | 라벨 | 사례 | R@20 | R@5 | MRR@5 | 개시율 | 개시+일부 | 겹침 1위 | 시간/건 | 토큰/건 | 비용/건 | 모델 | 후보/대비 | 상세 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    row = (f"| {date.today().isoformat()} | {label} | {agg['cases']} | {pct(agg['recall_at_20'])} | {pct(agg['recall_at_5'])} | "
           f"{agg['mrr_at_5']:.3f} | {pct(agg['disclosed_rate'])} | {pct(agg['disclosed_or_partial_rate'])} | "
           f"{pct(agg['overlap_top1_rate'])} | {agg['mean_seconds']:.0f}초 | {agg['mean_tokens']:,.0f} | "
           f"{usd(agg['mean_cost_usd'])} | {cfg['prior_art_model']} | "
           f"{cfg['candidates']}/{cfg['max_compare']} | [{detail_name}](prior_art_eval/{detail_name}) |")
    text = SUMMARY.read_text(encoding="utf-8") if SUMMARY.exists() else "\n".join(header)
    SUMMARY.write_text(text.rstrip("\n") + "\n" + row + "\n", encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="선행기술 조사 자동 평가")
    ap.add_argument("--label", required=True, help="실험 이름 (예: baseline, hybrid, rerank)")
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--limit", type=int, default=None, help="앞에서부터 N개 사례만")
    args = ap.parse_args()

    cases = json.loads(open(args.cases, encoding="utf-8").read())["cases"][: args.limit]
    if not cases:
        raise SystemExit("사례가 없습니다. 먼저 python -m src.eval.make_prior_art_cases 를 실행하세요.")
    cfg = settings()
    print(f"[{args.label}] 사례 {len(cases)}건, 설정 " + ", ".join(f"{k}={v}" for k, v in cfg.items()))

    rows, failed = [], []
    for i, case in enumerate(cases, 1):
        try:
            r = run_case(case)
        except Exception as e:   # 무료 등급 일일 한도(429) 등 — 한 사례가 실패해도 나머지 결과는 남긴다
            failed.append({"item_id": case["item_id"], "title": case["title"], "error": str(e)[:600]})
            print(f"{i:>2}. {case['item_id']} 실패: {str(e)[:120]}")
            continue
        rows.append(r)
        s = r["source_statuses"]
        head = f"{i:>2}. {case['item_id']} 후보 {r['candidate_rank'] or '-':>2} / 대비 {r['selected_rank'] or '-'}"
        if s:
            print(f"{head} / 겹침 {pct(r['source_overlap']):>4} (개시 {s['개시']}/{r['n_elements']})")
        else:
            print(f"{head} / 정답 대비 안 됨" + (f" ({r['skipped_note']})" if r["skipped_note"] else ""))

    if not rows:
        raise SystemExit(f"모든 사례가 실패했습니다 ({len(failed)}건). 첫 오류: {failed[0]['error'] if failed else ''}")
    agg = aggregate(rows)
    agg["failed"] = len(failed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.label}_{date.today().isoformat()}"
    (OUT_DIR / f"{stem}.json").write_text(
        json.dumps({"label": args.label, "settings": cfg, "summary": agg, "cases": rows, "failed": failed},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if failed:
        print(f"실패 {len(failed)}건 (결과에서 제외): " + ", ".join(f["item_id"] for f in failed))
    write_detail(OUT_DIR / f"{stem}.md", args.label, agg, rows, cfg)
    append_summary(args.label, agg, cfg, f"{stem}.md")

    print(f"\nRecall@20 {pct(agg['recall_at_20'])} · Recall@5 {pct(agg['recall_at_5'])} · MRR@5 {agg['mrr_at_5']:.3f} · "
          f"개시율 {pct(agg['disclosed_rate'])} · 개시+일부 {pct(agg['disclosed_or_partial_rate'])} · "
          f"겹침 1위 {pct(agg['overlap_top1_rate'])}")
    print(f"사례당 {agg['mean_seconds']:.0f}초, {agg['mean_tokens']:,.0f} 토큰"
          + (f", 약 ${agg['mean_cost_usd']:.3f}" if agg["mean_cost_usd"] is not None else ""))
    print(f"결과 → {OUT_DIR / (stem + '.md')}, 요약표 → {SUMMARY}")


if __name__ == "__main__":
    main()
