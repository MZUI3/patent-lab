# src/data_prep/collect_meta.py
# 분야별 IPC(config.KIPRIS_FIELDS)로 KIPRIS Plus 검색 결과를 페이지 단위로 받아 캐시하고 JSON으로 저장한다.
# 실행: python -m src.data_prep.collect_meta (프로젝트 루트에서)
import json
from collections import Counter

from src.config import (
    KIPRIS_FIELDS,
    KIPRIS_ROWS as ROWS,
    PATENTS_JSON as OUT_PATH,
    SEARCH_CACHE_DIR as RAW_DIR,
)
from src.data_prep import kipris

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def fetch_page(ipc: str, page: int) -> bytes:
    """페이지 응답을 캐시에서 읽고, 없을 때만 API를 호출한다."""
    ipc_tag = ipc.replace(" ", "").replace("/", "-")
    cache = RAW_DIR / f"ipc_{ipc_tag}_rows{ROWS}_p{page:03d}.xml"
    if cache.exists():
        return cache.read_bytes()

    params = {"ipcNumber": ipc, "patent": "true", "utility": "false", "numOfRows": ROWS, "pageNo": page}
    content = kipris.call("getAdvancedSearch", params, detail=f"ipc={ipc} page={page}")   # 재시도·간격·기록
    if kipris.parse_search(content)[1]:   # 빈 결과는 캐시하지 않는다
        cache.write_bytes(content)
    return content


def collect_field(label: str, ipc: str, limit: int | None) -> tuple[int, list[dict]]:
    """한 분야를 최대 limit건(None이면 전체) 받는다. (검색 결과 전체 건수, 행 목록)"""
    total, rows = kipris.parse_search(fetch_page(ipc, 1))
    want = total if limit is None else min(total, limit)
    pages = (want + ROWS - 1) // ROWS
    for p in range(2, pages + 1):
        rows += kipris.parse_search(fetch_page(ipc, p))[1]
    for r in rows:
        r["field"] = label
    return total, rows[:want]


def main() -> None:
    patents: dict[str, dict] = {}
    for label, ipc, limit in KIPRIS_FIELDS:
        total, rows = collect_field(label, ipc, limit)
        new = 0
        for r in rows:
            if r["ApplicationNumber"] and r["ApplicationNumber"] not in patents:   # 출원번호 기준 중복 제거
                patents[r["ApplicationNumber"]] = r
                new += 1
        print(f"[{label}] {ipc}: 검색 결과 {total:,}건 중 {len(rows):,}건 수신, 새 특허 {new:,}건")

    all_rows = list(patents.values())
    OUT_PATH.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(KIPRIS_FIELDS)}개 분야 {len(all_rows):,}건 저장 → {OUT_PATH}")
    empty = {k: sum(1 for r in all_rows if not r[k]) for k in kipris.KEY_MAP}
    print("필드별 빈 값 개수:", {k: v for k, v in empty.items() if v})
    print("분야 분포:", dict(Counter(r["field"] for r in all_rows)))
    print("상태 분포:", dict(Counter(r["RegistrationStatus"] for r in all_rows)))


if __name__ == "__main__":
    main()
