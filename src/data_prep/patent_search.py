# src/data_prep/patent_search.py — 검색할 때 KIPRIS 전 분야에서 특허를 받아 항목에 더한다
# 미리 받아 둔 분야(config.KIPRIS_FIELDS) 밖의 아이디어도 조사할 수 있게, 선행기술 조사 때 키워드로 KIPRIS를
# 검색해 새 특허를 patents_ondemand.json에 쌓는다. 같은 키워드는 캐시로 다시 호출하지 않는다 (NF2002).
import hashlib
import json

from src.config import (
    ONDEMAND_ROWS,
    ONDEMAND_FIELD,
    PATENTS_JSON,
    PATENTS_ONDEMAND_JSON,
    SEARCH_CACHE_DIR,
)
from src.data_prep import kipris


def _cached_search(op: str, params: dict, tag: str, query: str) -> tuple[int, list[dict]]:
    key = hashlib.sha1(f"{op}:{query}".encode("utf-8")).hexdigest()[:16]
    cache = SEARCH_CACHE_DIR / f"{tag}_{key}_rows{ONDEMAND_ROWS}.xml"
    if cache.exists():
        content = cache.read_bytes()
    else:
        content = kipris.call(op, {**params, "patent": "true", "utility": "false",
                                   "numOfRows": ONDEMAND_ROWS, "pageNo": 1}, detail=f"{tag}={query}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(content)   # 결과가 0건이어도 캐시한다 — 같은 키워드로 다시 호출하지 않게
    return kipris.parse_search(content)


def search_kipris(query: str) -> tuple[int, list[dict]]:
    """초록 키워드 검색(정밀) → 결과가 적으면 전체 텍스트 검색(넓게)으로 보충한다."""
    total, rows = _cached_search("getAdvancedSearch", {"astrtCont": query}, "abs", query)
    if total < 10:
        total2, rows2 = _cached_search("getWordSearch", {"word": query, "year": 0}, "word", query)
        seen = {r["ApplicationNumber"] for r in rows}
        rows += [r for r in rows2 if r["ApplicationNumber"] not in seen]
        total = max(total, total2)
    return total, rows


def _known_ids() -> set[str]:
    ids = set()
    for path in (PATENTS_JSON, PATENTS_ONDEMAND_JSON):
        if path.exists():
            ids |= {r["ApplicationNumber"] for r in json.loads(path.read_text(encoding="utf-8"))}
    return ids


def add_patents(rows: list[dict], query: str) -> list[dict]:
    """처음 보는 특허만 patents_ondemand.json에 더하고, 더한 행을 돌려준다."""
    known = _known_ids()
    new = []
    for r in rows:
        if r["ApplicationNumber"] and r["ApplicationNumber"] not in known:
            known.add(r["ApplicationNumber"])
            new.append({**r, "field": ONDEMAND_FIELD, "search_query": query})
    if new:
        saved = json.loads(PATENTS_ONDEMAND_JSON.read_text(encoding="utf-8")) if PATENTS_ONDEMAND_JSON.exists() else []
        PATENTS_ONDEMAND_JSON.parent.mkdir(parents=True, exist_ok=True)
        PATENTS_ONDEMAND_JSON.write_text(json.dumps(saved + new, ensure_ascii=False, indent=2), encoding="utf-8")
    return new
