# src/data_prep/wiki.py — 한국어 위키백과 검색·본문 수집. 응답은 캐시해 다시 호출하지 않는다
import hashlib
import json
import re

from src.config import WIKI_CACHE_DIR, WIKI_CALL_INTERVAL, WIKI_SEARCH_LIMIT
from src.data_prep import http
from src.data_prep.docs import SourceDoc
from src.env import wiki_user_agent

API_URL = "https://ko.wikipedia.org/w/api.php"


def _api(params: dict, detail: str) -> dict:
    res = http.get(API_URL, params={**params, "format": "json", "formatversion": 2},
                   headers={"User-Agent": wiki_user_agent()},
                   tag="wiki", detail=detail, interval=WIKI_CALL_INTERVAL)
    return res.json()


def _cached(path, fetch):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def search(term: str) -> list[dict]:
    """검색어의 후보 문서 [{pageid, title, snippet}]. 선별 프롬프트에 보여줄 제목·요약만 받는다."""
    key = hashlib.sha1(term.encode("utf-8")).hexdigest()[:16]

    def fetch():
        data = _api({"action": "query", "list": "search", "srsearch": term,
                     "srlimit": WIKI_SEARCH_LIMIT, "srprop": "snippet"}, detail=f"search={term}")
        return {"term": term, "results": [
            {"pageid": h["pageid"], "title": h["title"],
             "snippet": re.sub(r"<[^>]+>", "", h.get("snippet", ""))}
            for h in data["query"]["search"]
        ]}

    return _cached(WIKI_CACHE_DIR / "search" / f"{key}.json", fetch)["results"]


def page(pageid: int, search_term: str = "") -> SourceDoc:
    """선별된 문서의 본문(평문)과 원문 URL."""
    def fetch():
        data = _api({"action": "query", "prop": "extracts|info", "explaintext": 1,
                     "inprop": "url", "pageids": pageid}, detail=f"page={pageid}")
        p = data["query"]["pages"][0]
        return {"pageid": pageid, "title": p["title"], "url": p["fullurl"], "text": p.get("extract", "")}

    p = _cached(WIKI_CACHE_DIR / "pages" / f"{pageid}.json", fetch)
    return SourceDoc(doc_id=f"wiki-{pageid}", title=p["title"], url=p["url"],
                     source_type="wikipedia", text=p["text"], search_term=search_term)
