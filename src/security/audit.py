# src/security/audit.py — 감사 로그: 누가 언제 무엇을 했고, 어떤 자료가 어느 LLM으로 나갔나
#
# 한 줄에 JSON 하나(logs/audit.jsonl, 추가만 한다). 입력 원문·비밀번호·키는 남기지 않는다.
# 입력은 SHA-256 앞 12자와 글자 수만 남겨, 같은 입력이 반복되는지는 보이지만 내용은 복원되지 않게 한다
# (미공개 발명이 로그로 새지 않게). 운영 현황 화면과 사용자별 일일 한도가 이 파일을 읽는다.
#
# 주요 필드: ts · event · user · role · outcome(ok|blocked|error|fail|locked|limit)
#           level(보낸 자료의 최고 등급) · route(LLM 경로) · sent·withheld(문서 ID) · tokens · seconds
import hashlib
import json
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import AUDIT_LOG
from src.security.acl import User

KST = timezone(timedelta(hours=9))
LLM_EVENTS = ("prior_art", "qa", "brief", "explore", "internal_qa")   # 일일 한도에 세는 요청 (LLM을 부르는 기능)
_LOCK = threading.Lock()


def digest(text: str) -> dict:
    """입력 원문 대신 남길 값."""
    return {"input_sha": hashlib.sha256(text.encode()).hexdigest()[:12], "input_chars": len(text)}


def record(event: str, user: User | None = None, path: Path = AUDIT_LOG, **fields) -> dict:
    entry = {"ts": datetime.now(KST).isoformat(timespec="seconds"), "event": event,
             "user": user.username if user else "-", "role": user.role if user else "-", **fields}
    entry.setdefault("outcome", "ok")
    line = json.dumps(entry, ensure_ascii=False)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return entry


def read(path: Path = AUDIT_LOG) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:   # 쓰다가 끊긴 줄은 건너뛴다
            continue
    return out


def today() -> str:
    return datetime.now(KST).date().isoformat()


def count_today(username: str, path: Path = AUDIT_LOG) -> int:
    """오늘 이 사용자가 LLM 기능을 쓴 횟수 (차단·한도 초과로 LLM을 부르지 않은 요청은 세지 않는다)."""
    day = today()
    return sum(1 for e in read(path)
               if e.get("user") == username and e.get("event") in LLM_EVENTS
               and e.get("outcome") in ("ok", "error") and str(e.get("ts", "")).startswith(day))


def summarize(events: list[dict], day: str | None = None) -> dict:
    """운영 현황 화면용 집계. day를 주면 그날만."""
    if day:
        events = [e for e in events if str(e.get("ts", "")).startswith(day)]
    llm = [e for e in events if e.get("event") in LLM_EVENTS]
    by_user = defaultdict(lambda: Counter())
    for e in llm:
        c = by_user[e.get("user", "-")]
        c["요청"] += 1
        c[{"ok": "성공", "blocked": "차단", "error": "오류", "limit": "한도 초과"}.get(e.get("outcome"), "기타")] += 1
        c["토큰"] += int(e.get("tokens", 0) or 0)
    seconds = sorted(float(e["seconds"]) for e in llm if e.get("outcome") == "ok" and e.get("seconds") is not None)
    return {
        "requests": len(llm),
        "ok": sum(1 for e in llm if e.get("outcome") == "ok"),
        "blocked": sum(1 for e in llm if e.get("outcome") == "blocked"),
        "errors": sum(1 for e in llm if e.get("outcome") == "error"),
        "limit": sum(1 for e in llm if e.get("outcome") == "limit"),
        "login_fail": sum(1 for e in events if e.get("event") == "login" and e.get("outcome") in ("fail", "locked")),
        "tokens": sum(int(e.get("tokens", 0) or 0) for e in llm),
        "withheld": sum(len(e.get("withheld", []) or []) for e in llm),
        "p50": seconds[len(seconds) // 2] if seconds else None,
        "p95": seconds[min(len(seconds) - 1, int(len(seconds) * 0.95))] if seconds else None,
        "by_event": Counter(e.get("event") for e in llm),
        "by_route": Counter(e.get("route", "-") for e in llm if e.get("outcome") == "ok"),
        "by_user": {u: dict(c) for u, c in by_user.items()},
    }
