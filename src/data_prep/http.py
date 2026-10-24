# src/data_prep/http.py — 외부 API 호출 공통 규칙: 재시도·호출 간격·호출 기록·키 가림
import time

import requests

from src.config import API_CALL_LOG


def log_call(tag: str, detail: str) -> None:
    API_CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with API_CALL_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{tag}\t{detail}\n")


def get(url: str, *, params: dict, tag: str, detail: str, interval: float,
        headers: dict | None = None, secret: str | None = None, retries: int = 3) -> requests.Response:
    """GET을 최대 retries번 시도한다. 매 시도를 기록하고, 성공 후에는 interval만큼 쉰다.

    실패 메시지에는 요청 URL이 들어가므로 secret(API 키)을 가린 뒤 올린다.
    """
    last_err = None
    for attempt in range(retries):
        log_call(tag, f"{detail} attempt={attempt + 1}")
        try:
            res = requests.get(url, params=params, headers=headers, timeout=30)
            res.raise_for_status()
            time.sleep(interval)
            return res
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 ** attempt)
    msg = str(last_err)
    if secret:
        msg = msg.replace(secret, "***")
    raise RuntimeError(f"{tag} 호출 실패 ({detail}): {msg}")
