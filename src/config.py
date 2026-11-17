# src/config.py — 품질에 영향을 주는 값은 전부 여기서 조정한다
import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트 (src/의 상위)
_ENV = dotenv_values(ROOT / ".env")


def _setting(name: str, default: str = "") -> str:
    """환경변수 → .env → 기본값. (env.py의 load_dotenv보다 먼저 읽히는 모듈이라 .env를 직접 본다)"""
    return os.getenv(name) or _ENV.get(name) or default


# 경로
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
# Chroma는 Windows에서 비ASCII 경로를 못 읽는 경우가 있어(store.chroma_path), 필요하면 .env의 CHROMA_DIR로 바꾼다
CHROMA_DIR = Path(_setting("CHROMA_DIR") or ROOT / "chroma_db")
ENV_PATH = ROOT / ".env"

SEARCH_CACHE_DIR = RAW_DIR / "search"            # KIPRIS 검색 응답 캐시
DETAIL_CACHE_DIR = RAW_DIR / "detail"            # KIPRIS 서지 상세(청구항) 캐시
WIKI_CACHE_DIR = RAW_DIR / "wiki"                # 위키백과 검색·본문 캐시
API_CALL_LOG = RAW_DIR / "api_calls.log"         # 외부 호출 기록 (KIPRIS 월 한도 추적)
PATENTS_JSON = PROCESSED_DIR / "patents_all.json"  # 항목 메타데이터 — 분야별 일괄 수집
PATENTS_ONDEMAND_JSON = PROCESSED_DIR / "patents_ondemand.json"  # 검색할 때 추가로 받은 특허
COLLECT_LOG = PROCESSED_DIR / "collect_log.json"   # 항목별 수집한 검색어·문서

# 컬렉션
META_COLLECTION = "patent_meta"      # RAG 1: 항목 메타데이터 (분야별 일괄 수집 + 검색 수집)
DOC_COLLECTION = "patent_docs"       # RAG 2: 공보(요약·청구항)·위키 청크

# KIPRIS 수집 — 분야별 (이름, IPC, 최대 건수). 최대 건수 None은 검색 결과 전체
# 반도체 테스트는 전체를, 다른 분야는 대표 IPC에서 일부를 미리 받고, 나머지는 검색할 때 받는다 (PATENTS_ONDEMAND_JSON)
KIPRIS_FIELDS = [
    ("반도체 테스트", "G01R 31/3183", None),
    ("반도체 소자", "H01L", 1000),
    ("전지", "H01M", 1000),
    ("컴퓨팅·데이터 처리", "G06F", 1000),
    ("인공지능", "G06N", 1000),
    ("디지털 통신", "H04L", 1000),
    ("의료 진단", "A61B", 1000),
    ("전기차", "B60L", 1000),
    ("재료 분석", "G01N", 1000),
]
ONDEMAND_FIELD = "검색 수집"         # 선행기술 조사 중 키워드 검색으로 받은 특허의 분야 이름
ONDEMAND_ROWS = 100                  # 검색 수집 1회에 받을 건수 — 많을수록 임베딩이 길어져 조사가 느려진다
KIPRIS_ROWS = 500                    # 페이지당 건수
KIPRIS_CALL_INTERVAL = 0.5           # 호출 간격(초)
KIPRIS_MONTHLY_LIMIT = 10000

# 외부 문서 수집
WIKI_CALL_INTERVAL = 0.5             # 호출 간격(초)
MAX_TERMS = 3                        # 프롬프트1이 뽑을 검색어 최대 수
WIKI_SEARCH_LIMIT = 5                # 검색어당 후보 문서 수
MAX_SELECTED_DOCS = 3                # 프롬프트2가 고를 문서 최대 수
DOC_MIN_CHARS = 200                  # 이보다 짧은 위키 문서는 색인하지 않는다
DOC_MAX_CHARS = 20000                # 긴 위키 문서는 앞부분만 쓴다 (특허 자체 문서는 예외)

# 청킹
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 검색
TOP_K = 5
SEARCH_TYPE = "mmr"                  # "similarity" | "mmr" — 비교 결과로 MMR을 기본값으로 (docs/experiments/mmr_vs_similarity.md)
MMR_FETCH_K = 30                     # MMR: 후보를 넉넉히 가져온 뒤 TOP_K개를 다양성 기준으로 고른다
MMR_LAMBDA = 0.5                     # 1이면 유사도만, 0이면 다양성만

# 서비스 기능
BRIEF_TOP_K = 8                      # 청구항 해설에 쓸 외부 문서 근거 수
EXPLORE_CANDIDATES = 20              # 조건 탐색에서 메타데이터 색인으로 가져올 후보 수
EXPLORE_MAX_RESULTS = 8              # 조건 탐색 결과 최대 수
PRIOR_ART_CANDIDATES = 20            # 선행기술 조사에서 메타데이터 색인으로 가져올 후보 수
PRIOR_ART_MAX = 5                    # 청구항까지 받아 구성요소를 대비할 특허 수 (KIPRIS 호출 최대 5회)

# ── LLM·임베딩 제공자 ─────────────────────────────────────────────────────────
# LLM은 OpenAI 호환 API 하나로 다룬다. .env의 키 종류로 제공자를 정한다 (env.openai_kwargs):
#   OPENAI_API_KEY → OpenAI 직접 / GEMINI_API_KEY → Google Gemini의 OpenAI 호환 주소 (무료 등급 있음)
#   / GMS_KEY → SSAFY GMS 프록시. LLM_PROVIDER로 직접 지정할 수도 있다.
# 임베딩은 "openai"(text-embedding-3-small) 또는 "local"(sentence-transformers, CPU, 비용·호출 없음).
# 임베딩 모델을 바꾸면 벡터가 달라지므로 색인을 처음부터 다시 만들어야 한다 (README '임베딩 모델 바꾸기').
GMS_BASE_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1"
OPENAI_BASE_URL = _setting("OPENAI_BASE_URL")   # 다른 OpenAI 호환 주소를 쓸 때만 (env.openai_kwargs)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_PROVIDER = _setting("LLM_PROVIDER") or (
    "openai" if _setting("OPENAI_API_KEY") else "gemini" if _setting("GEMINI_API_KEY") else "gms")
_DEFAULT_MODELS = {   # 제공자별 (기본 모델, 선행기술 조사용 상위 모델)
    "openai": ("gpt-4o-mini", "gpt-4.1"),
    "gms": ("gpt-4o-mini", "gpt-4.1"),
    # 2.5 세대는 신규 사용자에게 제공되지 않고, 3.5-flash는 무료 등급 일일 20회라 조사 3번이면 끝난다 (2026-10) → 둘 다 flash-lite
    "gemini": ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),
}
CHAT_MODEL = _setting("CHAT_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, _DEFAULT_MODELS["openai"])[0])
PRIOR_ART_MODEL = _setting("PRIOR_ART_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, _DEFAULT_MODELS["openai"])[1])
# 선행기술 조사의 후보 선별·구성요소 대비는 미세한 판정이라 상위 모델을 쓴다

EMBEDDING_PROVIDER = _setting("EMBEDDING_PROVIDER", "local" if LLM_PROVIDER == "gemini" else "openai")
EMBEDDING_MODEL = _setting("EMBEDDING_MODEL",
                           "BAAI/bge-m3" if EMBEDDING_PROVIDER == "local" else "text-embedding-3-small")
LOCAL_EMBEDDING_MAX_TOKENS = 1024    # 로컬 모델 입력 길이 상한 — 색인 텍스트(≤1,200자)·청크(1,000자)가 들어가는 크기

# 무료 등급은 분당·일당 요청 수 제한이 있어 429가 흔하다 → 호출 간격을 두고(LLM_RPM), 재시도를 늘리고, 병렬 호출을 줄인다
# Gemini 무료 등급은 모델별 분당 10~15회 (2026-10). 429의 "40초 후 재시도"는 SDK 백오프(최대 8초)로는 못 기다리므로 미리 간격을 둔다
LLM_RPM = int(_setting("LLM_RPM", "10" if LLM_PROVIDER == "gemini" else "0"))   # 분당 요청 상한, 0이면 제한 없음
LLM_MAX_RETRIES = int(_setting("LLM_MAX_RETRIES", "10" if LLM_PROVIDER == "gemini" else "2"))
# 요청 하나의 응답 대기 상한(초). SDK 기본값은 600초라, 응답 없이 멈춘 요청 하나가 조사·평가 전체를 10분씩 묶었다
LLM_TIMEOUT = float(_setting("LLM_TIMEOUT", "90"))
COMPARE_WORKERS = int(_setting("COMPARE_WORKERS", "1" if LLM_PROVIDER == "gemini" else str(PRIOR_ART_MAX)))

# ── 사내 문서 ─────────────────────────────────────────────────────────────────
# 원본 파일과 카탈로그(문서마다 등급·열람 주체·버전)를 분리한다. 카탈로그에 없는 파일은 색인하지 않는다.
INTERNAL_DIR = ROOT / "data" / "internal"
INTERNAL_CATALOG = INTERNAL_DIR / "catalog.yaml"
INTERNAL_MANIFEST = ROOT / "data" / "index" / "internal_manifest.json"   # 문서별 버전·지문·청크 수 (색인 현황)
INTERNAL_COLLECTION = "internal_docs"
INTERNAL_TOP_K = 5
INTERNAL_FETCH = 10                  # 권한 필터 검색으로 가져올 수 — 이 중 전송 가능한 것으로 TOP_K를 채운다
TABLE_ROWS_PER_CHUNK = 8             # 표(CSV)는 행 묶음마다 머리글을 다시 붙여 청크 하나만 봐도 열 이름이 보이게 한다

# ── 계정·감사 로그·전송 정책 ──────────────────────────────────────────────────
APP_ENV = _setting("APP_ENV", "local")           # local | production (production에서는 데모 계정으로 시작하지 않는다)
USERS_FILE = Path(_setting("USERS_FILE") or ROOT / "config" / "users.toml")
AUDIT_LOG = Path(_setting("AUDIT_LOG") or ROOT / "logs" / "audit.jsonl")
LOGIN_MAX_FAILURES = 5               # 연속 실패 횟수 → 잠금
LOGIN_LOCK_MINUTES = 15

# 기본 LLM 경로로 보낼 수 있는 최고 등급 (공개 < 내부 < 기밀). 비우면 제공자 약관 기준 기본값 (src/security/policy.py)
LLM_MAX_LEVEL = _setting("LLM_MAX_LEVEL")
# 사내 운영 모델 (선택) — Ollama·vLLM 같은 OpenAI 호환 서버. 기본 경로로 못 보내는 기밀 자료는 여기로만 보낸다
LOCAL_LLM_BASE_URL = _setting("LOCAL_LLM_BASE_URL")
LOCAL_LLM_MODEL = _setting("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")
LOCAL_LLM_MAX_LEVEL = _setting("LOCAL_LLM_MAX_LEVEL", "기밀")
