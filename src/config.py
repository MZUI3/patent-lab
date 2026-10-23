# src/config.py — 품질에 영향을 주는 값은 전부 여기서 조정한다 (F2005, F2006, F2014)
import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트 (src/의 상위)

# 경로
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
# Chroma는 Windows에서 비ASCII 경로를 못 읽는 경우가 있어(store.chroma_path), 필요하면 .env의 CHROMA_DIR로 바꾼다
CHROMA_DIR = Path(os.getenv("CHROMA_DIR") or dotenv_values(ROOT / ".env").get("CHROMA_DIR") or ROOT / "chroma_db")
ENV_PATH = ROOT / ".env"

SEARCH_CACHE_DIR = RAW_DIR / "search"            # KIPRIS 검색 응답 캐시 (NF2002)
DETAIL_CACHE_DIR = RAW_DIR / "detail"            # KIPRIS 서지 상세(청구항) 캐시
WIKI_CACHE_DIR = RAW_DIR / "wiki"                # 위키백과 검색·본문 캐시
API_CALL_LOG = RAW_DIR / "api_calls.log"         # 외부 호출 기록 (KIPRIS 월 한도 추적)
PATENTS_JSON = PROCESSED_DIR / "patents_all.json"  # 항목 메타데이터 — 분야별 일괄 수집 (F2001 입력)
PATENTS_ONDEMAND_JSON = PROCESSED_DIR / "patents_ondemand.json"  # 검색할 때 추가로 받은 특허
COLLECT_LOG = PROCESSED_DIR / "collect_log.json"   # 항목별 수집한 검색어·문서 (F2004, NF2002)

# 컬렉션
META_COLLECTION = "patent_meta"      # RAG 1: 항목 메타데이터 (분야별 일괄 수집 + 검색 수집)
DOC_COLLECTION = "patent_docs"       # RAG 2: 공보(요약·청구항)·위키 청크

# KIPRIS 수집 — 분야별 (이름, IPC, 최대 건수). 최대 건수 None은 검색 결과 전체 (F2001)
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
KIPRIS_CALL_INTERVAL = 0.5           # 호출 간격(초) (NF2006)
KIPRIS_MONTHLY_LIMIT = 10000

# 외부 문서 수집 (F2003, F2004)
WIKI_CALL_INTERVAL = 0.5             # 호출 간격(초) (NF2006)
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
SEARCH_TYPE = "mmr"                  # "similarity" | "mmr" — 비교 결과로 MMR을 기본값으로 (docs/eval/f2014)
MMR_FETCH_K = 30                     # MMR: 후보를 넉넉히 가져온 뒤 TOP_K개를 다양성 기준으로 고른다 (F2014)
MMR_LAMBDA = 0.5                     # 1이면 유사도만, 0이면 다양성만

# 서비스 기능 (F2010, F2011)
BRIEF_TOP_K = 8                      # 청구항 해설에 쓸 외부 문서 근거 수
EXPLORE_CANDIDATES = 20              # 조건 탐색에서 메타데이터 색인으로 가져올 후보 수
EXPLORE_MAX_RESULTS = 8              # 조건 탐색 결과 최대 수
PRIOR_ART_CANDIDATES = 20            # 선행기술 조사에서 메타데이터 색인으로 가져올 후보 수
PRIOR_ART_MAX = 5                    # 청구항까지 받아 구성요소를 대비할 특허 수 (KIPRIS 호출 최대 5회)

# 모델 (.env에 GMS_KEY만 있으면 SSAFY GMS 프록시를 거친다)
GMS_BASE_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
PRIOR_ART_MODEL = "gpt-4.1"            # 선행기술 조사의 후보 선별·구성요소 대비 (미세한 판정이라 상위 모델)
