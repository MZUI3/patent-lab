# 검증 결과 (QA)

- 점검일: 2026-10-22, 실데이터(특허 10,284건, 외부 문서를 수집한 특허 7건) 기준. LLM·임베딩은 당시 설정(gpt-4o-mini / gpt-4.1, text-embedding-3-small).
- 확인 방법
  - **자동**: `python -m src.eval.verify` — 실제 색인·파이프라인·LLM으로 실행하고, 화면 동작은 Streamlit AppTest로 확인. 결과 원문은 [eval/verification_run.md](eval/verification_run.md). 점검 이름(`index-count`, `collect-no-repeat` …)으로 일부만 다시 돌릴 수 있다.
  - **브라우저**: `streamlit run src/ui/app.py`로 띄운 앱을 직접 조작.
  - **문서**: [experiments/](experiments/)의 실행 결과.
- 그 뒤의 변경(무료 스택 전환, 자동 평가)은 [eval/prior_art_eval.md](eval/prior_art_eval.md)의 실험 요약표로 추적한다.

## 요약
| 영역 | 항목 수 | 통과 | 실패 | 미확인 |
|---|---|---|---|---|
| 데이터와 색인 | 5 | 5 | 0 | 0 |
| 문서 수집과 색인 | 5 | 5 | 0 | 0 |
| 답변과 근거 | 6 | 6 | 0 | 0 |
| 서비스 기능과 화면 | 6 | 6 | 0 | 0 |
| 실험 (MMR · RAG 전후 · 출처 결합) | 3 | 3 | 0 | 0 |
| 운영 품질 (성능·키·오류 노출·구조·외부 호출·출처) | 7 | 7 | 0 | 0 |

## 데이터와 색인
| 점검 | 확인 항목 | 결과 | 방법 | 근거 |
|---|---|---|---|---|
| index-count | 로드한 항목 수와 색인된 청크 수가 일치한다 | 통과 | 자동 | 원본 10,284건 → 항목 10,284건(분야별 일괄 8,747 + 검색 수집 1,537), `patent_meta` 청크 10,284개 |
| index-idempotent | 색인을 두 번 실행해도 청크 수가 늘지 않는다 | 통과 | 자동 | 연속 2회 모두 "변경 없음 10,284", 청크 10,284 → 10,284. 재구축(`--reset`) 직후 재실행도 "변경 없음" |
| index-text-fields | 항목 1건의 색인 텍스트에 의도한 속성이 모두 들어간다 | 통과 | 자동 | 7개 속성(발명의 명칭·출원인·출원번호·출원일·IPC·등록 상태·요약) 모두 포함 |
| index-vs-bruteforce | 색인 검색 결과가 전수 계산과 일치한다 (인덱스 손상 점검) | 통과 | 자동 | "스캔 체인 고장 진단 기계 학습" 상위 10건: 색인 검색과 전수 계산 공통 10/10. Windows 한글 경로로 인한 인덱스 손상을 겪은 뒤 추가 ([결정 기록](decisions/004-index-snapshot.md)) |
| snapshot-restore | 저장소의 색인 스냅샷만으로 다른 PC에서 같은 색인을 만들 수 있다 | 통과 | 자동 | 빈 폴더에 복원 18초, API 호출 없음. 복원본의 검색 상위 10건이 원본과 순서까지 같음 |

## 문서 수집과 색인
| 점검 | 확인 항목 | 결과 | 방법 | 근거 |
|---|---|---|---|---|
| collect-terms | 하나의 질문에 대해 뽑힌 검색어 목록 | 통과 | 자동 | "다목적 메모리를 지닌 테스터 시스템", 질문 "자동으로 생성하는 회로는 무엇이고 메모리 자체 시험은 왜 필요한가?" → `['자동 시험 패턴 생성', '내장 자체시험', '집적 회로']` |
| collect-select | 선별로 색인된 문서와 버려진 문서 구분 | 통과 | 자동 | 색인: 자동 시험 패턴 생성, 자동 시험 장비, 내장 자체시험 / 버림 11건: 인공지능, 엘림넷, 부하 테스트, BITS, 녹내장, 건축공학 … 새 문서 4건·청크 28개, 외부 호출 3회 |
| collect-no-repeat | 같은 질문을 다시 물으면 수집이 다시 일어나지 않는다 | 통과 | 자동 | 재질문: 지난번 검색어 재사용(검색어 추출 호출도 생략), 새 문서 0건, 외부 호출 0회. 한 번은 같은 질문에 다른 검색어가 나와 재수집이 일어났고, 질문별 검색어 재사용으로 고쳤다 |
| collect-own-document | 선택한 특허 자체를 다룬 문서가 빠지지 않는다 | 통과 | 자동 | 문서를 수집한 특허 9건 모두 KIPRIS 공보 청크 보유. 공보는 선별·길이 규칙 없이 항상 수집 |
| chunk-size-effect | 청크 크기·중복 길이를 바꾸면 청크 수가 달라진다 | 통과 | 자동 | 전체 재색인(`--rebuild`, 외부 호출 없음): 500/100 → 587청크, 1,000/200 → 272청크. 점검 후 원래 설정으로 복원 |

## 답변과 근거
| 점검 | 확인 항목 | 결과 | 방법 | 근거 |
|---|---|---|---|---|
| topk-effect | Top-k를 바꾸면 같은 질문의 근거 목록이 달라진다 | 통과 | 자동 | 지터 특허 "적분 회로와 비교기의 역할": k=3 → 근거 4개, k=10 → 근거 11개 |
| graph-nodes | LangGraph가 retrieve와 generate 두 노드로 구성된다 | 통과 | 자동 | 노드 `__start__, retrieve, generate, __end__`. `python -m src.rag.graph --show-graph` (Mermaid) |
| citations-match | 답변 본문의 근거 번호와 화면의 근거 목록이 일치한다 | 통과 | 자동 + 브라우저 | 본문 번호 {1} = 표시 근거 {1}, 검색 근거 6개 중. 브라우저에서도 [근거 1]과 "근거 문서 1개" 일치 |
| source-types | 근거 목록에 메타데이터와 외부 문서가 구분되어 표시된다 | 통과 | 자동 + 브라우저 | 출처 유형: 특허 메타데이터 / KIPRIS 특허공보(청구항) / 한국어 위키백과. 목록 머리에 종류별 개수 |
| not-found-answer | 문서에 없는 것을 물으면 추측하지 않고 확인되지 않는다고 답한다 | 통과 | 자동 | "이 회로의 양산 단가와 삼성전자 제품 적용 시점은?" → "제공된 문서에서 확인되지 않습니다." 인용 0개 |
| no-citation-notice | 인용된 근거가 하나도 없으면 화면에 안내된다 | 통과 | 자동 | "답변에 인용된 근거가 없습니다. 문서에서 확인된 내용이 아닐 수 있습니다." (AppTest) |

## 서비스 기능과 화면
| 점검 | 확인 항목 | 결과 | 방법 | 근거 |
|---|---|---|---|---|
| chat-isolation | 특허를 바꿨을 때 이전 대화가 섞이지 않는다 | 통과 | 자동 | 지터 특허에서 질문 → 메시지 2개, 다른 특허 선택 → 0개, 복귀 → 2개 |
| explore-conditions | 조건 탐색 결과에 조건과 맞지 않는 특허가 섞이지 않는다 | 통과 | 자동 + 브라우저 | 조건 3개, 연도·상태·출원인 위반 0건: "2015년 이후 … 스캔 체인 고장 진단 … 등록" → 3건(2020~2022, 모두 등록) 등 |
| brief-evidence · prior-art-quotes | 선행기술 조사·청구항 해설 결과에 근거가 함께 제시된다 | 통과 | 자동 + 브라우저 | 선행기술 조사: 스캔 체인 ML 진단 아이디어 → 1위 "스캔 셀의 고장 검출 방법"(겹침 70%), 개시·일부 판정 모두 청구항 원문 구절 확인 / 청구항 해설: 12/12줄 인용 |
| (브라우저) | 수집·생성 중 진행 상태가 표시된다 | 통과 | 브라우저 | 선행기술 조사 5단계 상태, 질문 "수집·검색 중 → 답변 생성 중" |
| missing-index-and-key | 색인이 없는 상태와 키가 없는 상태에서 안내가 표시된다 | 통과 | 자동 | 색인 없음 → 색인 명령 안내 / 키 없음 → `.env` 안내 |
| disclaimer | 법적 판단과 맞닿은 도메인의 면책 안내가 표시된다 | 통과 | 자동 + 브라우저 | 사이드바 상시 + 선행기술 조사 결과 위("신규성·진보성 판단이 아님") |

## 실험
| 실험 | 결과 | 문서 |
|---|---|---|
| 유사도 vs MMR 근거 목록 | k=5에서 근거 출처 문서 수 2→5, 1→3, 2→3 / 청크 간 겹침 0.649→0.338, 0.866→0.529, 0.658→0.507. MMR을 질의응답 기본값으로 | [experiments/mmr_vs_similarity.md](experiments/mmr_vs_similarity.md) |
| RAG 전후 답변 품질 | 대표 질문 6개: 사실 정확성 0/4 → 4/4, 근거 제시 0 → 4건 모두 원문 링크, 모름 응답 2/2 → 2/2 | [experiments/rag_vs_norag.md](experiments/rag_vs_norag.md) |
| 여러 출처 결합 | 근거마다 출처 유형 태그, 목록 머리에 종류별 개수. 출처 간 분량 불균형 발견 | [experiments/multi_source.md](experiments/multi_source.md) |
| 선행기술 조사 자동 평가 (전환 뒤) | 40사례: Recall@20 88%, Recall@5 85%, 개시율 96% | [eval/prior_art_eval.md](eval/prior_art_eval.md) |

## 운영 품질
| 점검 | 항목 | 결과 | 근거 |
|---|---|---|---|
| (측정) | 응답 시간 | 통과 | 질의응답: 수집 생략 4.1초, 이미 수집한 특허 5.0~5.9초, 처음 질문하는 특허 11.8~22초 / 선행기술 조사: 키워드 캐시 16초, 새 키워드 45~50초. 오래 걸리는 단계는 진행 상태로 표시 |
| collect-no-repeat | 수집 캐시 | 통과 | 재질문 시 외부 호출 0회. KIPRIS 검색 재실행도 캐시로 호출 0회 |
| no-keys-in-repo | 키 관리 | 통과 | 키 값 검색: 추적 파일(수집 데이터·응답 캐시·스냅샷 포함)·전체 커밋 이력 0건, `.env`·`chroma_db` 추적 0개 |
| no-internal-leak | 내부 정보 비노출 | 통과 | 키·"Traceback"이 담긴 내부 오류를 일으켜도 화면에는 "처리 중 오류가 발생했습니다"만 표시. 예상하지 못한 오류도 `.streamlit/config.toml`의 `showErrorDetails = "none"` |
| module-layout | 모듈 분리 | 통과 | `data_prep / vectorstore / rag / service / ui / eval`. 외부 API 주소는 `data_prep/kipris.py`·`wiki.py`에만. 하위 모듈이 UI를 가져다 쓰지 않음 |
| http-etiquette | 외부 호출 규칙 | 통과 | 2회 실패 후 성공: 시도 3회, 1·2초 대기, 성공 후 0.5초 간격, 시도마다 기록. 3회 모두 실패하면 키를 `***`로 가린 메시지. 위키 User-Agent에 연락처 |
| (문서) | 출처와 이용 조건 | 통과 | [README 데이터 출처](../README.md#데이터-출처와-이용-조건), [requirements.md 5절](requirements.md). 위키 근거마다 제목·원문 URL |

## 실행 화면 증거
화면은 `docs/screenshots/`, 터미널 실행 결과는 [eval/cli_outputs.md](eval/cli_outputs.md), 자동 점검 원문은 [eval/verification_run.md](eval/verification_run.md).

| 확인 내용 | 증거 |
|---|---|
| 색인 건수 상시 표시 | [01_home.png](screenshots/01_home.png) |
| 항목 수 = 청크 수, 재실행 시 유지, 색인 텍스트 1건 | [cli_outputs.md](eval/cli_outputs.md) |
| 다른 PC 첫 실행: 빈 색인 → 스냅샷 자동 복원 | [11_snapshot_restore.png](screenshots/11_snapshot_restore.png) → [12_snapshot_restored.png](screenshots/12_snapshot_restored.png) |
| 검색어 추출, 색인/버림 문서, 재질문 시 수집 없음 | [cli_outputs.md](eval/cli_outputs.md) |
| 청크 크기 변경 시 청크 수 변화 | [verification_run.md chunk-size-effect](eval/verification_run.md) |
| LangGraph retrieve → generate 구조, Top-k 3/10 차이 | [cli_outputs.md](eval/cli_outputs.md) |
| 답변의 [근거 n] ↔ 근거 목록 일치, 출처 유형 구분 | [06_qa_answer_evidence.png](screenshots/06_qa_answer_evidence.png) |
| 문서에 없는 질문 → "확인되지 않습니다" | [07_qa_not_found.png](screenshots/07_qa_not_found.png) |
| 선행기술 조사 진행 상태 / 질문 진행 상태 | [02_prior_art_progress.png](screenshots/02_prior_art_progress.png), [05_qa_progress.png](screenshots/05_qa_progress.png) |
| 선행기술 조사 결과: 구성요소, 겹침 비율, 판정 배지 | [03_prior_art_result.png](screenshots/03_prior_art_result.png) |
| 판정의 근거: 청구항 원문 구절 | [04_prior_art_evidence.png](screenshots/04_prior_art_evidence.png) |
| 청구항 해설: 줄마다 근거 번호 | [08_claim_brief.png](screenshots/08_claim_brief.png) |
| 조건 탐색 → 특허 패널 | [09_find_condition.png](screenshots/09_find_condition.png) |
| 이름·번호로 특허 선택 | [10_find_by_name.png](screenshots/10_find_by_name.png) |
| 대화 분리 | [verification_run.md chat-isolation](eval/verification_run.md) |
| 키 누락 안내 | [13_missing_key.png](screenshots/13_missing_key.png) |
| 면책 안내 | [01_home.png](screenshots/01_home.png), [03_prior_art_result.png](screenshots/03_prior_art_result.png) |
| 오류 시 안내만 표시 | [verification_run.md no-internal-leak](eval/verification_run.md) |
