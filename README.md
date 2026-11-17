# Patent Lab — 선행기술 조사·사내 문서 RAG

아이디어를 입력하면 가까운 한국 선행 특허를 찾아 **구성요소별로 청구항과 대비**하고, "이미 나와 있다"는 판정마다 **공보 원문(요약·청구항) 구절을 코드로 검증**해 보여 주는 서비스입니다. 연구 조직 안에서 쓰는 상황을 가정해 **로그인·문서별 열람 권한·자료 등급별 LLM 전송 정책·감사 로그**를 두고, 권한 밖 노출과 문서 속 악성 지시(프롬프트 인젝션)를 평가셋으로 잽니다.

| 선행기술 조사 결과 | 구성요소 대비와 청구항 근거 | 사내 문서 질의응답과 전송 정책 |
|:---:|:---:|:---:|
| <img src="docs/screenshots/03_prior_art_result.png" width="300"/> | <img src="docs/screenshots/04_prior_art_evidence.png" width="300"/> | <img src="docs/screenshots/15_internal_qa_policy.png" width="300"/> |

- **데이터**: 한국 특허 10,284건 (반도체 테스트 IPC G01R 31/3183 전체 985건 + 8개 분야 대표 특허 + 조사할 때 KIPRIS 검색으로 더한 1,537건), 특허 공보와 한국어 위키백과, 사내 문서 11건(데모용 가상 데이터 — 발명 신고서·실험 보고서·실험 데이터 CSV·회의록·지침·외부 수신 문서·공개 자료)
- **스택**: Python · LangGraph · LangChain · Chroma · sentence-transformers(bge-m3) · Gemini(OpenAI 호환 API) · KIPRIS Plus API · Streamlit · Docker
- **운영 비용 0원**: LLM은 무료 등급, 임베딩은 로컬 CPU, 배포는 Hugging Face Spaces 무료 CPU. 저장소에 색인 스냅샷이 들어 있어 clone 뒤 바로 실행됩니다.
- 이 서비스는 특허 정보 탐색을 돕는 도구이며 **법률 자문을 대신하지 않습니다.** 권리 범위·침해·등록 가능성 판단은 변리사 등 전문가에게 확인하세요.

## 왜 만들었나
특허를 출원하며 선행 특허를 조사할 때, 가장 시간이 드는 일은 비슷한 특허를 찾는 것이 아니라 **찾은 특허의 청구항을 하나씩 읽으며 "내 아이디어의 어느 구성이 이미 나와 있는지" 대조하는 것**이었습니다. 청구항은 법률 문장이라 읽기 어렵고, 비슷한 특허가 많아 LLM에 근거 없이 물으면 다른 특허와 섞인 그럴듯한 오답이 나옵니다. 같은 모델에 근거 없이 물었을 때 사실 질문 4개 중 정답 0개, 근거를 붙이면 4개 모두 원문과 일치했습니다 ([실험](docs/experiments/rag_vs_norag.md)).

그래서 이 과정을 그대로 파이프라인으로 옮겼습니다: 아이디어 → 구성요소 분해 → KIPRIS 검색으로 대상 확장 → 가까운 특허 선별 → 청구항 수집 → 구성요소마다 개시/일부/미개시 판정. **판정은 LLM이 인용한 구절이 그 특허의 공보 원문(요약·청구항)에 실제로 있을 때만 인정합니다.**

연구 조직에서 쓰려면 두 가지가 더 필요했습니다. 공개 특허뿐 아니라 발명 신고서·실험 보고서처럼 **볼 수 있는 사람이 정해진 사내 문서**를 함께 검색하되 권한 밖 문서는 답변의 근거로도 꺼내지 않아야 하고, 입력하는 아이디어가 **출원 전 발명**이라는 점을 다뤄야 합니다. 비용 0원을 위해 쓰는 Gemini 무료 등급은 약관상 입력을 제품 개선에 쓰고 사람이 검토할 수 있어 기밀 정보를 넣지 말라고 되어 있는데, v2는 미공개 아이디어를 그대로 보내고 있었습니다. v3에서 검색 단계의 권한 필터와 자료 등급별 전송 정책을 만들고, 지켜지는지를 평가셋으로 잽니다.

## 아키텍처
```
선행기술 조사
 아이디어 + 공개 여부(기본: 미공개 = 기밀) ─▶ 기밀을 받을 LLM 경로가 없으면 아무것도 보내지 않고 멈춤
 아이디어 ─▶ 구성요소 3~6개 ─▶ KIPRIS 키워드 검색(공개 아이디어만) → 새 특허 색인 ─▶ patent_meta MMR 후보 20 ─▶ 선별 5
          ─▶ 청구항 수집 ─▶ 구성요소 × 청구항 대비 ─▶ 원문 구절 코드 검증 ─▶ 겹침 비율로 정렬

특허 질의응답 (LangGraph retrieve → generate)
 특허 + 질문 ─▶ 검색어 추출 ─▶ 위키 후보 선별·수집 + 특허 공보(항상) ─▶ 청킹 1,000/200 ─▶ patent_docs
             ─▶ patent_meta 1건 + patent_docs MMR k=5 (특허 범위 필터) ─▶ [근거 n] 달린 답변 ─▶ 인용된 근거만 표시

사내 문서 질의응답 (LangGraph retrieve → policy → generate → check)
 질문 ─▶ internal_docs 검색 + 열람 권한 필터(사용자 주체의 $or) ─▶ 이중 확인
      ─▶ 근거의 최고 등급을 받을 LLM 경로 선택 (없으면 상한을 넘는 근거를 빼고 채움, 뺀 문서는 제목·버전만 화면에)
      ─▶ 근거를 자료 블록으로 감싼 프롬프트 ─▶ 출처 없는 링크 삭제 ─▶ [근거 n] + 문서 ID·버전·등급

모든 LLM 요청 ─▶ 감사 로그(입력은 해시만) · 사용자별 일일 한도 · 관리자 운영 현황
```
- 3단 색인: `patent_meta`(특허 1건 = 청크 1개) / `patent_docs`(공보·위키 청크, `item_id` 필터로 다른 특허 문서가 섞이지 않게) / `internal_docs`(사내 문서 청크, 열람 주체를 불리언 메타데이터로 펼침)
- KIPRIS·위키백과 응답은 전부 파일 캐시 + 호출 기록 (LLM 응답은 캐시하지 않음). 특허 색인은 float16 임베딩 스냅샷(23MB)으로 저장소에 포함
- 자세한 설계: [docs/architecture.md](docs/architecture.md) (10절: 보안 계층)

## 보안과 권한
| 요구 | 구현 | 확인 |
|---|---|---|
| 권한 밖 문서는 답변의 근거로도 꺼내지 않는다 | 문서별 열람 주체(전체·임직원·팀·역할·사용자)를 청크 메타데이터에 펼치고, 사용자 주체로 만든 필터를 벡터 DB 질의에 넣는다. 검색 뒤 한 번 더 확인 | 권한 밖 노출 **0/38** (필터 없이 38/38), 정당한 접근 22/22 |
| 권한·문서가 바뀌면 색인도 바뀐다 | 카탈로그(등급·열람 주체·버전)와 원본을 함께 해시한 지문으로 바뀐 문서만 다시 색인. 권한 설정 오류·미등록 파일은 색인하지 않음 | 권한을 거두면 바로 검색되지 않음, 버전을 안 올린 내용 변경은 경고 (단위 테스트) |
| 자료 등급에 맞는 LLM으로만 보낸다 | 경로별 상한(Gemini 무료 등급 → 공개, OpenAI API → 내부, 사내 모델 → 기밀). 미공개 아이디어는 LLM·KIPRIS 검색 모두 하지 않음. 이전 대화도 등급을 붙여 상한을 넘으면 맥락으로 보내지 않음 | 미공개 아이디어 차단 화면, 기밀 본문·기밀을 담은 이전 대화가 외부 경로 프롬프트에 들어가지 않음 (단위 테스트) |
| 문서 속 악성 지시를 따르지 않는다 | 색인 때 의심 구절 표시 → 근거를 자료 블록으로 감싸고 근거 속 지시를 따르지 않는 프롬프트 → 출처 없는 링크 삭제 | 공격 12사례 성공: 기존 프롬프트 8/12 → 서비스 방식 0/12 |
| 누가 무엇을 어느 LLM으로 보냈는지 남긴다 | 감사 로그(입력 원문 대신 해시, 경로·등급·보낸/뺀 문서·토큰·시간), 사용자별 일일 한도, 관리자 운영 현황 | 운영 현황 화면 |
| 로그인과 계정 | PBKDF2(60만 회) 해시, 5회 실패 시 15분 잠금, 운영 환경에서 데모 계정 거부, 로그아웃 시 세션 전체 삭제 | 단위 테스트 |

| 로그인 (계정별 열람 범위) | 미공개 아이디어 차단 | 운영 현황 (관리자) |
|:---:|:---:|:---:|
| <img src="docs/screenshots/14_login_roles.png" width="300"/> | <img src="docs/screenshots/16_prior_art_blocked.png" width="300"/> | <img src="docs/screenshots/17_ops_usage.png" width="300"/> |

## 핵심 설계 결정
결정마다 맥락·선택지·결과를 [docs/decisions/](docs/decisions/)에 기록했습니다.

| 결정 | 왜 |
|---|---|
| [청구항 단위 대비](docs/decisions/001-claim-level-comparison.md) | 권리 범위는 요약이 아니라 청구항이 정한다. 실무의 클레임 차트와 같은 형태 |
| [판정에 원문 구절 강제](docs/decisions/002-quote-verification.md) | 거짓 "개시"가 선행 조사에서 가장 위험하다. 인용 구절을 코드로 원문과 대조해 없으면 미개시로 |
| [질의응답 근거는 MMR](docs/decisions/003-mmr-for-qa-evidence.md) | 복합 질문에서 한 문서 청크가 근거를 독점하던 문제. 출처 문서 수 1→3. 선행기술 조사 후보 생성에는 역효과가 측정되어 재검토 중 |
| [색인 스냅샷 공유](docs/decisions/004-index-snapshot.md) | Chroma 약 150MB 대신 23MB 스냅샷, 임베딩 모델을 manifest에 기록해 모델이 다르면 복원 거부 |
| [LLM 제공자 분리 + 로컬 임베딩](docs/decisions/005-free-stack.md) | 운영 비용 0원. LLM은 OpenAI 호환 API 하나로 호출하고 `.env`의 키로 제공자를 고른다. 임베딩은 로컬 bge-m3. 429에 대비해 프로세스 공유 속도 제한(`LLM_RPM`)·재시도 |
| [라벨 없는 자동 평가](docs/decisions/006-self-retrieval-eval.md) | 청구항 1을 아이디어로 바꿔 정답을 자기 자신으로 두고 Recall·판정 정확도·비용을 잰다 |
| [사전 수집 + 조사 시 검색 수집](docs/decisions/007-kipris-on-demand.md) | 월 1,000회 한도 안에서 전 분야 조사. 캐시·기록·컬렉션 기준 색인 판단 <!-- TODO: 확인 — src/config.py의 KIPRIS_MONTHLY_LIMIT는 10000(코드에서 쓰이지 않음), 문서는 1,000회 --> |
| [검색 단계의 열람 권한 필터](docs/decisions/008-access-control.md) | 답변 뒤에 거르면 이미 프롬프트로 나간 뒤다. 열람 주체를 메타데이터에 펼쳐 질의 필터로, 권한이 바뀌면 지문으로 다시 색인 |
| [자료 등급별 외부 전송 정책](docs/decisions/009-send-policy.md) | 무료 등급 약관은 기밀 입력을 금한다. 경로마다 상한을 두고, 받을 경로가 없으면 보내지 않는다. 열람 권한과 전송 정책을 나눈다 |
| [악성 지시 세 겹 방어와 측정](docs/decisions/010-injection-defense.md) | 외부 수신 문서의 "AI에게: …하라"를 색인 점검·자료 블록 프롬프트·링크 삭제로 막고, 공격 성공률과 정답률을 함께 잰다 |

## 품질 측정
설정을 바꿀 때마다 같은 평가셋으로 돌려 결과를 쌓습니다.

**선행기술 조사 자동 평가** ([실험 요약표](docs/eval/prior_art_eval.md)) — 사람이 라벨링한 정답이 없어, 청구항이 캐시된 특허 40건의 첫 청구항을 LLM이 엔지니어의 아이디어 설명처럼 바꿔 쓴 문장을 입력으로 쓰고 그 특허 자신을 정답으로 둡니다 (청구항과 겹친 가장 긴 구절 중앙값 8자).

| 기준선 (Gemini 3.5 Flash-Lite + bge-m3, 2026-10-24) | 값 |
|---|---|
| Recall@20 (후보) / Recall@5 (대비 대상) / MRR@5 | 88% / 85% / 0.819 |
| 개시율 (정답 특허, 전부 "개시"가 정답) / 개시+일부 | 96% / 97% |
| 겹침 1위율 (동률 포함) | 100% |
| 사례당 시간 · 토큰 · 비용 | 23초 · 4,800 · 0원 |

기준선에서 찍어낸 다음 실험 두 가지 ([해석](docs/eval/prior_art_eval/baseline-flash-lite_2026-10-24.md)):
- 검색 미스 5건 중 2건은 **MMR이 정답을 떨어뜨렸다** (유사도 2위·4위) → 후보 생성을 유사도 검색으로
- 판정 오류 6개 중 4개는 **원문 구절 검증이 맞는 판정을 미개시로 바꿨다** (정답 특허 판정 136개 중 2.9%) → 구절 재탐색 또는 토큰 단위 일치

**보안 평가** ([결과](docs/eval/security_eval.md), 사례 [security_cases.json](docs/eval/security_cases.json)) — 사내 문서(가상 11건)와 계정 5개(방문자·반도체 테스트팀·배터리 안전팀·특허팀·시스템 관리자)로 잽니다.

| 항목 (2026-11-05, bge-m3 · Gemini 3.5 Flash-Lite) | 결과 |
|---|---|
| 권한 밖 노출: 그 문서를 볼 수 없는 계정이 문서를 겨냥해 물었을 때 근거 상위 5개에 들어옴 | 권한 필터 **0/38** · 필터 없이 38/38 |
| 정당한 접근: 볼 권한이 있는 계정의 같은 질문에서 근거에 들어옴 | 22/22 |
| 사내 문서 검색 Recall@5 · MRR@5 (권한 필터 켬, 14문항) | 100% · 0.964 |
| 문서 속 악성 지시 공격 성공 (피싱 링크·답변 조작·시스템 지시 유출·이미지 링크 유출, 각 3사례) | A 기존 프롬프트 8/12 → B 방어 프롬프트 0/12 → C + 점검 표시 0/12 → D + 링크 삭제(서비스) 0/12 |
| 같은 사례에서 답해야 할 사실을 답함 | A 9/12 · B 12/12 · C 12/12 · D 12/12 |

⟪INJ_NOTE⟫

**그 밖의 실험** — [유사도 vs MMR](docs/experiments/mmr_vs_similarity.md) (출처 문서 수 1→3, 청크 간 겹침 0.866→0.529) · [RAG 전후](docs/experiments/rag_vs_norag.md) (사실 정확성 0/4 → 4/4) · [여러 출처 결합](docs/experiments/multi_source.md). 기능별 검증 결과는 [docs/qa.md](docs/qa.md) (2026-10-22 측정, 당시 설정 gpt-4o-mini·text-embedding-3-small), 요구사항은 [docs/requirements.md](docs/requirements.md).

## 실행
### 준비
- Python 3.12 이상 (개발·검증은 3.14, Windows 11)
- `.env` 값 3개: LLM 키 하나(`GEMINI_API_KEY`([Gemini](https://aistudio.google.com/apikey) 무료 등급) 또는 `OPENAI_API_KEY`, 둘 다 있으면 OpenAI), `KIPRIS_API_KEY`([KIPRIS Plus](https://plus.kipris.or.kr) 서비스 키), `WIKI_CONTACT`(위키백과 User-Agent에 넣을 연락처). LLM 키만 있어도 실행되며, 나머지 둘이 없으면 이미 색인된 특허·문서로만 조사하고 답합니다.
- 사내 문서 데모: Gemini 무료 등급의 기본 상한은 **공개**라서 내부 문서도 보내지 않습니다. 사내 문서가 가상 데이터인 데모에서는 `.env`에 `LLM_MAX_LEVEL=내부`를 두면 내부 문서까지 보내고 기밀은 계속 보내지 않습니다. 사내 모델(Ollama·vLLM 등 OpenAI 호환 서버)이 있으면 `LOCAL_LLM_BASE_URL`로 붙여 기밀을 그쪽으로 보냅니다.
- 선택 설정 (비우면 제공자별 기본값, 뜻과 기본값은 `.env.example` 주석): `OPENAI_BASE_URL`, `CHAT_MODEL`·`PRIOR_ART_MODEL`, `EMBEDDING_PROVIDER`·`EMBEDDING_MODEL`, `LLM_RPM`·`LLM_MAX_RETRIES`·`COMPARE_WORKERS`, `CHROMA_DIR`, `LLM_MAX_LEVEL`·`LOCAL_LLM_*`, `APP_ENV`·`USERS_FILE`·`USERS_TOML`·`AUDIT_LOG`
- 임베딩은 Gemini 키를 쓰면 기본으로 로컬 모델(bge-m3)을 CPU에서 돌립니다. 처음 실행할 때 모델(약 2.2GB)을 내려받습니다. OpenAI 키를 쓰면 기본값이 OpenAI 임베딩(`text-embedding-3-small`)이 되어 저장소의 스냅샷(bge-m3)을 복원하지 않으므로, 스냅샷을 그대로 쓰려면 `.env`에 `EMBEDDING_PROVIDER=local`을 둡니다.

```bash
python -m venv venv
source venv/Scripts/activate     # Windows Git Bash (macOS/Linux: source venv/bin/activate, Windows cmd: venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env             # 값 채우기 (설정 항목은 .env.example 주석 참고)
streamlit run src/ui/app.py --server.address localhost
```
http://localhost:8501 — 처음 실행할 때 색인이 비어 있으면 저장소의 스냅샷으로 특허 색인을 복원하고(약 20초, 외부 호출 없음), 사내 문서를 색인합니다(1분 안팎).

**로그인**: 로컬 데모 계정(가상 인물)은 `config/users.toml`에 있고 비밀번호는 모두 `patentlab-demo`입니다 — `kim`(반도체 테스트팀) · `lee`(배터리 안전팀) · `park`(특허팀) · `choi`(시스템 관리자, 운영 현황) · `guest`(방문자). 계정마다 볼 수 있는 사내 문서가 다릅니다. 운영 환경(`APP_ENV=production`)에서는 이 데모 계정 파일로 시작하지 않습니다.

**화면**: 상단 메뉴 **선행기술 조사**(기본) · 특허 찾기 · 사내 문서 · 운영 현황(관리자). 조사 결과 카드를 고르면 오른쪽 패널에서 **구성요소 대비 · 질문하기 · 청구항 해설**로 이어집니다. 사용자용 안내: [docs/user-guide.md](docs/user-guide.md).

### 데이터와 색인
저장소에는 수집한 특허 데이터(`data/processed`), KIPRIS·위키백과 응답 캐시(`data/raw`), 색인 스냅샷(`data/index`), 사내 문서 원본과 카탈로그(`data/internal`)가 들어 있습니다. 조사·질문으로 특허·문서가 늘어나면 커밋 전에 스냅샷을 갱신합니다.

```bash
python -m src.data_prep.collect_meta             # 처음부터: KIPRIS에서 분야별 메타데이터 수집 (18회 호출, 캐시됨)
python -m src.vectorstore.index_meta             # 메타데이터 색인 (--reset: 지우고 다시)
python -m src.vectorstore.index_docs --rebuild   # 문서 컬렉션을 캐시로 재청킹·재색인 (외부 호출 없음)
python -m src.vectorstore.index_internal         # 사내 문서 카탈로그와 색인 동기화 (--check: 검사만)
python -m src.vectorstore.snapshot export        # 색인 스냅샷 저장 (커밋 전에) / import: 복원
python -m src.security.auth hash                 # 계정 파일에 넣을 비밀번호 해시 만들기 (list: 계정 확인)
```

**사내 문서 추가·권한 변경** — 파일을 `data/internal/` 아래에 두고 `data/internal/catalog.yaml`에 등급(`공개`·`내부`·`기밀`)·열람 주체(`all`·`employee`·`team:<팀>`·`role:<역할>`·`user:<아이디>`)·버전을 적습니다. 카탈로그에 없는 파일과 권한 설정이 틀린 문서(예: 기밀인데 `employee`)는 색인하지 않습니다.

**임베딩 모델 바꾸기** — 스냅샷에는 어떤 임베딩 모델로 만든 벡터인지가 기록되어(`data/index/manifest.json`) 설정과 다르면 복원하지 않습니다. 모델을 바꾸면 `index_meta --reset` → `index_docs --rebuild` → `index_internal` → `snapshot export` 순서로 다시 만듭니다 (1만 건, CPU에서 건당 약 0.7초)<!-- TODO: 확인 — 0.7초/건(docs/decisions/005)이면 10,284건에 약 2시간. 이전 README는 '수십 분' -->. LLM 요청 한도는 `.env`의 `LLM_RPM`(분당 요청 수)·`LLM_MAX_RETRIES`(재시도)·`COMPARE_WORKERS`(대비 병렬 수)로 조정합니다.

> **Windows 경로 주의**: Chroma는 한글 등 영문이 아닌 문자가 든 절대 경로의 색인 파일을 읽지 못합니다. 프로젝트 루트에서 실행하거나 `.env`에 `CHROMA_DIR=영문 경로`를 지정하세요.

### 평가·점검 명령
```bash
python -m src.service.prior_art "스캔 체인 고장 로그로 학습 데이터를 만들어 고장 셀을 예측하는 진단 방법"   # 선행기술 조사 (CLI)
python -m src.rag.graph --item 1020230136220 --question "지터를 측정하는 원리는?" --show-graph           # 특허 질의응답 + 그래프 구조
python -m src.service.internal_qa --user kim --question "March SS의 불량 검출률은?"                    # 사내 문서 질의응답 (CLI)
python -m src.eval.make_prior_art_cases           # 평가셋 생성 → docs/eval/prior_art_cases.json
python -m src.eval.prior_art_eval --label <이름>  # 자동 평가 → docs/eval/prior_art_eval.md 요약표에 한 줄 추가
python -m src.eval.security_eval [--skip-llm]     # 보안 평가 (권한 밖 노출·악성 지시·사내 문서 검색) → docs/eval/security_eval.md
python -m src.eval.compare_search --top-k 5       # 유사도 vs MMR 근거 비교
python -m src.eval.rag_vs_norag                   # RAG 전후 답변 생성
python -m src.eval.verify [점검 이름 ...]          # 통합 점검 (실제 색인·LLM·AppTest) → docs/eval/verification_run.md
pytest                                            # 단위 테스트 (LLM·네트워크 없이 실행)
```

## 배포
GitHub Pages는 정적 파일만 올릴 수 있어(그리고 비공개 저장소는 유료 플랜) 이 앱(Streamlit 서버 + 로컬 임베딩 모델 2.2GB + Chroma)을 돌릴 수 없습니다. 무료 CPU 2 vCPU·16GB를 주는 **Hugging Face Spaces(Docker)** 에 올립니다.

```bash
HF_TOKEN=hf_... python deploy/huggingface/push.py --space <아이디>/patent-lab   # 기본 비공개 Space
```
- 시크릿: `GEMINI_API_KEY`, `USERS_TOML`(운영 계정 — 컨테이너는 `APP_ENV=production`이라 데모 계정 파일로는 시작하지 않음). 가상 데이터 데모라면 변수 `LLM_MAX_LEVEL=내부`.
- 공개 Space로 열기 전에 KIPRIS Plus의 재배포 조건을 확인합니다 (`data/`의 수집 데이터가 함께 공개됨).
- `Dockerfile`·`requirements.txt`가 바뀌면 GitHub Actions가 이미지를 빌드하고 앱 상태 확인 주소(`/_stcore/health`)가 응답하는지 봅니다.
- 계정·권한 변경, 비밀키 교체, 감사 로그, 장애 대응, 백업·복구: [docs/operations.md](docs/operations.md)

## 데이터 출처와 이용 조건
| 데이터 | 출처 | 범위 | 이용 조건과 지킨 방법 |
|---|---|---|---|
| 특허 메타데이터 (명칭·출원인·일자·IPC·상태·요약) | 한국특허정보원 **KIPRIS Plus** Open API `getAdvancedSearch`, `getWordSearch` | 9개 분야 IPC 8,747건 일괄 + 조사 때 키워드 검색으로 받은 특허(검색 1회 최대 100건) | KIPRIS Plus 이용약관과 서비스 키 발급 조건을 따른다. 무료 이용은 월 1,000회 한도<!-- TODO: 확인 — KIPRIS Plus 현재 한도. src/config.py KIPRIS_MONTHLY_LIMIT=10000과 다름 -->라 응답을 캐시하고 모든 호출을 `data/raw/api_calls.log`에 기록한다. **저장소를 공개하거나 배포할 때는 재배포 조건을 먼저 확인한다.** |
| 특허 공보 (요약·청구항) | KIPRIS Plus `getBibliographyDetailInfoSearch` | 질문하거나 조사에서 대비한 특허만. 이 API는 명세서 상세 설명을 주지 않아 요약과 청구항을 공보 문서로 쓴다 | 출원번호별로 한 번만 호출하고 캐시한다. 원문 링크는 특허청 DOI(`https://doi.org/10.8080/<출원번호>`) |
| 기술 개념 문서 | **한국어 위키백과** MediaWiki API | 질문마다 선별된 문서 | 본문은 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ko). 근거로 쓸 때 제목과 원문 URL을 표시한다. [위키미디어 User-Agent 정책](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy)에 따라 연락처를 보내고 0.5초 간격을 둔다 |
| 사내 문서 11건 | 이 저장소에서 만든 **가상 데이터** (`data/internal`) | 연구소·팀·인물·회사·수치는 모두 데모용으로 지어낸 것 | 실제 회사·인물과 관계없다. 외부 수신 문서의 공격 문장도 평가를 위해 넣은 것이다 |

외부 API 키는 `.env`(배포는 시크릿)에만 두며 저장소·화면·로그에 노출하지 않습니다. 전체 커밋 이력에서 키 값이 없음을 확인했습니다.

## 프로젝트 구조
```
src/
├── config.py               # 경로·청킹·검색·모델·제공자·요청 한도·전송 상한 등 조정 가능한 값
├── env.py                  # .env 키 읽기 (제공자 결정), 누락 안내
├── data_prep/              # 데이터 준비 — 도메인(KIPRIS·위키백과·사내 문서)에 묶인 코드는 여기에
│   ├── collect_meta.py     #   특허 메타데이터 수집
│   ├── items.py            #   특허 1건 → 색인 텍스트·메타데이터
│   ├── http.py             #   외부 호출 공통 규칙: 재시도·간격·기록·키 가림
│   ├── kipris.py           #   KIPRIS 호출·응답 파싱, 특허 공보(요약·청구항)
│   ├── patent_search.py    #   조사 때 KIPRIS 키워드 검색으로 특허 추가
│   ├── wiki.py             #   위키백과 검색·본문
│   ├── docs.py             #   외부 문서 형태, 출처 유형 이름
│   ├── collect_docs.py     #   검색어 추출·후보 선별로 수집할 문서 결정, 수집 기록
│   └── internal.py         #   사내 문서 카탈로그 검사, Markdown·CSV(표 청크) 변환
├── vectorstore/            # 색인
│   ├── store.py            #   컬렉션 열기, 임베딩(OpenAI / 로컬 sentence-transformers)
│   ├── index_meta.py       #   메타데이터 색인
│   ├── index_docs.py       #   문서 청킹·색인
│   ├── index_internal.py   #   사내 문서 색인 동기화 (지문 비교·권한 변경 반영·의심 구절 표시)
│   └── snapshot.py         #   색인 스냅샷 저장·복원 (임베딩 모델 manifest)
├── security/               # 보안 계층
│   ├── acl.py              #   등급·사용자·열람 주체, 벡터 DB 필터, 권한 설정 검사
│   ├── auth.py             #   계정 파일·비밀번호 해시·로그인 잠금
│   ├── policy.py           #   외부 전송 정책: LLM 경로별 상한, 경로 선택
│   ├── audit.py            #   감사 로그 (입력은 해시만), 일일 한도·운영 현황 집계
│   └── guard.py            #   악성 지시 점검(색인), 출처 없는 링크 삭제(답변)
├── rag/                    # RAG 파이프라인
│   ├── prompts.py          #   프롬프트, 면책 문구 (사내 문서용 방어 프롬프트 포함)
│   ├── llm.py              #   채팅 모델(요청 속도 제한·재시도, 사내 모델), JSON 파싱
│   ├── retriever.py        #   근거 검색·병합
│   ├── citations.py        #   인용 근거 추출
│   └── graph.py            #   LangGraph retrieve → generate
├── service/                # 서비스 기능
│   ├── prior_art.py        #   선행기술 조사 (구성요소 대비, 원문 구절 검증, 아이디어 등급)
│   ├── internal_qa.py      #   사내 문서 질의응답 (LangGraph retrieve → policy → generate → check)
│   ├── claim_brief.py      #   청구항 해설
│   └── explore.py          #   조건 기반 탐색, 이름·번호 검색
├── ui/app.py               # Streamlit 화면 (로그인, 선행기술 조사, 특허 찾기, 사내 문서, 운영 현황)
└── eval/                   # 평가·점검
    ├── make_prior_art_cases.py, prior_art_eval.py   # 선행기술 조사 자동 평가
    ├── security_eval.py                              # 보안 평가
    ├── compare_search.py, rag_vs_norag.py            # 실험
    └── verify.py                                     # 통합 점검
tests/                      # 단위 테스트 (LLM·네트워크 없음, 권한 필터는 임시 Chroma로 확인)
config/users.toml           # 로컬 데모 계정 (가상 인물, 비밀번호 해시)
data/internal/              # 사내 문서 원본(가상 데이터)과 카탈로그(등급·열람 주체·버전)
docs/
├── architecture.md         # RAG 파이프라인·보안 계층 설계, 설정값의 이유, 겪은 문제와 해결
├── decisions/              # 설계 결정 기록
├── experiments/            # 실험 결과 (MMR, RAG 전후, 출처 결합)
├── eval/                   # 평가셋, 실험 요약표, 보안 평가, 실행별 결과, 통합 점검 원문
├── operations.md           # 운영 문서: 배포·계정·비밀키·감사 로그·장애 대응·백업과 복구
├── user-guide.md           # 사용자 안내 (연구원용)
├── requirements.md         # 제품 요구사항
├── qa.md                   # 검증 결과
└── screenshots/            # 실행 화면
deploy/huggingface/         # Hugging Face Space용 README와 배포 스크립트
Dockerfile                  # CPU torch + bge-m3를 넣은 이미지, 운영 모드로 실행
.env.example                # 설정 항목과 제공자별 기본값 (복사해서 .env로)
.streamlit/config.toml      # 화면에 오류 추적 정보를 보이지 않는 설정
.github/workflows/          # ci.yml: push·PR마다 ruff + pytest / docker.yml: 이미지 빌드·상태 확인
pyproject.toml              # pytest·ruff 설정 (런타임 의존성은 requirements.txt에 버전 고정)
```

## 버전 기록
| 버전 | 날짜 | 주요 변경 | 결과 |
|---|---|---|---|
| v3 | 2026-11-05 | 로그인·역할 · 사내 문서(카탈로그·버전·표 청크) · 검색 단계 열람 권한 필터 · 자료 등급별 LLM 전송 정책(미공개 아이디어 차단) · 악성 지시 세 겹 방어 · 감사 로그·운영 현황 · Docker·Hugging Face Spaces 배포 구성 | 권한 밖 노출 0/38 (필터 없이 38/38) · 정당한 접근 22/22 · 악성 지시 공격 성공 8/12 → 0/12 ([보안 평가](docs/eval/security_eval.md)) |
| v2 | 2026-10-24 | LLM 제공자 분리(`.env`의 키로 Gemini 무료 등급·OpenAI 선택) · 로컬 임베딩 bge-m3로 전체 재색인 · 라벨 없는 자동 평가셋 40사례 · CI(push·PR마다 ruff·pytest) | 운영 비용 0원. 선행기술 조사 기준선 Recall@5 85% · 개시율 96% · 사례당 23초 ([실험 요약표](docs/eval/prior_art_eval.md)) |
| v1 | 2026-10-22 | 질의응답 RAG(LangGraph) · 선행기술 조사(구성요소 × 청구항 대비, 원문 구절 검증) · 청구항 해설 · 특허 찾기. LLM gpt-4o-mini·gpt-4.1, 임베딩 text-embedding-3-small | RAG 전후 사실 정확성 0/4 → 4/4 · MMR로 질의응답 근거 출처 문서 수 1~2 → 3~5(특허 3건) ([검증 결과](docs/qa.md)) |

v1은 SSAFY AI 과정에서 개인 프로젝트로 만들었습니다.

## 한계와 다음 단계
- **판정 엄격도**: 원문 구절 검증이 거짓 개시를 막는 대신 맞는 판정도 약 2.9%(정답 특허 판정 136개 중 4개) 떨어뜨린다. 구절 재탐색·토큰 단위 일치 실험 예정
- **후보 생성의 MMR**: 비슷한 특허끼리 밀어내 정답이 빠지는 사례가 측정됐다. 유사도 검색·BM25 하이브리드 실험 예정
- **평가의 한계**: 선행기술 조사 평가는 정답이 자기 자신이라 검색 상한과 판정 일관성을 재는 것이지 진짜 선행 특허 탐색 능력은 아니다. 보안 평가는 가상 문서 11건·공격 12사례로 작다 — 권한 필터의 0건은 규모와 상관없이 지켜야 하는 값이지만, 악성 지시 방어율은 사례를 늘려 다시 재야 한다
- **사내 모델**: 기밀 자료를 보낼 사내 모델 경로는 OpenAI 호환 서버면 붙도록 만들었지만 실제 사내 모델로 품질을 재지 못했다
- **인증**: 계정은 TOML 파일과 브라우저 세션이다. 사내 SSO(OIDC, Streamlit `st.login`)로 바꾸고 IdP 그룹을 열람 주체로 쓰는 것이 다음 단계
- **감사 로그 보관**: 무료 Space는 재시작하면 디스크가 비워진다. 운영에서는 영구 저장소나 로그 수집기로 보내야 한다
- **질문 문장의 등급**: 문서와 아이디어에는 등급이 있지만 질문 문장 자체는 분류하지 않는다
- **출처 간 분량 불균형**: 위키 문서가 청구항보다 청크 수가 20배 많은 특허에서 근거가 위키로 치우친다. 출처별 근거 수 배분 검토
- **보고서 내보내기**: 구성요소 × 특허 매트릭스(클레임 차트)를 Markdown/PDF로 저장하는 기능

## 라이선스
코드는 [MIT 라이선스](LICENSE), 수집한 특허 데이터는 KIPRIS Plus 이용약관을 따릅니다. `data/internal`의 사내 문서는 데모용 가상 데이터입니다.
