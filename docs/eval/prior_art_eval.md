# 선행기술 조사 자동 평가 — 실험 요약

사례: [prior_art_cases.json](prior_art_cases.json) (청구항 1을 바꿔 쓴 아이디어, 정답 = 그 특허). 실행: `python -m src.eval.prior_art_eval --label <이름>`. 사례별 결과는 [prior_art_eval/](prior_art_eval/).

| 날짜 | 라벨 | 사례 | R@20 | R@5 | MRR@5 | 개시율 | 개시+일부 | 겹침 1위 | 시간/건 | 토큰/건 | 비용/건 | 모델 | 후보/대비 | 상세 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | baseline-gemini | 6 | 67% | 67% | 0.583 | 100% | 100% | 100% | 41초 | 4,969 | - | gemini-3.5-flash | 20/5 | [baseline-gemini_2026-10-24.md](prior_art_eval/baseline-gemini_2026-10-24.md) |
