# 인간/LLM 생성 텍스트 판별 연구

&nbsp;인간이 작성한 텍스트와 LLM(GPT·Gemini·Claude·DeepSeek)이 생성한 텍스트를 해석 가능한 통계적 피처로 판별하는 연구의 코드베이스

> **Publication (공동 1저자)**
> 
> Hyungbin Park†, **Shinsung Kang†**, Kihoon Lee, Gwangsu Kim. (2026).
> Study on comparative and discriminatory methodology of human and machine-generated languages.
> *Korean Journal of Applied Statistics*, **39**(3), 235–255. [DOI](https://doi.org/10.5351/KJAS.2026.39.3.235)

## 수행 작업 및 퍼포먼스

* Binary Classification: human vs LLM 판별 — Acc 0.9778, F1-Score 0.9864
* Multi-Classification (5-way): human/GPT/Gemini/Claude/DeepSeek 생성 주체 식별 — Acc 0.8598, F1-Score 0.8607
* Ablation으로 perplexity 피쳐의 기여를 정량화 (제외 시 이진 −1.8%p, 다중 −4.4%p)

## 데이터

* Quora 질문에 대한 인간 답변과 4개 LLM의 답변 총 14,833건.
  * LLM 생성 텍스트에 의한 오염을 배제하기 위해 연구 시점 기준 3년 이전에 게시된 질문만 사용
  * 5개 주제 카테고리(가상 시나리오, 개인 경험, 철학, 자기계발, 대인관계)로 구성

## 방법론

* Feature Extraction
  * Stylometric Features — 어휘 밀도, 고유 단어 수, 가독성 지수(Flesch-Kincaid, Gunning Fog, SMOG) 등
  * Perplexity — Llama-3.1-8B로 계산 (장문은 stride 512 슬라이딩 윈도우)
  * LDA Topics — coherence + KneeLocator로 토픽 수 자동 선택 후 서브토픽 확률 분포 사용
* Classifer: DNN (256→128→64→32, BatchNorm/Dropout, AdamW)

## 리포 구조

```
├── 원본 파일.ipynb / 정리.ipynb / textstat.ipynb   # 데이터 구축·문체 피처·EDA
├── Perplexity/        # Llama-3.1-8B perplexity 계산·분석
├── LDA/               # 계층적 LDA 토픽 모델링
├── model/             # 피처 통합, DNN 학습, ablation, 평가
└── token_feature_model/  # 원시 토큰 임베딩 베이스라인
```

## 기여

&nbsp;Perplexity 계산·피처 주입 구현, 시각화, LDA·DNN·평가 지표 코드 오류 검토 및 버전 관리, 담당 파트 논문 집필, 관련 연구 조사.
