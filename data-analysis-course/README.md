# 데이터 분석 바이브 코딩 입문 과정
## AI 기반 커뮤니케이션 데이터 분석 (5주 과정)

**대상:** 겐트 그로벌 캠퍼스 / 유타대 커뮤니케이션 학과 신입생
**수업 시간:** 주 1회, 2시간 (총 5회)
**수업 방식:** 바이브 코딩 (AI 어시스턴트 활용 - GPT, Claude, Antigravity)

---

## 📚 과정 개요

이 과정은 프로그래밍 경험이 없는 커뮤니케이션 학과 학생들이 AI 도구를 활용하여 데이터 분석, 시각화, 통계, 텍스트 분석을 배우는 실습 중심 강의입니다.

### 학습 목표
- AI 어시스턴트(Claude, GPT)를 활용한 데이터 분석 능력 습득
- Python 기반 데이터 분석 도구 활용
- 커뮤니케이션 연구에 필요한 통계 분석 기초
- 텍스트 데이터 분석 및 시각화
- 실험 데이터 처리 및 보고서 작성

---

## 📅 세션별 커리큘럼

### Session 1: AI 기반 데이터 분석 입문 (2시간)
**학습 내용:**
- AI 어시스턴트를 활용한 코딩 시작하기
- Python과 Jupyter Notebook 기초
- 데이터 불러오기 및 탐색 (pandas)
- 소셜 미디어 데이터 첫 분석

**실습 데이터:** 소셜 미디어 참여도 데이터

---

### Session 2: 데이터 시각화 마스터하기 (2시간)
**학습 내용:**
- 데이터 시각화 기본 원칙
- Matplotlib과 Seaborn을 활용한 차트 생성
- 커뮤니케이션 데이터 시각화 베스트 프랙티스
- AI로 시각화 코드 개선하기

**실습 데이터:** 뉴스 소비 패턴 데이터

---

### Session 3: 통계 분석 기초 (2시간)
**학습 내용:**
- 기술 통계 (평균, 표준편차, 상관관계)
- 가설 검정 기초 (t-test, ANOVA)
- 회귀 분석 입문
- AI로 통계 해석하기

**실습 데이터:** 광고 효과 실험 데이터

---

### Session 4: 텍스트 분석 & 워드클라우드 (2시간)
**학습 내용:**
- 텍스트 전처리 (토큰화, 불용어 제거)
- 워드클라우드 생성
- 감성 분석 (Sentiment Analysis)
- 소셜 미디어 텍스트 분석

**실습 데이터:** 트위터/댓글 텍스트 데이터

---

### Session 5: 실험 데이터 처리 & 보고서 작성 (2시간)
**학습 내용:**
- 실험 데이터 전처리 및 정제
- 통합 분석 파이프라인 구축
- 자동화된 보고서 생성
- 최종 프로젝트: 종합 분석 리포트 작성

**실습 데이터:** 커뮤니케이션 실험 데이터

---

## 🛠 필요한 도구

### 1. Python 환경
```bash
# Anaconda 또는 Python 3.8+ 설치
# 필요한 패키지 설치
pip install pandas numpy matplotlib seaborn scipy wordcloud nltk scikit-learn jupyter
```

### 2. AI 어시스턴트
- **Claude** (Anthropic): 코드 설명, 디버깅, 분석 해석
- **ChatGPT** (OpenAI): 코드 생성, 문제 해결
- **Cursor/Antigravity**: AI 통합 코드 에디터

### 3. Jupyter Notebook
```bash
jupyter notebook
```

---

## 📁 디렉토리 구조

```
data-analysis-course/
├── README.md (이 파일)
├── session1/
│   ├── lecture_notes.md
│   ├── code_examples.ipynb
│   └── exercises.md
├── session2/
│   ├── lecture_notes.md
│   ├── code_examples.ipynb
│   └── exercises.md
├── session3/
│   ├── lecture_notes.md
│   ├── code_examples.ipynb
│   └── exercises.md
├── session4/
│   ├── lecture_notes.md
│   ├── code_examples.ipynb
│   └── exercises.md
├── session5/
│   ├── lecture_notes.md
│   ├── code_examples.ipynb
│   └── exercises.md
└── datasets/
    ├── social_media_engagement.csv
    ├── news_consumption.csv
    ├── advertising_experiment.csv
    ├── social_media_comments.csv
    └── communication_experiment.csv
```

---

## 🎯 바이브 코딩 철학

**바이브 코딩**은 AI 어시스턴트와 협업하여 코딩하는 새로운 패러다임입니다:

1. **코드 작성:** AI에게 자연어로 원하는 것을 설명
2. **코드 이해:** AI에게 코드 설명 요청
3. **오류 해결:** AI와 함께 디버깅
4. **개선:** AI에게 코드 개선 제안 요청

### 예시 프롬프트
```
"이 데이터에서 연령대별 소셜 미디어 사용 시간의 평균을 구하고 막대그래프로 시각화해줘"

"이 코드의 오류를 찾아서 수정해줘"

"이 분석 결과를 일반인도 이해할 수 있게 해석해줘"
```

---

## 📊 평가 방법

1. **세션별 실습 과제** (50%): 각 세션 종료 후 제출
2. **최종 프로젝트** (30%): Session 5에서 완성하는 종합 분석 리포트
3. **참여도** (20%): 수업 참여 및 질문

---

## 💡 학습 팁

1. **두려워하지 마세요:** 프로그래밍 경험이 없어도 AI가 도와줍니다
2. **질문하세요:** AI에게 모르는 것을 계속 질문하세요
3. **실험하세요:** 코드를 수정하고 실행해보며 배웁니다
4. **커뮤니케이션 관점:** 항상 "이 데이터가 무엇을 말하는가?"를 고민하세요

---

## 📧 문의

강의 관련 문의사항은 담당 교수에게 연락하세요.

**Good luck with your data analysis journey! 🚀**
