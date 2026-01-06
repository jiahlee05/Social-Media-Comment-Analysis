# Session 1: AI 기반 데이터 분석 입문

**수업 시간:** 2시간
**목표:** AI 어시스턴트를 활용하여 첫 데이터 분석 완성하기

---

## 📋 수업 목차

1. **바이브 코딩 소개** (20분)
2. **Python & Jupyter Notebook 시작하기** (30분)
3. **Pandas로 데이터 다루기** (40분)
4. **첫 데이터 분석 실습** (30분)

---

## 1. 바이브 코딩 소개 (20분)

### 바이브 코딩이란?
- AI 어시스턴트(Claude, GPT)와 협업하는 새로운 코딩 방식
- 자연어로 코드를 요청하고, AI가 생성
- 코드를 이해하고 수정하는 과정

### AI 어시스턴트 활용법

#### 좋은 프롬프트 예시
```
✅ "소셜 미디어 데이터를 불러와서 플랫폼별 평균 좋아요 수를 계산해줘"
✅ "이 코드가 무엇을 하는지 한국어로 설명해줘"
✅ "NameError가 발생했어. 어떻게 고치면 될까?"
```

#### 피해야 할 프롬프트
```
❌ "코드 짜줘"
❌ "안 돼"
❌ "분석해줘" (구체적이지 않음)
```

### 실습: AI에게 첫 질문하기
```
프롬프트: "Python에서 CSV 파일을 읽으려면 어떤 라이브러리를 사용해야 해?
그리고 간단한 예제 코드를 보여줘"
```

---

## 2. Python & Jupyter Notebook 시작하기 (30분)

### Jupyter Notebook 실행
```bash
# 터미널에서 실행
jupyter notebook
```

### 기본 셀 사용법
- **코드 셀**: Python 코드 실행
- **마크다운 셀**: 설명 작성
- **단축키**:
  - `Shift + Enter`: 셀 실행
  - `Esc + A`: 위에 셀 추가
  - `Esc + B`: 아래에 셀 추가

### Python 기초 (5분 리뷰)
```python
# 변수
name = "Communication Study"
count = 100

# 리스트
platforms = ["Instagram", "Twitter", "Facebook"]

# 딕셔너리
engagement = {"likes": 500, "shares": 100}

# 함수
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
```

### 필수 라이브러리 임포트
```python
import pandas as pd  # 데이터 분석
import numpy as np   # 수치 계산
import matplotlib.pyplot as plt  # 시각화
```

---

## 3. Pandas로 데이터 다루기 (40분)

### 데이터 불러오기
```python
# CSV 파일 읽기
df = pd.read_csv('datasets/social_media_engagement.csv')

# 데이터 확인
print(df.head())  # 처음 5행
print(df.info())  # 데이터 정보
print(df.describe())  # 기술 통계
```

### 데이터 탐색 핵심 명령어

#### 1. 데이터 확인
```python
df.shape  # (행 수, 열 수)
df.columns  # 컬럼 이름
df.dtypes  # 데이터 타입
```

#### 2. 데이터 선택
```python
# 컬럼 선택
df['platform']
df[['platform', 'likes']]

# 행 선택 (조건)
df[df['likes'] > 100]
df[df['platform'] == 'Instagram']
```

#### 3. 데이터 집계
```python
# 평균
df['likes'].mean()

# 플랫폼별 그룹화
df.groupby('platform')['likes'].mean()

# 여러 통계 한 번에
df.groupby('platform').agg({
    'likes': 'mean',
    'shares': 'sum',
    'comments': 'median'
})
```

### AI 활용 팁
```
프롬프트: "이 데이터프레임에서 연령대(age_group)별로
평균 참여도(engagement)를 계산하고 내림차순으로 정렬하는 코드를 짜줘"
```

---

## 4. 첫 데이터 분석 실습 (30분)

### 실습 과제: 소셜 미디어 참여도 분석

**분석 질문:**
1. 어떤 플랫폼이 가장 높은 평균 좋아요를 받는가?
2. 연령대별 참여도 차이가 있는가?
3. 시간대별 포스트 패턴은?

### 단계별 접근

#### Step 1: 데이터 로드 및 확인
```python
import pandas as pd

# 데이터 읽기
df = pd.read_csv('../datasets/social_media_engagement.csv')

# AI에게 물어보기: "이 데이터셋의 구조를 설명해줘"
print(df.head())
print(df.info())
```

#### Step 2: 플랫폼별 분석
```python
# 플랫폼별 평균 좋아요
platform_likes = df.groupby('platform')['likes'].mean().sort_values(ascending=False)
print(platform_likes)
```

#### Step 3: 간단한 시각화
```python
import matplotlib.pyplot as plt

platform_likes.plot(kind='bar')
plt.title('Average Likes by Platform')
plt.ylabel('Average Likes')
plt.tight_layout()
plt.show()
```

#### Step 4: AI와 함께 해석하기
```
프롬프트: "이 분석 결과를 보면 Instagram이 가장 높은 평균 좋아요를 받았어.
이것이 커뮤니케이션 전략에 어떤 의미가 있을까?"
```

---

## 💡 핵심 요약

### 꼭 기억할 것
1. **AI는 파트너**: 코드를 설명하고 생성하는 도우미
2. **Pandas는 핵심**: 데이터 분석의 기본 도구
3. **탐색이 우선**: 항상 `head()`, `info()`, `describe()`로 시작
4. **질문 중심**: 데이터가 무엇을 말하는지 계속 질문

### 다음 세션 예고
- 데이터 시각화 심화
- 다양한 차트 타입
- 효과적인 비주얼 커뮤니케이션

---

## 📝 과제

1. **AI 연습**: Claude나 ChatGPT와 대화하며 다음 코드 생성
   - 데이터에서 특정 조건 필터링
   - 두 개 이상의 그룹화 조건 사용

2. **데이터 분석**: `social_media_engagement.csv`로 다음 질문 답하기
   - 어떤 연령대가 가장 활발히 댓글을 다는가?
   - 주말과 평일의 참여도 차이는?

3. **리플렉션**: 오늘 배운 것 중 가장 유용한 것 3가지 적기

---

## 🔗 추가 학습 자료

- [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- [Claude AI 사용 가이드](https://www.anthropic.com/claude)
- [Python for Data Analysis (온라인 무료)](https://wesmckinney.com/book/)

---

**다음 시간에 만나요! 질문이 있으면 AI에게 먼저 물어보고, 그래도 안 되면 교수님께!** 🚀
