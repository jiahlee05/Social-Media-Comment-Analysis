제시해주신 Python 강의 자료를 **R과 Tidyverse** 생태계에 맞춰 변환한 마크다운 파일입니다. RStudio와 Quarto(또는 R Markdown) 환경을 기준으로 작성되었습니다.

---

# Session 1: AI 기반 데이터 분석 입문 (R)

**수업 시간:** 2시간
**목표:** AI 어시스턴트를 활용하여 R로 첫 데이터 분석 완성하기

---

## 📋 수업 목차

1. **바이브 코딩 소개** (20분)
2. **R & RStudio 시작하기** (30분)
3. **Tidyverse로 데이터 다루기** (40분)
4. **첫 데이터 분석 실습** (30분)

---

## 1. 바이브 코딩 소개 (20분)

### 바이브 코딩이란?

* AI 어시스턴트(Claude, GPT)와 협업하는 새로운 코딩 방식
* 자연어로 데이터 분석 로직을 설명하고, AI가 R 코드를 생성
* 생성된 코드를 이해하고 내 데이터에 맞게 수정하는 과정

### AI 어시스턴트 활용법

#### 좋은 프롬프트 예시

```
✅ "R의 tidyverse를 사용해서 소셜 미디어 데이터를 불러오고 플랫폼별 평균 좋아요 수를 계산해줘"
✅ "이 R 코드가 무엇을 하는지 단계별로 한국어로 설명해줘"
✅ "Error: object 'platform' not found라는 오류가 발생했어. 어떻게 고치면 될까?"

```

#### 피해야 할 프롬프트

```
❌ "코드 짜줘"
❌ "안 돼"
❌ "분석해줘" (구체적이지 않음)

```

### 실습: AI에게 첫 질문하기

```
프롬프트: "R에서 CSV 파일을 읽으려면 어떤 패키지를 사용해야 해? 
tidyverse를 사용한 간단한 예제 코드를 보여줘"

```

---

## 2. R & RStudio 시작하기 (30분)

### RStudio 환경 이해

* **Source**: R 스크립트(.R) 또는 Quarto(.qmd) 편집창
* **Console**: 코드가 실행되는 창
* **Environment**: 생성된 데이터(객체) 확인
* **Files/Plots**: 파일 관리 및 시각화 결과 확인

### 기본 단축키

* **코드 실행**: `Ctrl + Enter` (또는 `Cmd + Enter`)
* **할당 연산자(`<-`) 입력**: `Alt + -` (맥은 `Option + -`)
* **파이프 연산자(`%>%`) 입력**: `Ctrl + Shift + M`

### R 기초 (5분 리뷰)

```r
# 변수 할당 (<- 사용 권장)
name <- "Communication Study"
count <- 100

# 벡터 (Python의 리스트와 유사)
platforms <- c("Instagram", "Twitter", "Facebook")

# 리스트 (다양한 형태를 담는 바구니)
engagement <- list(likes = 500, shares = 100)

# 함수 정의
calculate_average <- function(numbers) {
  return(mean(numbers))
}

```

### 필수 패키지 로드

```r
# tidyverse는 데이터 분석에 필요한 여러 패키지(dplyr, ggplot2, readr 등)를 포함합니다.
library(tidyverse) 

```

---

## 3. Tidyverse로 데이터 다루기 (40분)

### 데이터 불러오기

```r
# CSV 파일 읽기
df <- read_csv('datasets/social_media_engagement.csv')

# 데이터 확인
head(df)      # 처음 6행 확인
glimpse(df)   # 데이터 구조와 타입 확인 (Python의 info()와 유사)
summary(df)   # 요약 통계 확인

```

### 데이터 탐색 핵심 함수 (dplyr)

#### 1. 데이터 확인

```r
dim(df)       # (행 수, 열 수)
colnames(df)  # 컬럼 이름

```

#### 2. 데이터 선택 및 필터링

```r
# 컬럼 선택 (select)
df %>% select(platform)
df %>% select(platform, likes)

# 행 선택 (filter)
df %>% filter(likes > 100)
df %>% filter(platform == "Instagram")

```

#### 3. 데이터 집계

```r
# 전체 평균
df %>% summarize(mean_likes = mean(likes))

# 플랫폼별 그룹화 및 집계 (group_by + summarize)
df %>% 
  group_by(platform) %>% 
  summarize(mean_likes = mean(likes))

# 여러 통계 한 번에 계산
df %>% 
  group_by(platform) %>% 
  summarize(
    mean_likes = mean(likes),
    total_shares = sum(shares),
    median_comments = median(comments)
  )

```

### AI 활용 팁

```
프롬프트: "R의 dplyr를 사용해서 이 데이터프레임의 연령대(age_group)별로 
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

```r
library(tidyverse)

# 데이터 읽기
df <- read_csv('../datasets/social_media_engagement.csv')

# AI에게 물어보기: "이 데이터셋의 glimpse() 결과를 바탕으로 구조를 설명해줘"
glimpse(df)

```

#### Step 2: 플랫폼별 분석

```r
# 플랫폼별 평균 좋아요 계산 및 정렬
platform_likes <- df %>% 
  group_by(platform) %>% 
  summarize(avg_likes = mean(likes)) %>% 
  arrange(desc(avg_likes))

print(platform_likes)

```

#### Step 3: 간단한 시각화 (ggplot2)

```r
# ggplot2를 이용한 막대 그래프
ggplot(platform_likes, aes(x = reorder(platform, -avg_likes), y = avg_likes)) +
  geom_col(fill = "steelblue") +
  labs(title = "Average Likes by Platform",
       x = "Platform",
       y = "Average Likes") +
  theme_minimal()

```

#### Step 4: AI와 함께 해석하기

```
프롬프트: "R 분석 결과를 보면 Instagram이 가장 높은 평균 좋아요를 받았어. 
이것이 마케팅 커뮤니케이션 전략에 어떤 의미가 있을까?"

```

---

## 💡 핵심 요약

### 꼭 기억할 것

1. **AI는 파트너**: 코드를 생성하고 에러를 해결해주는 든든한 조력자
2. **Tidyverse는 핵심**: R 데이터 분석의 표준 라이브러리 (dplyr, ggplot2)
3. **탐색이 우선**: 분석 전 `glimpse()`, `summary()`로 데이터와 먼저 친해지기
4. **파이프 연산자(`%>%`)**: 데이터 흐름을 왼쪽에서 오른쪽으로 자연스럽게 연결

### 다음 세션 예고

* **ggplot2** 심화: 다양한 차트 그리기
* 심미적인 그래프 디자인 (색상, 테마 설정)
* 시각화를 통한 인사이트 도출 및 비주얼 커뮤니케이션

---

## 📝 과제

1. **AI 연습**: Claude나 ChatGPT와 대화하며 다음 코드 생성
* 특정 연령대만 필터링한 후 새로운 데이터셋 만들기
* 두 개 이상의 그룹(플랫폼, 연령대)을 동시에 고려한 집계 코드


2. **데이터 분석**: `social_media_engagement.csv`로 다음 질문 답하기
* 어느 연령대가 가장 활발하게 댓글(`comments`)을 다는가?
* 주말과 평일의 참여도 차이를 `mutate()` 함수를 사용하여 분석해보기


3. **리플렉션**: R을 처음 써보며 Python과 다르다고 느낀 점이나 편리했던 점 3가지 적기

---

## 🔗 추가 학습 자료

* [Tidyverse 공식 문서](https://www.tidyverse.org/)
* [R for Data Science (2e) - 온라인 무료 도서](https://r4ds.hadley.nz/)
* [ggplot2 치트 시트](https://rstudio.github.io/cheatsheets/data-visualization.pdf)

---

**다음 시간에 만나요! 에러가 나면 먼저 AI에게 물어보고, 해결이 안 되면 질문해 주세요!** 🚀

---

### [참고: R 시각화 구조 이해도]

R의 `ggplot2`는 '그래프의 문법'이라는 독특한 체계를 가집니다. 이를 이해하면 훨씬 정교한 시각화가 가능합니다.

---

**다음 단계로 무엇을 도와드릴까요?**

1. 이 내용을 바탕으로 실제 실습용 가상 데이터를 생성하는 코드를 짜드릴까요?
2. 특정 구간(예: ggplot2 시각화)에 대해 더 자세한 보충 자료를 만들어 드릴까요?