제공해주신 Python 강의 자료의 흐름을 그대로 유지하면서, R의 핵심 시각화 라이브러리인 **ggplot2**와 **tidyverse** 생태계를 기반으로 한 강의 설명서(Markdown)입니다.

---

# Session 2: 데이터 시각화 마스터하기 (R)

**수업 시간:** 2시간
**목표:** R의 '그래프의 문법'을 이해하고 효과적인 데이터 스토리텔링 완성하기

---

## 📋 수업 목차

1. **데이터 시각화의 원칙** (20분)
2. **ggplot2 기초: 그래프의 문법** (30분)
3. **고급 커스터마이징과 통계적 시각화** (40분)
4. **커뮤니케이션을 위한 비주얼 스토리텔링** (30분)

---

## 1. 데이터 시각화의 원칙 (20분)

### 왜 시각화가 중요한가?

**"데이터는 숫자로 말하고, 시각화는 이야기로 전달한다"**

* 인간은 시각적 패턴 인식에 최적화된 존재
* 수천 행의 데이터프레임보다 하나의 차트가 더 강력한 인사이트 제공
* 데이터 분석 결과를 의사결정권자에게 설득하는 핵심 도구

### 좋은 시각화 vs 나쁜 시각화

#### ✅ 좋은 시각화의 특징

1. **명확한 목적**: 비교인가, 추세인가, 분포인가?
2. **적절한 차트 선택**: 데이터의 성격(연속형 vs 범주형)에 맞는 형식
3. **높은 데이터-잉크 비유 (Data-Ink Ratio)**: 불필요한 장식 제거
4. **색상의 전략적 사용**: 강조할 부분에만 시선을 집중시킴

#### ❌ 피해야 할 것들

* 가독성을 해치는 3D 효과 및 그림자
* 너무 화려해서 데이터 본연의 의미를 가리는 배경
* 축의 시작점을 0이 아닌 곳으로 설정하여 왜곡하는 행위 (특별한 목적 제외)

### 차트 타입 선택 가이드

| 목적 | 추천 차트 (R ggplot2 함수) |
| --- | --- |
| **비교** | `geom_col()`, `geom_bar()` |
| **시간 추이** | `geom_line()`, `geom_area()` |
| **분포** | `geom_histogram()`, `geom_boxplot()`, `geom_violin()` |
| **관계** | `geom_point()`, `geom_jitter()` |
| **상관관계** | `geom_tile()` (Heatmap), `GGally::ggpairs()` |

---

## 2. ggplot2 기초: 그래프의 문법 (30분)

### ggplot2 구조 이해: Grammar of Graphics

R의 시각화는 **'층(Layer)을 쌓는 과정'**입니다. 마치 투명한 필름을 하나씩 겹쳐 그림을 완성하는 것과 같습니다.

```r
library(tidyverse)

# 기본 공식
# ggplot(data = 데이터, aes(x = 가로축, y = 세로축, color = 색상 등)) +
#   geom_함수() (그래프 형태 결정) +
#   labs() (제목 및 레이블)

```

### 기본 플롯 만들기

#### 1. 선 그래프 (추세 확인)

```r
library(tidyverse)

# 데이터 생성
time_data <- tibble(day = 1:10, consumption = c(5, 7, 6, 9, 12, 11, 15, 14, 18, 20))

ggplot(time_data, aes(x = day, y = consumption)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(size = 3) +
  labs(title = "Daily News Consumption", x = "Day", y = "Articles Read") +
  theme_minimal()

```

#### 2. 막대그래프 (범주별 비교)

```r
platforms <- c('Instagram', 'Twitter', 'Facebook', 'TikTok')
likes <- c(850, 620, 730, 920)
df_bar <- tibble(platform = platforms, avg_likes = likes)

ggplot(df_bar, aes(x = reorder(platform, -avg_likes), y = avg_likes, fill = platform)) +
  geom_col(show.legend = FALSE) +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Average Likes by Platform", x = "Platform", y = "Likes")

```

#### 3. 산점도 (관계 파악)

```r
ggplot(df, aes(x = likes, y = shares)) +
  geom_point(alpha = 0.5, color = "darkred") +
  labs(title = "Likes vs Shares Relationship")

```

---

## 3. 고급 커스터마이징과 통계적 시각화 (40분)

### 통계적 시각화의 강자 ggplot2

#### 1. 분포와 이상치 확인

```r
# 히스토그램과 밀도 곡선
ggplot(df, aes(x = likes)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "gray") +
  geom_density(color = "blue", size = 1)

# 박스플롯 (플랫폼별 분포 비교)
ggplot(df, aes(x = platform, y = likes, fill = platform)) +
  geom_boxplot() +
  coord_flip() # 가로 세로 전환

```

#### 2. 관계와 회귀선

```r
# 산점도 위에 회귀선(추세선) 추가
ggplot(df, aes(x = likes, y = shares)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", color = "red") + # 선형 회귀선
  labs(title = "Likes to Shares Trend")

```

#### 3. 패싯(Faceting): 그룹별 차트 자동 분할

Python의 subplots보다 훨씬 강력하고 쉬운 기능입니다.

```r
# 플랫폼별로 차트를 쪼개서 보기
ggplot(df, aes(x = age, y = engagement)) +
  geom_point() +
  facet_wrap(~platform) # 변수 하나를 기준으로 차트 분할

```

### 테마와 스타일 커스터마이징

```r
ggplot(df, aes(x = platform, y = likes)) +
  geom_col() +
  theme_classic() + # 테마 변경: minimal, light, dark, classic 등
  theme(
    plot.title = element_text(size = 20, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

```

---

## 4. 커뮤니케이션을 위한 비주얼 스토리텔링 (30분)

### 데이터 스토리의 구조

단순히 그래프를 그리는 것을 넘어, 독자에게 **'의미'**를 전달해야 합니다.

1. **관심(Hook)**: 무엇이 평소와 다른가?
2. **맥락(Context)**: 이 수치가 우리 비즈니스에 어떤 의미인가?
3. **강조(Emphasis)**: 어디를 봐야 하는가?
4. **제안(Call to Action)**: 그래서 우리는 무엇을 해야 하는가?

### 실전 예제: 대시보드 구성 (patchwork 패키지)

R에서는 `patchwork` 라이브러리를 통해 여러 그래프를 아주 쉽게 결합합니다.

```r
library(patchwork)

p1 <- ggplot(df, aes(x = platform)) + geom_bar()
p2 <- ggplot(df, aes(x = age, y = likes)) + geom_point()

# 그래프 결합
(p1 | p2) + plot_annotation(title = "Social Media Analysis Dashboard")

```

### 주석(Annotation)으로 통찰 더하기

```r
ggplot(df_bar, aes(x = platform, y = avg_likes)) +
  geom_col(fill = "lightgray") +
  # 특정 데이터 강조
  geom_col(data = filter(df_bar, platform == "TikTok"), fill = "red") +
  # 텍스트 주석 추가
  annotate("text", x = "TikTok", y = 1000, 
           label = "Most Engaging!", color = "red", fontface = "bold") +
  # 화살표 추가
  annotate("segment", x = "TikTok", xend = "TikTok", y = 980, yend = 930,
           arrow = arrow(length = unit(0.2, "cm")), color = "red")

```

---

## 💡 핵심 요약

### 꼭 기억할 것

1. **Layering**: `ggplot()` 다음에 `+` 기호를 사용하여 층을 쌓으세요.
2. **Aes**: 데이터의 변수를 시각적 요소(x, y, color, size)에 매핑하는 핵심 단계입니다.
3. **Faceting**: 데이터를 쪼개서 보는 습관이 숨겨진 패턴을 찾아냅니다.
4. **Annotation**: 그래프만 던지지 말고, 무엇을 봐야 하는지 글로 적어주세요.

### Python (Seaborn) vs R (ggplot2)

| 특징 | Seaborn | ggplot2 |
| --- | --- | --- |
| **철학** | 완성된 차트 형태 호출 | 문법을 조합하여 생성 |
| **커스텀** | Matplotlib 파라미터 조절 | 레이어 추가 및 테마 함수 |
| **그룹화** | `hue`, `col` 파라미터 | `facet_wrap`, `facet_grid` |
| **코드 스타일** | 함수형 (Functional) | 선언형 (Declarative) |

---

## 📝 과제

1. **AI 활용 연습**:
* "ggplot2에서 색맹 친화적인(colorblind-friendly) 팔레트를 적용하는 코드를 알려줘"라고 질문해보기


2. **데이터 분석**:
* `news_consumption.csv` 데이터를 로드하여 연령대별 선호 카테고리를 `facet_wrap`으로 시각화하기


3. **스토리텔링**:
* 가장 소비가 많은 시간대를 찾아내고, `annotate` 함수로 해당 시간을 강조한 선 그래프 그리기



---

## 🔗 추가 학습 자료

* [ggplot2 Cheat Sheet (필수 소장)](https://rstudio.github.io/cheatsheets/data-visualization.pdf)
* [The R Graph Gallery (다양한 예제 코드)](https://r-graph-gallery.com/)
* [Fundamentals of Data Visualization (온라인 도서)](https://clauswilke.com/dataviz/)

---

**다음 시간: 실제 데이터를 활용한 통계 분석과 가설 검정! 📊**

**어떤 부분을 더 자세히 알아보고 싶으신가요?**

* `ggplot2`의 특정 차트(예: 히트맵) 코드가 궁금하신가요?
* 아니면 이 자료를 바탕으로 직접 실습해볼 가상 데이터를 만들어 드릴까요?