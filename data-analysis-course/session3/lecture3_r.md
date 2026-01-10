제시해주신 Python 강의 자료를 **R의 Tidyverse와 기본 통계 함수**를 사용하는 버전으로 변환한 마크다운 파일입니다. R은 통계 분석을 위해 태어난 언어인 만큼, 수식과 분석 과정이 훨씬 직관적입니다.

---

# Session 3: 통계 분석 기초 (R 버전)

**수업 시간:** 2시간
**목표:** R과 Tidyverse를 활용하여 커뮤니케이션 연구에 필요한 기초 통계 분석 마스터하기

---

## 📋 수업 목차

1. **통계의 기초** (20분)
2. **기술 통계** (25분)
3. **가설 검정 (t-test, ANOVA)** (40분)
4. **상관관계와 회귀 분석** (35분)

---

## 1. 통계의 기초 (20분)

### 왜 통계가 필요한가?

**커뮤니케이션 연구의 질문들:**

* 광고 캠페인 노출 전후로 브랜드 인지도가 정말 변했는가?
* 인스타그램과 트위터 사용자의 정치적 성향 차이가 통계적으로 유의미한가?
* 기사의 제목 길이가 클릭률(CTR)을 얼마나 예측할 수 있는가?

### 기본 개념

#### 모집단 vs 표본

* **모집단 (Population)**: 우리가 알고 싶은 전체 대상
* **표본 (Sample)**: 모집단에서 실제로 데이터를 수집한 일부
* **추론 통계**: 표본을 통해 모집단의 성질을 확률적으로 추측하는 것

#### 확률 분포 (Probability Distribution)

* **정규분포 (Normal Distribution)**: 데이터가 평균을 중심으로 대칭적인 종 모양을 이루는 분포입니다.

---

## 2. 기술 통계 (Descriptive Statistics) (25분)

### 중심 경향성 및 분산도

R에서는 별도의 라이브러리 없이도 기초 통계량을 쉽게 계산할 수 있습니다.

```r
# 평균, 중앙값
mean(df$likes)
median(df$likes)

# 분산, 표준편차
var(df$likes)
sd(df$likes)

# 사분위수
quantile(df$likes, probs = c(0.25, 0.5, 0.75))

```

### Tidyverse를 활용한 그룹별 요약

```r
library(tidyverse)

df <- read_csv('advertising_experiment.csv')

# 1. 전체 데이터 요약 (Python의 describe와 유사)
summary(df)

# 2. 그룹별 통계 (광고 유형별 성과 요약)
df %>%
  group_by(ad_type) %>%
  summarise(
    n = n(),
    mean_eng = mean(engagement, na.rm = TRUE),
    sd_eng = sd(engagement, na.rm = TRUE),
    median_conv = median(conversion_rate, na.rm = TRUE)
  )

```

---

## 3. 가설 검정 (Hypothesis Testing) (40분)

### 가설 검정의 핵심: p-value

* **p < 0.05**: "우연히 이런 결과가 나타날 확률이 5% 미만이다." → **귀무가설 기각 (유의미한 차이 있음)**

### t-test (두 집단 비교)

#### 독립 표본 t-test (성별에 따른 참여도 차이)

```r
# 공식(Formula) 문법: 종속변수 ~ 독립변수
t_result <- t.test(engagement ~ gender, data = df)

print(t_result)
# p-value 확인: t_result$p.value

```

### ANOVA (셋 이상 집단 비교)

광고 유형(이미지, 영상, 카드뉴스 등)에 따라 클릭률에 차이가 있는지 분석합니다.

```r
# ANOVA 실행
anova_result <- aov(engagement ~ ad_type, data = df)
summary(anova_result)

# p-value가 0.05보다 작다면 사후 분석(Tukey HSD) 실시
if (summary(anova_result)[[1]][["Pr(>F)"]][1] < 0.05) {
  tukey <- TukeyHSD(anova_result)
  plot(tukey) # 그룹 간 차이 시각화
}

```

---

## 4. 상관관계와 회귀 분석 (35분)

### 상관관계 (Correlation)

두 수치형 변수가 얼마나 밀접하게 움직이는지 나타냅니다.

```r
# 상관계수 계산
cor(df$likes, df$shares, method = "pearson")

# 상관관계 행렬 시각화
library(GGally)
df %>%
  select(likes, shares, comments, time_spent) %>%
  ggpairs() # 산점도와 상관계수를 한 번에!

```

### 단순 선형 회귀 (Linear Regression)

하나의 독립변수()가 종속변수()에 미치는 영향을 수식화합니다: 

```r
# 회귀 모델 생성
model <- lm(shares ~ likes, data = df)

# 결과 확인
summary(model)

# 회귀 시각화
ggplot(df, aes(x = likes, y = shares)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", color = "red") + # 회귀선 추가
  labs(title = paste("R-squared:", round(summary(model)$r.squared, 3)))

```

### 다중 회귀 (Multiple Regression)

```r
# 여러 변수를 동시에 고려: likes, comments, time_spent가 engagement_score에 미치는 영향
multi_model <- lm(engagement_score ~ likes + comments + time_spent, data = df)

summary(multi_model)

# broom 패키지를 사용해 결과를 표로 정리하기
library(broom)
tidy(multi_model) # 계수 확인
glance(multi_model) # 모델 전체 성능(R-squared 등) 확인

```

---

## 💡 핵심 요약

### R 통계 분석 필수 함수

| 분석 목적 | R 함수 | 비고 |
| --- | --- | --- |
| **기술 통계** | `summary()`, `sd()`, `quantile()` | 기본 내장 |
| **그룹 요약** | `group_by() %>% summarise()` | tidyverse |
| **두 집단 비교** | `t.test(y ~ x)` | 수식(Formula) 기반 |
| **다집단 비교** | `aov(y ~ x)` | `TukeyHSD()` 연계 |
| **상관관계** | `cor()`, `ggpairs()` | 시각화 권장 |
| **회귀 분석** | `lm(y ~ x1 + x2)` | Linear Model의 약자 |

---

## 🤖 AI 협업 가이드 (R 버전)

```text
1. 분석 설계:
   "R로 3개 이상의 플랫폼별 좋아요 수 차이를 ANOVA로 분석하고 싶어. 
   코드와 사후 분석 방법까지 알려줘."

2. 시각화 개선:
   "ggplot2로 회귀 분석 산점도를 그렸는데, 여기에 회귀 방정식 수식을 
   그래프 위에 텍스트로 올리는 법을 알려줘."

3. 에러 해결:
   "lm() 함수에서 'variable lengths differ' 에러가 났어. 원인과 해결책은?"

4. 통계적 가정 확인:
   "회귀 분석 후에 잔차의 정규성과 등분산성을 확인하는 gvlma 패키지 사용법 알려줘."

```

---

## 📝 다음 수업 예고

**Session 4: 텍스트 분석 & 워드클라우드**

* 한국어 형태소 분석 (`KoNLP`, `tidytext`)
* 워드클라우드 시각화
* 감성 분석과 토픽 모델링 기초

---

**R은 통계 분석의 표준입니다. AI와 함께라면 데이터 속에 숨겨진 진실을 더 정확하게 찾을 수 있습니다! 📊**

---

**이 강의안을 실제 수업에서 바로 활용하실 수 있도록, 수업 중에 사용할 예제 데이터셋을 생성하는 R 코드를 만들어 드릴까요?**