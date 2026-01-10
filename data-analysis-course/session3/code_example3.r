# ============================================
# Session 3: 통계 분석 기초
# 라이브러리 로드
# ============================================
library(tidyverse)
library(ggpubr) # 통계 기반 시각화 및 t-test/ANOVA 간편화
library(broom) # 모델 결과를 깔끔한 표(Tibble)로 변환
library(patchwork) # 여러 그래프 배치
library(corrplot) # 상관관계 히트맵

cat("============================================================\n")
cat("Session 3: 통계 분석 기초\n")
cat("============================================================\n")

# 데이터 로드 (실습용 가상 데이터 생성 포함)
# df <- read_csv('./datasets/advertising_experiment.csv')

# 코드 실행을 위한 가상 데이터 생성
set.seed(42)
df <- tibble(
    ad_type = sample(c("Social", "Search", "Display", "Video"), 200, replace = TRUE),
    gender = sample(c("Male", "Female"), 200, replace = TRUE),
    engagement = rnorm(200, 500, 100),
    conversion_rate = (engagement * 0.0001) + rnorm(200, 0.02, 0.005),
    cost_per_click = runif(200, 0.5, 2.5),
    age_group = sample(c("18-24", "25-34", "35-44", "45+"), 200, replace = TRUE)
)

cat(sprintf("\n데이터 로드 완료: %d행 %d열\n", nrow(df), ncol(df)))
print(head(df))

# ============================================
# Part 1: 기술 통계
# ============================================
cat("\n============================================================\n")
cat("Part 1: 기술 통계\n")
cat("============================================================\n")

# 기본 기술 통계
cat("\n기본 기술 통계:\n")
print(summary(df))

# 그룹별 통계 (tidyverse 방식)
cat("\n광고 유형별 통계:\n")
ad_stats <- df %>%
    group_by(ad_type) %>%
    summarise(
        count = n(),
        mean_eng = mean(engagement),
        median_eng = median(engagement),
        sd_eng = sd(engagement),
        mean_conv = mean(conversion_rate),
        mean_cpc = mean(cost_per_click)
    ) %>%
    mutate(across(where(is.numeric), ~ round(., 2)))
print(ad_stats)

# ============================================
# Part 2: t-test (두 집단 비교)
# ============================================
cat("\n============================================================\n")
cat("Part 2: t-test (성별에 따른 참여도 비교)\n")
cat("============================================================\n")

t_result <- t.test(engagement ~ gender, data = df)
print(t_result)

# p-value 추출 및 해석
p_val_t <- t_result$p.value
if (p_val_t < 0.05) {
    cat("✅ 성별에 따라 참여도에 유의미한 차이가 있습니다\n")
} else {
    cat("❌ 성별에 따른 유의미한 차이를 발견하지 못했습니다\n")
}

# 시각화 (ggpubr 사용)
p_ttest <- ggboxplot(df,
    x = "gender", y = "engagement",
    fill = "gender", palette = "jco",
    add = "jitter",
    title = sprintf("t-test: Male vs Female (p=%.4f)", p_val_t)
)

# ============================================
# Part 3 & 4: ANOVA 및 사후 분석
# ============================================
cat("\n============================================================\n")
cat("Part 3: ANOVA (광고 유형별 전환율 비교)\n")
cat("============================================================\n")

anova_model <- aov(conversion_rate ~ ad_type, data = df)
anova_summary <- summary(anova_model)
print(anova_summary)

p_val_anova <- anova_summary[[1]][["Pr(>F)"]][1]

# 시각화
p_anova <- ggviolin(df,
    x = "ad_type", y = "conversion_rate",
    fill = "ad_type", palette = "npg",
    add = "boxplot",
    title = sprintf("ANOVA: Conversion Rate by Ad Type (p=%.4f)", p_val_anova)
)

if (p_val_anova < 0.05) {
    cat("\n✅ 광고 유형 간 유의미한 차이가 있습니다 (Tukey HSD 사후 분석 실시)\n")
    tukey_result <- TukeyHSD(anova_model)
    print(tukey_result)
}

# ============================================
# Part 5: 상관관계 분석
# ============================================
cat("\n============================================================\n")
cat("Part 5: 상관관계 분석\n")
cat("============================================================\n")

numeric_df <- df %>% select(where(is.numeric))
corr_matrix <- cor(numeric_df, use = "complete.obs")
print(round(corr_matrix, 3))

# 히트맵 시각화
corrplot(corr_matrix,
    method = "color", type = "upper",
    addCoef.col = "black", tl.col = "black",
    title = "\n\nCorrelation Heatmap", mar = c(0, 0, 1, 0)
)

# 피어슨 상관계수 검정
cor_test <- cor.test(df$engagement, df$conversion_rate)
cat(sprintf(
    "\n참여도 vs 전환율:\n - 상관계수 (r): %.4f\n - p-value: %.4e\n",
    cor_test$estimate, cor_test$p.value
))

# ============================================
# Part 6: 단순 선형 회귀
# ============================================
cat("\n============================================================\n")
cat("Part 6: 단순 선형 회귀\n")
cat("============================================================\n")

lm_simple <- lm(conversion_rate ~ engagement, data = df)
summary_simple <- summary(lm_simple)
print(summary_simple)

# 시각화
p_reg <- ggplot(df, aes(x = engagement, y = conversion_rate)) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = "lm", color = "red") +
    stat_regline_equation(label.y = max(df$conversion_rate)) +
    labs(title = sprintf("Linear Regression (R² = %.4f)", summary_simple$r.squared)) +
    theme_minimal()

# ============================================
# Part 7: 다중 회귀 분석
# ============================================
cat("\n============================================================\n")
cat("Part 7: 다중 회귀 분석\n")
cat("============================================================\n")

lm_multi <- lm(conversion_rate ~ engagement + cost_per_click, data = df)
print(tidy(lm_multi)) # broom 패키지로 깔끔하게 출력
cat(sprintf("\nR-squared: %.4f\n", summary(lm_multi)$r.squared))

# 예측 예제
new_data <- tibble(engagement = 500, cost_per_click = 1.5)
prediction <- predict(lm_multi, newdata = new_data)
cat(sprintf("\n예측 전환율 (eng=500, cpc=1.5): %.4f\n", prediction))

# ============================================
# Part 8: 종합 대시보드
# ============================================
# patchwork 패키지를 사용하여 그래프 병합
dashboard <- (p_ttest + p_anova) / (p_reg + p_anova) # p_anova 중복 배치 예시
# ggsave("session3_dashboard_r.png", dashboard, width = 12, height = 10)
