# Session 1: AI 기반 데이터 분석 입문 (R tidyverse 버전)
# Code Examples - RStudio 또는 Jupyter Notebook(R Kernel)에서 실행하세요

# ============================================
# Part 1: 환경 설정 및 라이브러리 로드
# ============================================

# tidyverse 설치가 안 되어 있다면: install.packages("tidyverse")
library(tidyverse)

library(patchwork) # 여러 그래프를 한 화면에 배치하기 위함

# 한글 폰트 설정 (MacOS/Linux의 경우 보통 "NanumGothic" 등 설치된 폰트명 사용)
# Windows 사용자는 "Malgun Gothic" 권장
theme_set(theme_minimal(base_family = "sans")) 

cat("✅ 라이브러리 로드 완료!\n")

# ============================================
# Part 2: 데이터 로드 및 기본 탐색
# ============================================

print(getwd())

# 데이터 읽기 (파일이 없으므로 예시 경로 유지)
# df <- read_csv('datasets/social_media_engagement.csv')

# 실습을 위한 가상 데이터 생성 (코드 실행 확인용)
set.seed(42)
df <- tibble(
  platform = sample(c("Instagram", "Facebook", "Twitter", "TikTok"), 100, replace = TRUE),
  likes = rpois(100, lambda = 400),
  shares = rpois(100, lambda = 50),
  comments = rpois(100, lambda = 30),
  age_group = sample(c("18-24", "25-34", "35-44", "45+"), 100, replace = TRUE),
  post_hour = sample(0:23, 100, replace = TRUE)
)

cat("==================================================\n")
cat("📊 데이터 미리보기\n")
cat("==================================================\n")
print(head(df, 10))

cat("\n==================================================\n")
cat("📋 데이터 정보\n")
cat("==================================================\n")
glimpse(df) # Python의 info()와 유사

cat("\n==================================================\n")
cat("📈 기술 통계\n")
cat("==================================================\n")
print(summary(df))

# ============================================
# Part 3: 데이터 선택 및 필터링
# ============================================

cat("\n==================================================\n")
cat("🔍 데이터 선택 예제\n")
cat("==================================================\n")

# 1. 단일 컬럼 선택
cat("\n1. 플랫폼 컬럼만 선택:\n")
df %>% select(platform) %>% head() %>% print()

# 2. 여러 컬럼 선택
cat("\n2. 플랫폼과 좋아요 컬럼 선택:\n")
df %>% select(platform, likes) %>% head() %>% print()

# 3. 조건 필터링 - 좋아요가 500 이상인 포스트
cat("\n3. 좋아요 500개 이상인 포스트:\n")
high_engagement <- df %>% filter(likes >= 500)
cat(sprintf("전체 %d개 중 %d개 포스트\n", nrow(df), nrow(high_engagement)))
high_engagement %>% head() %>% print()

# 4. 여러 조건 필터링
cat("\n4. Instagram이면서 좋아요 500 이상:\n")
instagram_high <- df %>% filter(platform == "Instagram", likes >= 500)
cat(sprintf("%d개 포스트\n", nrow(instagram_high)))

# ============================================
# Part 4: 데이터 집계 (Groupby)
# ============================================

cat("\n==================================================\n")
cat("📊 데이터 집계 - GroupBy\n")
cat("==================================================\n")

# 1. 플랫폼별 평균 좋아요
cat("\n1. 플랫폼별 평균 좋아요:\n")
platform_avg <- df %>%
  group_by(platform) %>%
  summarize(mean_likes = mean(likes)) %>%
  arrange(desc(mean_likes))
print(platform_avg)

# 2. 플랫폼별 여러 통계
cat("\n2. 플랫폼별 종합 통계:\n")
platform_stats <- df %>%
  group_by(platform) %>%
  summarize(
    mean_likes = mean(likes),
    median_likes = median(likes),
    max_likes = max(likes),
    total_shares = sum(shares),
    mean_comments = mean(comments)
  )
print(platform_stats)

# 3. 연령대별 참여도
cat("\n3. 연령대별 평균 참여도:\n")
age_engagement <- df %>%
  group_by(age_group) %>%
  summarize(across(c(likes, shares, comments), mean)) %>%
  mutate(across(where(is.numeric), ~round(., 2)))
print(age_engagement)

# 4. 플랫폼 & 연령대 교차 분석
cat("\n4. 플랫폼-연령대 교차 분석:\n")
cross_analysis <- df %>%
  group_by(platform, age_group) %>%
  summarize(mean_likes = mean(likes), .groups = 'drop') %>%
  mutate(mean_likes = round(mean_likes, 2))
print(cross_analysis)

# ============================================
# Part 5: 간단한 시각화
# ============================================

cat("\n==================================================\n")
cat("📈 데이터 시각화\n")
cat("==================================================\n")

# 1. 플랫폼별 평균 좋아요 막대그래프
p1 <- ggplot(platform_avg, aes(x = reorder(platform, -mean_likes), y = mean_likes)) +
  geom_col(fill = "skyblue", color = "black") +
  labs(title = "Average Likes by Platform", x = "Platform", y = "Average Likes") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p1)
ggsave("session1_platform_likes_r.png", width = 10, height = 6)

# 2. 연령대별 참여도 비교 (patchwork 활용)
# R에서는 pivot_longer를 사용하여 한번에 그리는 것이 더 'tidy'합니다.
age_long <- age_engagement %>%
  pivot_longer(cols = c(likes, shares, comments), names_to = "metric", values_to = "value")

p2 <- ggplot(age_long, aes(x = age_group, y = value, fill = metric)) +
  geom_col() +
  facet_wrap(~metric, scales = "free_y") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Average Engagement by Age Group")

print(p2)
ggsave("session1_age_engagement_r.png", width = 15, height = 5)

# 3. 좋아요 분포 히스토그램
mean_likes_val <- mean(df$likes)
p3 <- ggplot(df, aes(x = likes)) +
  geom_histogram(bins = 30, fill = "teal", color = "black", alpha = 0.7) +
  geom_vline(xintercept = mean_likes_val, color = "red", linetype = "dashed", size = 1) +
  annotate("text", x = mean_likes_val, y = 5, label = paste("Mean:", round(mean_likes_val)), color = "red", hjust = -0.1) +
  labs(title = "Distribution of Likes", x = "Number of Likes", y = "Frequency")

print(p3)
ggsave("session1_likes_distribution_r.png", width = 10, height = 6)

cat("✅ 시각화 완료! 이미지 파일 저장됨\n")

# ============================================
# Part 6: 실전 분석 예제
# ============================================

cat("\n==================================================\n")
cat("🎯 실전 분석: 포스트 시간대 분석\n")
cat("==================================================\n")

if ("post_hour" %in% names(df)) {
  hourly_engagement <- df %>%
    group_by(post_hour) %>%
    summarize(across(c(likes, shares, comments), mean)) %>%
    mutate(across(where(is.numeric), ~round(., 2)))

  cat("\n시간대별 평균 참여도:\n")
  print(hourly_engagement)

  # 시간대별 트렌드 시각화
  p4 <- hourly_engagement %>%
    pivot_longer(cols = -post_hour, names_to = "metric", values_to = "value") %>%
    ggplot(aes(x = post_hour, y = value, color = metric, shape = metric)) +
    geom_line(size = 1) +
    geom_point(size = 3) +
    scale_x_continuous(breaks = seq(0, 24, 2)) +
    labs(title = "Engagement Trends by Hour of Day", x = "Hour of Day", y = "Average Engagement") +
    theme_minimal()

  print(p4)
  ggsave("session1_hourly_trends_r.png", width = 12, height = 6)
}

# ============================================
# Part 8: 종합 실습 - 나만의 분석
# ============================================

# 예시 솔루션 1: 댓글 대비 좋아요 비율
df <- df %>%
  mutate(like_to_comment_ratio = likes / (comments + 1))

ratio_by_platform <- df %>%
  group_by(platform) %>%
  summarize(avg_ratio = mean(like_to_comment_ratio)) %>%
  arrange(desc(avg_ratio))

cat("\n💡 예시 답변 1: 플랫폼별 좋아요/댓글 비율\n")
print(ratio_by_platform)

cat("\n🎉 Session 1 완료! (R tidyverse 버전)\n")
