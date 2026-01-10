# Session 2: 데이터 시각화 마스터하기 (R tidyverse 버전)
# RStudio 또는 Jupyter Notebook (R Kernel)에서 실행하세요.

# ============================================
# Part 1: 환경 설정 및 데이터 생성
# ============================================

library(tidyverse)
library(patchwork) # 대시보드 구성을 위한 패키지
library(zoo) # 이동 평균 계산(rollmean)을 위한 패키지
library(GGally) # 페어플롯(ggpairs)을 위한 패키지

# 테마 설정 (Seaborn의 whitegrid와 유사)
theme_set(theme_minimal(base_size = 12))

cat("✅ 라이브러리 로드 완료!\n")

# 데이터 로드 (파일이 없을 경우를 대비해 가상 데이터 생성)
# df <- read_csv('./datasets/news_consumption.csv')

set.seed(42)
n <- 200
df <- tibble(
    category = sample(c("Politics", "Technology", "Entertainment", "Sports", "Health"), n, replace = TRUE),
    time_spent = rnorm(n, 45, 15),
    articles_read = rpois(n, 5),
    engagement_score = (time_spent * 0.5) + (articles_read * 2) + rnorm(n, 0, 5),
    age = sample(18:70, n, replace = TRUE),
    device = sample(c("Mobile", "Desktop", "Tablet"), n, replace = TRUE),
    age_group = case_when(
        age < 30 ~ "18-29",
        age < 40 ~ "30-39",
        age < 50 ~ "40-49",
        TRUE ~ "50+"
    )
)

# ============================================
# Part 2: ggplot2 기초 (Line, Bar, Scatter, Hist)
# ============================================

cat("\n📈 Part 2: 기본 차트 생성 중...\n")

# 예제 2-1: 선 그래프 (이동 평균 포함)
days <- 1:30
consumption <- 50 + 10 * sin(days / 5) + rnorm(30, 0, 3)
line_data <- tibble(day = days, val = consumption) %>%
    mutate(moving_avg = rollmean(val, k = 7, fill = NA, align = "right"))

p_line <- ggplot(line_data, aes(x = day)) +
    geom_line(aes(y = val), color = "#2E86AB", size = 1) +
    geom_point(aes(y = val), color = "#2E86AB") +
    geom_line(aes(y = moving_avg), color = "#A23B72", linetype = "dashed", size = 1.2) +
    labs(title = "News Consumption Trend (30 Days)", x = "Day", y = "Articles Read")

# 예제 2-2: 막대그래프 (정렬 및 값 표시)
category_summary <- df %>%
    group_by(category) %>%
    summarize(mean_read = mean(articles_read))

p_bar <- ggplot(category_summary, aes(x = reorder(category, -mean_read), y = mean_read, fill = category)) +
    geom_col(color = "black", show.legend = FALSE) +
    geom_text(aes(label = round(mean_read, 1)), vjust = -0.5, fontface = "bold") +
    scale_fill_viridis_d(option = "D", begin = 0.3, end = 0.9) +
    labs(title = "Average Articles Read by Category", x = "Category", y = "Avg Articles")

# 예제 2-3: 산점도
p_scatter <- ggplot(df, aes(x = time_spent, y = engagement_score, color = age)) +
    geom_point(size = 3, alpha = 0.6) +
    scale_color_gradient(low = "blue", high = "red") +
    labs(title = "Reading Time vs Engagement Score", x = "Time Spent (min)", color = "Age")

# 예제 2-4: 히스토그램 (단순 & 누적)
p_hist1 <- ggplot(df, aes(x = time_spent)) +
    geom_histogram(bins = 30, fill = "skyblue", color = "black", alpha = 0.7) +
    geom_vline(aes(xintercept = mean(time_spent)), color = "red", linetype = "dashed") +
    labs(title = "Distribution of Reading Time")

p_hist2 <- ggplot(df, aes(x = time_spent)) +
    stat_bin(aes(y = cumsum(..count..)), bins = 30, geom = "area", fill = "coral", alpha = 0.7) +
    labs(title = "Cumulative Distribution")

# ============================================
# Part 3: 고급 시각화
# ============================================

cat("🎨 Part 3: 고급 시각화 생성 중...\n")

# 예제 3-1: 박스플롯 & 바이올린 플롯
p_box <- ggplot(df, aes(x = device, y = time_spent, fill = device)) +
    geom_boxplot(alpha = 0.7) +
    scale_fill_brewer(palette = "Set2")

p_violin <- ggplot(df, aes(x = device, y = time_spent, fill = device)) +
    geom_violin(alpha = 0.7) +
    scale_fill_brewer(palette = "Set3")

# 예제 3-2: 신뢰구간 포함 막대그래프
p_ci_bar <- ggplot(df, aes(x = category, y = engagement_score, fill = category)) +
    stat_summary(fun = mean, geom = "bar", alpha = 0.8) +
    stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 예제 3-4: 히트맵 (상관관계)
cor_matrix <- df %>%
    select(time_spent, articles_read, engagement_score, age) %>%
    cor()
p_heatmap <- as_tibble(cor_matrix, rownames = "var1") %>%
    pivot_longer(-var1, names_to = "var2", values_to = "corr") %>%
    ggplot(aes(var1, var2, fill = corr)) +
    geom_tile() +
    geom_text(aes(label = round(corr, 2))) +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", limit = c(-1, 1)) +
    labs(title = "Correlation Heatmap")

# 예제 3-6: 회귀 플롯
p_reg <- ggplot(df, aes(x = articles_read, y = engagement_score)) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = "lm", color = "red", se = TRUE) +
    labs(title = "Articles Read vs Engagement Score (with LM)")

# ============================================
# Part 4: 스토리보드 (patchwork 활용)
# ============================================

cat("📊 Part 4: 대시보드 구성 중...\n")

# patchwork를 이용한 레이아웃 구성
dashboard <- (p_bar + p_hist1 + p_scatter) / (p_box + p_heatmap + p_reg) +
    plot_annotation(
        title = "News Consumption Pattern Analysis Dashboard",
        theme = theme(plot.title = element_text(size = 20, face = "bold"))
    )

ggsave("session2_dashboard_r.png", dashboard, width = 16, height = 12)

# ============================================
# Part 5: 주석과 강조
# ============================================

p_ann <- ggplot(category_summary, aes(x = reorder(category, -mean_read), y = mean_read)) +
    geom_col(aes(fill = mean_read == max(mean_read)), show.legend = FALSE) +
    scale_fill_manual(values = c("gray70", "#FF6B6B")) +
    annotate("label",
        x = 1.5, y = max(category_summary$mean_read),
        label = "Highest Content\nEngagement!", color = "red", fontface = "bold"
    ) +
    labs(title = "Engagement Score by Category (Highlighted)")

print(p_ann)

cat("\n✅ 모든 시각화 및 대시보드 완성! RStudio의 Plots 창을 확인하세요.\n")
