# Session 1: AI 기반 데이터 분석 입문
# Code Examples - Jupyter Notebook으로 실행하세요

# ============================================
# Part 1: 환경 설정 및 라이브러리 임포트
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (MacOS/Linux)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("✅ 라이브러리 임포트 완료!")

# ============================================
# Part 2: 데이터 로드 및 기본 탐색
# ============================================

# 데이터 읽기
df = pd.read_csv('../datasets/social_media_engagement.csv')

print("=" * 50)
print("📊 데이터 미리보기")
print("=" * 50)
print(df.head(10))

print("\n" + "=" * 50)
print("📋 데이터 정보")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("📈 기술 통계")
print("=" * 50)
print(df.describe())

# AI 프롬프트 예시:
# "이 데이터셋의 각 컬럼이 무엇을 의미하는지 설명해줘"

# ============================================
# Part 3: 데이터 선택 및 필터링
# ============================================

print("\n" + "=" * 50)
print("🔍 데이터 선택 예제")
print("=" * 50)

# 1. 단일 컬럼 선택
print("\n1. 플랫폼 컬럼만 선택:")
print(df['platform'].head())

# 2. 여러 컬럼 선택
print("\n2. 플랫폼과 좋아요 컬럼 선택:")
print(df[['platform', 'likes']].head())

# 3. 조건 필터링 - 좋아요가 500 이상인 포스트
print("\n3. 좋아요 500개 이상인 포스트:")
high_engagement = df[df['likes'] >= 500]
print(f"전체 {len(df)}개 중 {len(high_engagement)}개 포스트")
print(high_engagement.head())

# 4. 여러 조건 필터링
print("\n4. Instagram이면서 좋아요 500 이상:")
instagram_high = df[(df['platform'] == 'Instagram') & (df['likes'] >= 500)]
print(f"{len(instagram_high)}개 포스트")

# AI 프롬프트 예시:
# "20-29세 연령대의 평균 공유 수를 계산하는 코드를 짜줘"

# ============================================
# Part 4: 데이터 집계 (Groupby)
# ============================================

print("\n" + "=" * 50)
print("📊 데이터 집계 - GroupBy")
print("=" * 50)

# 1. 플랫폼별 평균 좋아요
print("\n1. 플랫폼별 평균 좋아요:")
platform_avg = df.groupby('platform')['likes'].mean().sort_values(ascending=False)
print(platform_avg)

# 2. 플랫폼별 여러 통계
print("\n2. 플랫폼별 종합 통계:")
platform_stats = df.groupby('platform').agg({
    'likes': ['mean', 'median', 'max'],
    'shares': 'sum',
    'comments': 'mean'
})
print(platform_stats)

# 3. 연령대별 참여도
print("\n3. 연령대별 평균 참여도:")
age_engagement = df.groupby('age_group').agg({
    'likes': 'mean',
    'shares': 'mean',
    'comments': 'mean'
}).round(2)
print(age_engagement)

# 4. 플랫폼 & 연령대 교차 분석
print("\n4. 플랫폼-연령대 교차 분석:")
cross_analysis = df.groupby(['platform', 'age_group'])['likes'].mean().round(2)
print(cross_analysis)

# ============================================
# Part 5: 간단한 시각화
# ============================================

print("\n" + "=" * 50)
print("📈 데이터 시각화")
print("=" * 50)

# 1. 플랫폼별 평균 좋아요 막대그래프
plt.figure(figsize=(10, 6))
platform_avg.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Likes by Platform', fontsize=16, fontweight='bold')
plt.xlabel('Platform', fontsize=12)
plt.ylabel('Average Likes', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('session1_platform_likes.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 연령대별 참여도 비교
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

age_engagement['likes'].plot(kind='bar', ax=axes[0], color='coral')
axes[0].set_title('Average Likes by Age Group')
axes[0].set_ylabel('Likes')
axes[0].tick_params(axis='x', rotation=45)

age_engagement['shares'].plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Average Shares by Age Group')
axes[1].set_ylabel('Shares')
axes[1].tick_params(axis='x', rotation=45)

age_engagement['comments'].plot(kind='bar', ax=axes[2], color='plum')
axes[2].set_title('Average Comments by Age Group')
axes[2].set_ylabel('Comments')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('session1_age_engagement.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 좋아요 분포 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(df['likes'], bins=30, color='teal', alpha=0.7, edgecolor='black')
plt.axvline(df['likes'].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: {df["likes"].mean():.0f}')
plt.title('Distribution of Likes', fontsize=16, fontweight='bold')
plt.xlabel('Number of Likes', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('session1_likes_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 시각화 완료! 이미지 파일 저장됨")

# ============================================
# Part 6: 실전 분석 예제
# ============================================

print("\n" + "=" * 50)
print("🎯 실전 분석: 포스트 시간대 분석")
print("=" * 50)

# 시간대별 참여도 분석
if 'post_hour' in df.columns:
    hourly_engagement = df.groupby('post_hour').agg({
        'likes': 'mean',
        'shares': 'mean',
        'comments': 'mean'
    }).round(2)

    print("\n시간대별 평균 참여도:")
    print(hourly_engagement)

    # 시간대별 트렌드 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_engagement.index, hourly_engagement['likes'],
             marker='o', linewidth=2, markersize=8, label='Likes')
    plt.plot(hourly_engagement.index, hourly_engagement['shares'],
             marker='s', linewidth=2, markersize=8, label='Shares')
    plt.plot(hourly_engagement.index, hourly_engagement['comments'],
             marker='^', linewidth=2, markersize=8, label='Comments')

    plt.title('Engagement Trends by Hour of Day', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Engagement', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig('session1_hourly_trends.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================
# Part 7: AI 협업 실습
# ============================================

print("\n" + "=" * 50)
print("🤖 AI 협업 프롬프트 예제")
print("=" * 50)

print("""
다음 프롬프트를 AI에게 시도해보세요:

1. 데이터 이해:
   "이 데이터셋에서 가장 흥미로운 인사이트 3가지를 찾아줘"

2. 코드 생성:
   "30-39세 연령대에서 가장 인기있는 플랫폼을 찾고 시각화하는 코드를 짜줘"

3. 오류 해결:
   "KeyError: 'platfrom' 오류가 났어. 어떻게 고쳐야 해?"

4. 코드 설명:
   "df.groupby('platform')['likes'].agg(['mean', 'std'])가 무엇을 하는 건지 설명해줘"

5. 개선 제안:
   "이 코드를 더 효율적으로 만들 수 있을까?"

6. 해석 도움:
   "Instagram의 평균 좋아요가 다른 플랫폼보다 2배 높은데,
    이것이 커뮤니케이션 전략에 어떤 의미가 있을까?"
""")

# ============================================
# Part 8: 종합 실습 - 나만의 분석
# ============================================

print("\n" + "=" * 50)
print("📝 종합 실습 과제")
print("=" * 50)

print("""
다음 질문들에 답하는 코드를 작성해보세요:

1. 어느 플랫폼이 댓글 대비 좋아요 비율이 가장 높은가?
2. 주말과 평일의 평균 참여도 차이는?
3. 가장 성공적인 포스트의 특징은?
4. 연령대별로 선호하는 플랫폼이 다른가?

💡 AI 활용 팁:
- 막히면 AI에게 질문하세요
- 코드를 실행하고 결과를 해석하세요
- 시각화를 추가해서 더 명확하게 만드세요
""")

# 예시 솔루션 1: 댓글 대비 좋아요 비율
df['like_to_comment_ratio'] = df['likes'] / (df['comments'] + 1)  # +1로 0으로 나누기 방지
ratio_by_platform = df.groupby('platform')['like_to_comment_ratio'].mean().sort_values(ascending=False)

print("\n💡 예시 답변 1: 플랫폼별 좋아요/댓글 비율")
print(ratio_by_platform)

# ============================================
# 마무리
# ============================================

print("\n" + "=" * 50)
print("🎉 Session 1 완료!")
print("=" * 50)
print("""
오늘 배운 것:
✅ Pandas로 데이터 읽기 및 탐색
✅ 데이터 필터링 및 선택
✅ GroupBy로 데이터 집계
✅ 기본 시각화
✅ AI와 협업하는 방법

다음 시간:
📊 고급 시각화 기법
📈 효과적인 차트 디자인
🎨 Seaborn을 활용한 비주얼 커뮤니케이션

계속 연습하고, AI에게 질문하세요! 🚀
""")
