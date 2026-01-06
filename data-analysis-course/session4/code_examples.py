# Session 4: 텍스트 분석 & 워드클라우드
# Code Examples

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re

# NLTK 라이브러리
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# 최초 1회 다운로드 (주석 해제하여 실행)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')

print("=" * 60)
print("Session 4: 텍스트 분석 & 워드클라우드")
print("=" * 60)

# ============================================
# Part 1: 텍스트 전처리
# ============================================

print("\n" + "=" * 60)
print("Part 1: 텍스트 전처리")
print("=" * 60)

# 샘플 텍스트
sample_text = """
I absolutely LOVE this product! It's amazing and works perfectly.
The customer service was excellent. Highly recommended!!! 😊😊😊
#BestPurchase #HappyCustomer
"""

print(f"\n원본 텍스트:\n{sample_text}")

# 전처리 함수
def preprocess_text(text, remove_numbers=True):
    # 소문자 변환
    text = text.lower()

    # URL 제거
    text = re.sub(r'http\S+|www\S+', '', text)

    # 이메일 제거
    text = re.sub(r'\S+@\S+', '', text)

    # 해시태그 기호 제거 (단어는 유지)
    text = re.sub(r'#', '', text)

    # 이모지 및 특수문자 제거
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # 숫자 제거 (옵션)
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # 토큰화
    tokens = word_tokenize(text)

    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 표제어 추출
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens

# 전처리 실행
cleaned_tokens = preprocess_text(sample_text)
print(f"\n전처리 결과: {cleaned_tokens}")

# ============================================
# Part 2: 데이터 로드 및 대량 전처리
# ============================================

print("\n" + "=" * 60)
print("Part 2: 소셜 미디어 댓글 데이터 분석")
print("=" * 60)

# 데이터 로드
df = pd.read_csv('../datasets/social_media_comments.csv')
print(f"\n데이터 로드: {df.shape}")
print(df.head())

# 모든 댓글 전처리
df['cleaned_tokens'] = df['comment'].apply(preprocess_text)
df['cleaned_text'] = df['cleaned_tokens'].apply(lambda x: ' '.join(x))

print("\n전처리 예시:")
for i in range(3):
    print(f"\n원본: {df.iloc[i]['comment']}")
    print(f"전처리: {df.iloc[i]['cleaned_text']}")

# ============================================
# Part 3: 워드클라우드 생성
# ============================================

print("\n" + "=" * 60)
print("Part 3: 워드클라우드")
print("=" * 60)

# 전체 댓글 합치기
all_text = ' '.join(df['cleaned_text'])

# 기본 워드클라우드
wordcloud = WordCloud(width=1200, height=600,
                      background_color='white',
                      colormap='viridis',
                      max_words=100).generate(all_text)

plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud: Most Common Words in Comments',
          fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('session4_wordcloud_all.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 워드클라우드 생성 완료")

# 빈도수 분석
all_words = []
for tokens in df['cleaned_tokens']:
    all_words.extend(tokens)

word_freq = Counter(all_words)
print("\n가장 빈번한 단어 Top 20:")
for word, count in word_freq.most_common(20):
    print(f"  {word}: {count}")

# ============================================
# Part 4: 감성 분석
# ============================================

print("\n" + "=" * 60)
print("Part 4: 감성 분석")
print("=" * 60)

# VADER 감성 분석기
sia = SentimentIntensityAnalyzer()

# 감성 점수 계산
df['sentiment_scores'] = df['comment'].apply(lambda x: sia.polarity_scores(x))
df['sentiment_compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
df['sentiment_pos'] = df['sentiment_scores'].apply(lambda x: x['pos'])
df['sentiment_neg'] = df['sentiment_scores'].apply(lambda x: x['neg'])
df['sentiment_neu'] = df['sentiment_scores'].apply(lambda x: x['neu'])

# 범주화
def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['sentiment_compound'].apply(categorize_sentiment)

# 감성 분포
print("\n감성 분포:")
print(df['sentiment'].value_counts())
print(f"\n긍정 비율: {(df['sentiment'] == 'Positive').sum() / len(df) * 100:.1f}%")
print(f"부정 비율: {(df['sentiment'] == 'Negative').sum() / len(df) * 100:.1f}%")
print(f"중립 비율: {(df['sentiment'] == 'Neutral').sum() / len(df) * 100:.1f}%")

# 감성 분포 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 카운트 플롯
sns.countplot(x='sentiment', data=df, ax=axes[0],
              palette={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'})
axes[0].set_title('Sentiment Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')

# 점수 분포 히스토그램
axes[1].hist(df['sentiment_compound'], bins=50, color='skyblue', edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Neutral')
axes[1].axvline(0.05, color='green', linestyle='--', linewidth=2, alpha=0.5)
axes[1].axvline(-0.05, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[1].set_xlabel('Sentiment Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Sentiment Scores', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('session4_sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# Part 5: 플랫폼별 감성 비교
# ============================================

print("\n" + "=" * 60)
print("Part 5: 플랫폼별 감성 비교")
print("=" * 60)

if 'platform' in df.columns:
    # 플랫폼별 감성 통계
    platform_sentiment = df.groupby('platform')['sentiment_compound'].agg(['mean', 'median', 'std'])
    print("\n플랫폼별 감성 점수:")
    print(platform_sentiment.round(3))

    # 크로스탭
    sentiment_crosstab = pd.crosstab(df['platform'], df['sentiment'], normalize='index') * 100
    print("\n플랫폼별 감성 비율 (%):")
    print(sentiment_crosstab.round(1))

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 스택 바 차트
    sentiment_crosstab.plot(kind='bar', stacked=True, ax=axes[0],
                           color=['#F44336', '#FFC107', '#4CAF50'])
    axes[0].set_title('Sentiment Distribution by Platform (%)',
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Percentage')
    axes[0].set_xlabel('Platform')
    axes[0].legend(title='Sentiment')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    # 박스플롯
    sns.boxplot(x='platform', y='sentiment_compound', data=df, ax=axes[1])
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_title('Sentiment Score by Platform', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Sentiment Score')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('session4_platform_sentiment.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================
# Part 6: 긍정/부정 워드클라우드
# ============================================

print("\n" + "=" * 60)
print("Part 6: 긍정/부정 워드클라우드")
print("=" * 60)

# 긍정/부정 댓글 분리
positive_comments = df[df['sentiment'] == 'Positive']['cleaned_text']
negative_comments = df[df['sentiment'] == 'Negative']['cleaned_text']

print(f"\n긍정 댓글: {len(positive_comments)}개")
print(f"부정 댓글: {len(negative_comments)}개")

# 긍정 워드클라우드
if len(positive_comments) > 0:
    pos_text = ' '.join(positive_comments)
    wc_positive = WordCloud(width=800, height=600,
                           background_color='white',
                           colormap='Greens',
                           max_words=80).generate(pos_text)

# 부정 워드클라우드
if len(negative_comments) > 0:
    neg_text = ' '.join(negative_comments)
    wc_negative = WordCloud(width=800, height=600,
                           background_color='white',
                           colormap='Reds',
                           max_words=80).generate(neg_text)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(wc_positive, interpolation='bilinear')
axes[0].set_title('Positive Comments Keywords',
                 fontsize=16, fontweight='bold', color='green')
axes[0].axis('off')

axes[1].imshow(wc_negative, interpolation='bilinear')
axes[1].set_title('Negative Comments Keywords',
                 fontsize=16, fontweight='bold', color='red')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('session4_sentiment_wordclouds.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 감성별 워드클라우드 완성")

# ============================================
# Part 7: 종합 대시보드
# ============================================

print("\n" + "=" * 60)
print("Part 7: 종합 분석 대시보드")
print("=" * 60)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 전체 워드클라우드
ax1 = fig.add_subplot(gs[0, :2])
ax1.imshow(wordcloud, interpolation='bilinear')
ax1.set_title('1. Most Common Words', fontsize=13, fontweight='bold')
ax1.axis('off')

# 2. 감성 분포
ax2 = fig.add_subplot(gs[0, 2])
sentiment_counts = df['sentiment'].value_counts()
colors = [{'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}[s]
          for s in sentiment_counts.index]
ax2.pie(sentiment_counts.values, labels=sentiment_counts.index,
       autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('2. Sentiment Distribution', fontsize=13, fontweight='bold')

# 3. 플랫폼별 댓글 수
if 'platform' in df.columns:
    ax3 = fig.add_subplot(gs[1, :])
    platform_counts = df['platform'].value_counts()
    ax3.barh(platform_counts.index, platform_counts.values,
            color='skyblue', edgecolor='black')
    ax3.set_xlabel('Number of Comments')
    ax3.set_title('3. Comments by Platform', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

# 4. 감성 점수 분포
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(df['sentiment_compound'], bins=30, color='coral', alpha=0.7, edgecolor='black')
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Sentiment Score')
ax4.set_ylabel('Frequency')
ax4.set_title('4. Score Distribution', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Top 키워드
ax5 = fig.add_subplot(gs[2, 1])
top_words = word_freq.most_common(10)
words, counts = zip(*top_words)
ax5.barh(words, counts, color='lightgreen', edgecolor='black')
ax5.set_xlabel('Frequency')
ax5.set_title('5. Top 10 Keywords', fontsize=13, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

# 6. 인사이트
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
insights = f"""
📊 분석 요약

총 댓글: {len(df)}개

감성 분포:
• 긍정: {(df['sentiment']=='Positive').sum()}
• 중립: {(df['sentiment']=='Neutral').sum()}
• 부정: {(df['sentiment']=='Negative').sum()}

평균 감성 점수:
{df['sentiment_compound'].mean():.3f}

가장 빈번한 단어:
"{word_freq.most_common(1)[0][0]}"
"""
ax6.text(0.1, 0.9, insights, fontsize=10, verticalalignment='top',
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Social Media Comments Analysis Dashboard',
             fontsize=18, fontweight='bold', y=0.995)
plt.savefig('session4_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 종합 대시보드 완성!")

# ============================================
# 마무리
# ============================================

print("\n" + "=" * 60)
print("🎉 Session 4 완료!")
print("=" * 60)
print("""
오늘 배운 것:
✅ 텍스트 전처리 (토큰화, 불용어 제거, 표제어 추출)
✅ 워드클라우드 생성 및 커스터마이징
✅ VADER 감성 분석
✅ 플랫폼별 감성 비교
✅ 긍정/부정 키워드 분석

다음 시간:
📋 실험 데이터 처리
📊 종합 분석 파이프라인
📝 자동화된 보고서 생성
🎯 최종 프로젝트
""")
