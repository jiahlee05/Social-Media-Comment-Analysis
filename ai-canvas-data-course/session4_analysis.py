import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import os
# 한글 폰트 설정 (Windows의 경우 Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# NLTK 데이터 다운로드 (VADER)
nltk.download('vader_lexicon', quiet=True)

# 1. 데이터 로드
file_path = os.path.join('datasets', 'social_media_comments.csv')
df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. 감성 분석 (VADER)
sia = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df['comment'].apply(lambda x: sia.polarity_scores(str(x)))
df['compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])

def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['compound'].apply(categorize_sentiment)

# 시각화 설정
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 (Windows)

# 3. 감성 분포 (파이 차트)
plt.figure(figsize=(8, 8))
sentiment_counts = df['sentiment'].value_counts()
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#99ff99','#ff9999'])
plt.title('Overall Sentiment Distribution')
plt.savefig('sentiment_pie_chart.png')
plt.close()

# 4. 플랫폼별 감성 비교 (스택 바 차트)
platform_sentiment = pd.crosstab(df['platform'], df['sentiment'], normalize='index') * 100
platform_sentiment.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Sentiment by Platform')
plt.ylabel('Percentage (%)')
plt.legend(title='Sentiment', loc='upper right')
plt.tight_layout()
plt.savefig('platform_sentiment_bar.png')
plt.close()

# 5. 시간에 따른 감성 변화 (선 그래프)
df_time = df.set_index('timestamp').resample('D')['compound'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_time, x='timestamp', y='compound', marker='o')
plt.title('Sentiment Trend Over Time')
plt.ylabel('Average Sentiment Score')
plt.savefig('sentiment_trend.png')
plt.close()

# 6. 워드클라우드 생성
def generate_wordcloud(text, title, filename, color_map, extra_stopwords=None):
    stopwords = set(STOPWORDS)
    if extra_stopwords:
        stopwords.update(extra_stopwords)
    
    wc = WordCloud(
        width=1200, height=600,
        background_color='white',
        stopwords=stopwords,
        colormap=color_map
    ).generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.savefig(filename)
    plt.close()
    
    # 단어 빈도 추출
    words = wc.process_text(text)
    sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:10]
    return sorted_words

# 전체 댓글
all_text = " ".join(df['comment'].astype(str))
top_words_all = generate_wordcloud(all_text, 'Total WordCloud', 'wordcloud_all.png', 'Blues')

# 긍정 댓글
pos_text = " ".join(df[df['sentiment'] == 'Positive']['comment'].astype(str))
top_words_pos = generate_wordcloud(pos_text, 'Positive WordCloud', 'wordcloud_pos.png', 'Greens')

# 부정 댓글
neg_text = " ".join(df[df['sentiment'] == 'Negative']['comment'].astype(str))
top_words_neg = generate_wordcloud(neg_text, 'Negative WordCloud', 'wordcloud_neg.png', 'Reds')

# 7. 토픽 모델링 (LDA)
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
data_vectorized = vectorizer.fit_transform(df['comment'].astype(str))

lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
lda_model.fit(data_vectorized)

feature_names = vectorizer.get_feature_names_out()
topics = []
for topic_idx, topic in enumerate(lda_model.components_):
    top_features = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    topics.append(f"Topic {topic_idx+1}: " + ", ".join(top_features))

# 8. 결과 요약 리포트
print("="*50)
print("Session 4: Social Media Content Analysis Report")
print("="*50)

print("\n[1. Sentiment Statistics]")
print(sentiment_counts)
print("\nTop 5 Positive Comments:")
print(df[df['sentiment'] == 'Positive'].sort_values(by='compound', ascending=False)['comment'].head(5).to_list())
print("\nTop 5 Negative Comments:")
print(df[df['sentiment'] == 'Negative'].sort_values(by='compound', ascending=True)['comment'].head(5).to_list())

print("\n" + "="*50)
print("[2. Key Keywords]")
print("\nTotal Top 10 Words:")
for word, freq in top_words_all: print(f"{word}: {freq}")
print("\nPositive Top 10 Words:")
for word, freq in top_words_pos: print(f"{word}: {freq}")
print("\nNegative Top 10 Words:")
for word, freq in top_words_neg: print(f"{word}: {freq}")

print("\n" + "="*50)
print("[3. Topic Modeling (LDA)]")
for t in topics:
    print(t)

print("\n" + "="*50)
print("[4. Summary & Insights]")
print("1. Sentiment: Most comments are {}.".format(sentiment_counts.idxmax()))
print("2. Platform: Sentiment varies across platforms, check 'platform_sentiment_bar.png'.")
print("3. Improvement: Negative keywords like {} suggest areas of concern.".format(top_words_neg[0][0] if top_words_neg else "N/A"))
print("="*50)
