# Session 4: 텍스트 분석 & 워드클라우드

**수업 시간:** 2시간
**목표:** 소셜 미디어 텍스트 데이터를 분석하고 인사이트 도출하기

---

## 📋 수업 목차

1. **텍스트 분석 입문** (15분)
2. **텍스트 전처리** (30분)
3. **워드클라우드 생성** (30분)
4. **감성 분석 (Sentiment Analysis)** (30분)
5. **실전: 소셜 미디어 데이터 분석** (15분)

---

## 1. 텍스트 분석 입문 (15분)

### 왜 텍스트 분석인가?

**커뮤니케이션 연구의 핵심 데이터:**
- 소셜 미디어 댓글/포스트
- 뉴스 기사
- 고객 리뷰
- 설문 응답
- 인터뷰 내용

### 텍스트 분석으로 할 수 있는 것

1. **빈도 분석**: 어떤 단어가 자주 나오는가?
2. **감성 분석**: 긍정적? 부정적? 중립적?
3. **토픽 모델링**: 주요 주제는?
4. **트렌드 분석**: 시간에 따른 변화
5. **비교 분석**: 그룹 간 차이

---

## 2. 텍스트 전처리 (30분)

### 전처리가 필요한 이유

원시 텍스트 → 분석 가능한 형태로 변환

**예시:**
```
원본: "I LOVE this product!!! 😍😍😍 #BestEver"
전처리 후: ["love", "product", "best"]
```

### 주요 전처리 단계

#### 1. 소문자 변환
```python
text = "Hello World!"
text_lower = text.lower()  # "hello world!"
```

#### 2. 토큰화 (Tokenization)
```python
from nltk.tokenize import word_tokenize

text = "This is a sentence."
tokens = word_tokenize(text)
# ['This', 'is', 'a', 'sentence', '.']
```

#### 3. 불용어 제거 (Stopwords Removal)
```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
filtered = [word for word in tokens if word.lower() not in stop_words]
# ['sentence', '.']
```

**한글 불용어:**
```python
korean_stopwords = ['은', '는', '이', '가', '을', '를', '의', '에', '에서', '와', '과']
```

#### 4. 특수문자 제거
```python
import re

text = "Hello, World! @#$%"
cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
# "Hello World"
```

#### 5. 어간 추출 (Stemming) / 표제어 추출 (Lemmatization)
```python
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Stemming
stemmer.stem("running")  # "run"
stemmer.stem("ran")      # "ran"

# Lemmatization (더 정확)
lemmatizer.lemmatize("running", pos='v')  # "run"
lemmatizer.lemmatize("ran", pos='v')      # "run"
```

### 종합 전처리 함수

```python
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # 소문자 변환
    text = text.lower()

    # 특수문자 제거
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # 토큰화
    tokens = word_tokenize(text)

    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    # 표제어 추출
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens
```

---

## 3. 워드클라우드 생성 (30분)

### 워드클라우드란?

텍스트에서 자주 등장하는 단어를 크기로 시각화

### 기본 워드클라우드

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 텍스트 데이터
text = "data analysis python python data visualization statistics"

# 워드클라우드 생성
wordcloud = WordCloud(width=800, height=400,
                      background_color='white').generate(text)

# 시각화
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 커스터마이징

```python
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    colormap='viridis',      # 색상 팔레트
    max_words=100,            # 최대 단어 수
    relative_scaling=0.5,     # 단어 크기 상대적 조정
    min_font_size=10,         # 최소 글자 크기
    stopwords=stop_words,     # 불용어
    contour_width=2,          # 윤곽선
    contour_color='steelblue'
).generate(text)
```

### 빈도수 기반 워드클라우드

```python
from collections import Counter

# 단어 빈도 계산
words = text.split()
word_freq = Counter(words)

# 빈도수로 워드클라우드
wordcloud = WordCloud(width=800, height=400,
                      background_color='white').generate_from_frequencies(word_freq)
```

### 모양 마스크 적용

```python
from PIL import Image
import numpy as np

# 마스크 이미지 (예: 하트 모양)
mask = np.array(Image.open('heart_mask.png'))

wordcloud = WordCloud(mask=mask, background_color='white',
                      contour_width=1, contour_color='red').generate(text)
```

---

## 4. 감성 분석 (30분)

### 감성 분석이란?

텍스트의 감정/의견 판단: 긍정, 부정, 중립

**활용 예:**
- 제품 리뷰 분석
- 브랜드 평판 모니터링
- 소셜 미디어 반응 추적
- 여론 조사

### VADER (Valence Aware Dictionary and sEntiment Reasoner)

소셜 미디어 텍스트에 최적화된 감성 분석 도구

```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# 단일 텍스트 분석
text = "I absolutely love this product! It's amazing!"
scores = sia.polarity_scores(text)
print(scores)
# {'neg': 0.0, 'neu': 0.406, 'pos': 0.594, 'compound': 0.875}
```

**점수 해석:**
- `compound`: -1 (매우 부정) ~ +1 (매우 긍정)
  - ≥ 0.05: 긍정
  - ≤ -0.05: 부정
  - 그 외: 중립

### 대량 텍스트 분석

```python
import pandas as pd

# 데이터프레임에 감성 점수 추가
df['sentiment'] = df['comment'].apply(lambda x: sia.polarity_scores(x)['compound'])

# 범주화
def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)

# 분포 확인
print(df['sentiment_category'].value_counts())
```

### 시각화

```python
import seaborn as sns

# 감성 분포
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment_category', data=df, palette='Set2')
plt.title('Sentiment Distribution')
plt.show()

# 감성 점수 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(df['sentiment'], bins=50, color='skyblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Neutral')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Sentiment Scores')
plt.legend()
plt.show()
```

---

## 5. 실전: 소셜 미디어 데이터 분석 (15분)

### 종합 분석 예제

```python
import pandas as pd
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv('../datasets/social_media_comments.csv')

# 1. 텍스트 전처리
df['cleaned_text'] = df['comment'].apply(preprocess_text)

# 2. 워드클라우드
all_words = ' '.join([' '.join(words) for words in df['cleaned_text']])
wordcloud = WordCloud(width=1200, height=600,
                      background_color='white').generate(all_words)

plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Comments', fontsize=18, fontweight='bold')
plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 감성 분석
sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['comment'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment'] = df['sentiment_score'].apply(categorize_sentiment)

# 4. 플랫폼별 감성 비교
sentiment_by_platform = pd.crosstab(df['platform'], df['sentiment'], normalize='index') * 100

plt.figure(figsize=(12, 6))
sentiment_by_platform.plot(kind='bar', stacked=True, colormap='RdYlGn')
plt.title('Sentiment Distribution by Platform')
plt.ylabel('Percentage')
plt.legend(title='Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. 긍정/부정 키워드 분석
positive = df[df['sentiment'] == 'Positive']
negative = df[df['sentiment'] == 'Negative']

# 긍정 워드클라우드
pos_words = ' '.join([' '.join(words) for words in positive['cleaned_text']])
wc_positive = WordCloud(background_color='white', colormap='Greens').generate(pos_words)

# 부정 워드클라우드
neg_words = ' '.join([' '.join(words) for words in negative['cleaned_text']])
wc_negative = WordCloud(background_color='white', colormap='Reds').generate(neg_words)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(wc_positive)
axes[0].set_title('Positive Comments', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(wc_negative)
axes[1].set_title('Negative Comments', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('sentiment_wordclouds.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 💡 핵심 요약

### 텍스트 분석 워크플로우

1. **데이터 수집** → 소셜 미디어, 리뷰, 설문 등
2. **전처리** → 소문자, 토큰화, 불용어 제거
3. **분석** → 빈도, 워드클라우드, 감성
4. **시각화** → 워드클라우드, 차트
5. **해석** → 인사이트 도출

### NLTK 다운로드 (최초 1회)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

### 한글 텍스트 분석

```python
# KoNLPy 사용 (한글 형태소 분석)
from konlpy.tag import Okt

okt = Okt()
text = "데이터 분석은 정말 재미있습니다"
nouns = okt.nouns(text)  # 명사 추출
# ['데이터', '분석', '재미']
```

---

## 🤖 AI 협업 가이드

```
1. "이 댓글 데이터에서 가장 자주 언급되는 불만사항을 찾아줘"

2. "워드클라우드를 특정 모양(예: 로고)으로 만들고 싶어. 어떻게 해?"

3. "VADER가 이 문장을 잘못 분류한 것 같아. 다른 방법이 있을까?"

4. "긍정 댓글과 부정 댓글의 키워드 차이를 비교하는 코드 짜줘"

5. "감성 분석 결과를 시간대별로 추적하는 트렌드 차트 만들기"
```

---

## 📝 다음 수업 예고

**Session 5: 실험 데이터 처리 & 보고서 작성**
- 실험 데이터 클리닝
- 종합 분석 파이프라인
- 자동화된 보고서 생성
- 최종 프로젝트

---

**텍스트에 숨겨진 인사이트를 발견하세요! 💬**
