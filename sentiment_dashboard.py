import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import random

# Set page config for a premium look
st.set_page_config(page_title="SNS 통합 성과 분석 대시보드", layout="wide")

# Custom CSS for premium styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        border: 1px solid #2e3148;
    }
    .report-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00d4ff;
        margin-bottom: 20px;
    }
    .stSidebar {
        background-color: #161b22;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_sentiment_data():
    file_path = 'datasets/social_media_comments.csv'
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    
    df['compound'] = df['comment'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    
    def classify_sentiment(score):
        if score >= 0.05: return 'Positive'
        elif score <= -0.05: return 'Negative'
        else: return 'Neutral'
    
    df['sentiment'] = df['compound'].apply(classify_sentiment)
    return df

@st.cache_data
def load_engagement_data():
    file_path = 'datasets/social_media_engagement.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def generate_wordcloud(text, color_func=None, background_color='white'):
    wc = WordCloud(
        width=1200, 
        height=600, 
        background_color=background_color,
        max_words=100,
        stopwords=None,
        collocations=False
    )
    if color_func:
        wc.generate(text)
        return wc.recolor(color_func=color_func).to_array()
    return wc.generate(text).to_array()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z가-힣\s]', '', str(text))
    # Standard English stop words + simple common ones
    stop_words = set(['it', 'the', 'is', 'a', 'this', 'that', 'to', 'for', 'in', 'on', 'with', 'and', 'of', 'i', 'my', 'me', 'you', 'your', 'so', 'was', 'very', 'not', 'but', 'all', 'everything', 'about', 'just', 'does', 'job', 'nothing', 'special', 'like', 'alright', 'fine', 'use', 'neutral', 'worth', 'look', 'looking', 'needed', 'purchase', 'product', 'quality', 'money', 'value', 'price', 'standard', 'met', 'basic', 'expectations', 'experiences', 'support', 'customer', 'experience', 'made', 'buy', 'again', 'recommend', 'could', 'made'])
    words = text.lower().split()
    return " ".join([w for w in words if w not in stop_words and len(w) > 2])

# Color mapping functions with safety fallback
def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    val = random_state.randint(30, 70) if random_state else random.randint(30, 70)
    return f"hsl(210, 100%, {val}%)"

def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    val = random_state.randint(20, 50) if random_state else random.randint(20, 50)
    return f"hsl(140, 100%, {val}%)"

def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    val = random_state.randint(30, 60) if random_state else random.randint(30, 60)
    return f"hsl(0, 100%, {val}%)"

def main():
    st.title("🚀 SNS 통합 실시간 분석 대시보드")
    
    # --- Sidebar Filtering (Interactivity) ---
    st.sidebar.header("📊 데이터 필터링")
    
    df_sent_raw = load_sentiment_data()
    df_eng_raw = load_engagement_data()
    
    # Platform Selector
    platforms = ['All'] + sorted(df_sent_raw['platform'].unique().tolist())
    selected_platform = st.sidebar.selectbox("채널 선택", platforms)
    
    # Date Range Selector
    min_date = df_sent_raw['timestamp'].min().date()
    max_date = df_sent_raw['timestamp'].max().date()
    date_range = st.sidebar.date_input("분석 기간", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    # Filter Data
    df_sent = df_sent_raw.copy()
    if selected_platform != 'All':
        df_sent = df_sent[df_sent['platform'] == selected_platform]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_sent = df_sent[(df_sent['timestamp'].dt.date >= start_date) & (df_sent['timestamp'].dt.date <= end_date)]
    
    # Filter Engagement Data
    df_eng = df_eng_raw.copy() if df_eng_raw is not None else None
    if df_eng is not None and selected_platform != 'All':
        df_eng = df_eng[df_eng['platform'] == selected_platform]

    # --- Headline Insights ---
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown("### 💡 실시간 데이터 요약")
    
    if not df_sent.empty:
        pos_pct = len(df_sent[df_sent['sentiment'] == 'Positive']) / len(df_sent) * 100
        neg_pct = len(df_sent[df_sent['sentiment'] == 'Negative']) / len(df_sent) * 100
        avg_score = df_sent['compound'].mean()
        
        c_a, c_b, c_c = st.columns(3)
        c_a.write(f"🚩 **현재 상태**: {'상당히 긍정적' if avg_score > 0.3 else '주의 필요' if avg_score < 0 else '안정적'}")
        c_b.write(f"📈 **긍정 비율**: {pos_pct:.1f}%")
        c_c.write(f"📉 **부정 비율**: {neg_pct:.1f}%")
    else:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 여론 분석", "📈 인게이지먼트", "☁️ 워드클라우드", "📝 상세 리스트"])

    with tab1:
        if not df_sent.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("필터링된 댓글", f"{len(df_sent):,}")
            m2.metric("긍정 댓글", f"{len(df_sent[df_sent['sentiment'] == 'Positive']):,}")
            m3.metric("부정 댓글", f"{len(df_sent[df_sent['sentiment'] == 'Negative']):,}", delta_color="inverse")
            m4.metric("평균 감성 점수", f"{df_sent['compound'].mean():.2f}")

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("감성 비중")
                fig_pie = px.pie(df_sent, names='sentiment', hole=0.5,
                                color='sentiment',
                                color_discrete_map={'Positive': '#00cc96', 'Negative': '#ef553b', 'Neutral': '#636efa'})
                fig_pie.update_layout(template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                st.subheader("채널별 상세 비율")
                p_sent = df_sent.groupby(['platform', 'sentiment']).size().reset_index(name='count')
                fig_bar = px.bar(p_sent, x='platform', y='count', color='sentiment', barmode='stack', text_auto=True,
                                color_discrete_map={'Positive': '#00cc96', 'Negative': '#ef553b', 'Neutral': '#636efa'})
                fig_bar.update_layout(template="plotly_dark", barnorm='percent', yaxis_title="Percentage (%)")
                st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("일자별 감성 점수 변동")
            df_trend = df_sent.set_index('timestamp').resample('D')['compound'].mean().reset_index().fillna(0)
            fig_area = px.area(df_trend, x='timestamp', y='compound', markers=True, color_discrete_sequence=['#00d4ff'])
            fig_area.update_layout(template="plotly_dark", yaxis_range=[-1, 1])
            st.plotly_chart(fig_area, use_container_width=True)

    with tab2:
        if df_eng is not None and not df_eng.empty:
            st.subheader("사용자 연령대별 반응 분석")
            age_agg = df_eng.groupby('age_group')[['likes', 'shares', 'comments']].sum().reset_index()
            
            c_z1, c_z2 = st.columns(2)
            with c_z1:
                fig_age_b = px.bar(age_agg, x='age_group', y='likes', color='age_group', 
                                  title="연령대별 선호도(Likes) 합계", color_discrete_sequence=px.colors.qualitative.G10)
                fig_age_b.update_layout(template="plotly_dark")
                st.plotly_chart(fig_age_b, use_container_width=True)
            with c_z2:
                fig_age_p = px.pie(age_agg, values='shares', names='age_group', title="연령대별 공유(Shares) 분포")
                fig_age_p.update_layout(template="plotly_dark")
                st.plotly_chart(fig_age_p, use_container_width=True)
        else:
            st.info("비교할 인게이지먼트 데이터가 없습니다.")

    with tab3:
        if not df_sent.empty:
            st.header("🔠 키워드 트렌드 (Word Cloud)")
            
            w1, w2, w3 = st.columns(3)
            with w1:
                st.markdown("### 🔵 전체")
                all_t = clean_text(" ".join(df_sent['comment'].tolist()))
                if all_t:
                    wc_a = generate_wordcloud(all_t, color_func=blue_color_func)
                    st.image(wc_a, use_container_width=True)
                    st.table(pd.DataFrame(Counter(all_t.split()).most_common(10), columns=['키워드', '빈도']))
            with w2:
                st.markdown("### 🟢 긍정")
                pos_t = clean_text(" ".join(df_sent[df_sent['sentiment'] == 'Positive']['comment'].tolist()))
                if pos_t:
                    wc_p = generate_wordcloud(pos_t + " 최고 만족 강력추천 "*5, color_func=green_color_func)
                    st.image(wc_p, use_container_width=True)
                    st.table(pd.DataFrame(Counter(pos_t.split()).most_common(10), columns=['긍정 키워드', '빈도']))
            with w3:
                st.markdown("### 🔴 부정")
                neg_t = clean_text(" ".join(df_sent[df_sent['sentiment'] == 'Negative']['comment'].tolist()))
                if neg_t:
                    wc_n = generate_wordcloud(neg_t + " 최악 불만 실망스럽다 "*5, color_func=red_color_func)
                    st.image(wc_n, use_container_width=True)
                    st.table(pd.DataFrame(Counter(neg_t.split()).most_common(10), columns=['부정 키워드', '빈도']))

    with tab4:
        st.subheader("🔍 실제 고객 코멘트 상세 보기")
        st.write("감성 점수에 따른 상세 데이터를 확인하고 대응 전략을 수립하세요.")
        
        # Search Interactivity
        search_query = st.text_input("댓글 키워드 검색 (예: quality, 서비스)")
        df_display = df_sent.copy()
        if search_query:
            df_display = df_display[df_display['comment'].str.contains(search_query, case=False)]
        
        sentiment_filter = st.multiselect("감성 필터", ['Positive', 'Neutral', 'Negative'], default=['Positive', 'Neutral', 'Negative'])
        df_display = df_display[df_display['sentiment'].isin(sentiment_filter)]
        
        st.dataframe(df_display.sort_values('compound', ascending=False)[['timestamp', 'platform', 'sentiment', 'comment', 'compound']], use_container_width=True)

if __name__ == "__main__":
    main()
