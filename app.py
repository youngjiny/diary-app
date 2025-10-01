# app.py (N-gram 분석 + 사전 연동 + 추천 기능 최종 완성 버전)

import streamlit as st
import re
import pandas as pd
import json
from pathlib import Path
from soynlp.tokenizer import LTokenizer
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- 1. 기본 설정 및 사전 정의 ---

# 경로 설정
SENTIMENT_LEXICON_PATH = Path("SentiWord_info.json")

# 한글 폰트 설정
try:
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
except FileNotFoundError:
    st.warning("Malgun Gothic 폰트를 찾을 수 없어 그래프의 한글이 깨질 수 있습니다.")

# 감정 및 키워드 사전
EMOTIONS = ["Happiness", "Sadness", "Love", "Anger"]
TIME_KEYWORDS = {
    "아침": ["아침", "오전", "출근"],
    "점심": ["점심", "낮"],
    "저녁": ["저녁", "오후", "퇴근", "밤", "새벽", "자기 전", "꿈"],
}
MODIFIERS = { 
    "정말": 1.5, "진짜": 1.5, "너무": 1.5, "완전": 1.5, "매우": 1.5, "엄청": 1.5,
    "살짝": 0.7, "조금": 0.7, "약간": 0.7,
    "안": -1.0, "못": -1.0, "별로": -1.0,
}
SWEAR_LEXICON = {
    "존나": 2.0, "씨발": 2.0, "개빡치네": 2.0, "지랄": 2.0, "염병": 2.0,
    "새끼": 1.8, "미친": 1.8, "개돼지": 2.0, "시발": 2.0
}

# 형태소 분석기 및 감성 사전 로딩
@st.cache_resource
def load_resources():
    tokenizer = LTokenizer()
    senti_dic = {}
    if SENTIMENT_LEXICON_PATH.exists():
        with open(SENTIMENT_LEXICON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # {단어 또는 단어 묶음: 점수} 형태로 사전을 미리 가공
            senti_dic = {item['word_root']: int(item['polarity']) for item in data}
    else:
        st.error(f"감성 사전 파일({SENTIMENT_LEXICON_PATH})을 찾을 수 없습니다.")
    return tokenizer, senti_dic

tokenizer, SENTIMENT_LEXICON = load_resources()

# --- 2. 분석 및 추천 함수 정의 (N-gram 방식 적용) ---

def analyze_diary_advanced(text):
    sentences = re.split(r'[.?!]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    time_scores = { "아침": {e: 0.0 for e in EMOTIONS}, "점심": {e: 0.0 for e in EMOTIONS}, "저녁": {e: 0.0 for e in EMOTIONS} }
    
    for sentence in sentences:
        current_time = "저녁"
        for time_key, keywords in TIME_KEYWORDS.items():
            if any(keyword in sentence for keyword in keywords):
                current_time = time_key
                break
        
        tokens = tokenizer.tokenize(sentence)
        
        # 단어 묶음 (N-grams) 생성: 3개, 2개, 1개 순서로 확인
        for n in range(3, 0, -1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i : i + n])
                score = SENTIMENT_LEXICON.get(ngram, 0)
                
                if score != 0:
                    modifier_weight = 1.0
                    if i > 0 and tokens[i-1] in MODIFIERS:
                        modifier_weight = MODIFIERS[tokens[i-1]]
                    
                    # N-gram으로 찾은 단어는 중복 분석을 피하기 위해 마킹
                    for j in range(n):
                        tokens[i+j] = ""
                    
                    # 점수 적용 (Love, Anger는 별도 처리 없이 Happiness/Sadness로 통합)
                    if score > 0:
                        time_scores[current_time]["Happiness"] += (score * modifier_weight)
                    else:
                        time_scores[current_time]["Sadness"] += (abs(score) * modifier_weight)

        # 비속어 점수 추가
        for word in tokens: # N-gram으로 처리되지 않은 나머지 단어들
             if word in SWEAR_LEXICON:
                time_scores[current_time]["Anger"] += SWEAR_LEXICON[word]

    return time_scores

def recommend(final_emotion):
    BOOKS = {
        "Happiness": ["아주 작은 습관의 힘", "어떤 하루"], "Sadness": ["아몬드", "죽음에 관하여"],
        "Love": ["사랑의 기술", "연애의 문장들"], "Anger": ["감정 어휘", "멈추면, 비로소 보이는 것들"],
    }
    MUSIC = {
        "Happiness": ["Happy Pop", "Sunny Day"], "Sadness": ["Sad Piano", "Rainy Mood"],
        "Love": ["Acoustic Love Songs", "Romantic Ballads"], "Anger": ["Calm Instrumental", "Peaceful Piano"],
    }
    MOVIES = {
        "Happiness": ["리틀 포레스트", "라라랜드"], "Sadness": ["맨체스터 바이 더 씨", "문라이트"],
        "Love": ["비긴 어게인", "비포 선라이즈"], "Anger": ["위플래쉬", "조커"],
    }
    if final_emotion not in EMOTIONS: final_emotion = "Happiness"
    return BOOKS.get(final_emotion, []), MUSIC.get(final_emotion, []), MOVIES.get(final_emotion, [])

# --- 3. Streamlit UI 구성 ---
st.set_page_config(layout="wide")
st.title("📊 하루 감정 분석 리포트")
diary = st.text_area("오늘의 일기를 시간의 흐름에 따라 작성해보세요:", 
                     value="아침에 출근하는데 차가 존나 막혀서 씨발 개빡쳤다. 그래도 점심에 맛있는 파스타를 먹어서 기분이 좀 풀렸다. 저녁에는 친구와 즐거운 시간을 보냈고, 자기 전에 좋은 꿈을 꿨다.",
                     height=200)

if st.button("내 하루 감정 분석하기"):
    if not diary.strip():
        st.warning("일기를 입력해주세요!")
    else:
        with st.spinner('일기를 분석하고 있습니다...'):
            scores = analyze_diary_advanced(diary)
            df_scores = pd.DataFrame(scores).T
            st.subheader("🕒 시간대별 감정 분석 결과")
            if df_scores.sum().sum() == 0:
                st.write("일기에서 감정 표현을 찾지 못했어요. 😢")
            else:
                final_emotion = df_scores.sum().idxmax()
                col1, col2 = st.columns([1, 1.2])
                with col1:
                    st.dataframe(df_scores.style.format("{:.2f}").background_gradient(cmap='viridis', axis=1))
                    st.success(f"오늘 하루의 최종 감정은 **'{final_emotion}'** 입니다!")
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    df_scores.plot(kind='bar', ax=ax, width=0.8)
                    ax.set_title("시간대별 감정 변화 그래프")
                    ax.set_ylabel("감정 점수")
                    ax.set_xticklabels(df_scores.index, rotation=0)
                    ax.legend(title="감정")
                    plt.tight_layout()
                    st.pyplot(fig)
                st.divider()
                st.subheader(f"'{final_emotion}' 감정에 어울리는 콘텐츠 추천")
                books, music, movies = recommend(final_emotion)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write("📚 **책**")
                    for i in books: st.write(f"- {i}")
                with c2:
                    st.write("🎵 **음악**")
                    for i in music: st.write(f"- {i}")
                with c3:
                    st.write("🎬 **영화**")
                    for i in movies: st.write(f"- {i}")