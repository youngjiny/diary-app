# app.py (비속어 가중치 포함 최종 버전)

import streamlit as st
import re
import pandas as pd
from mecab import MeCab
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- 1. 기본 설정 및 사전 정의 ---

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
    "저녁": ["저녁", "오후", "퇴근", "밤", "새벽", "자기 전"],
}
EMOTION_LEXICON = {
    "Happiness": {"행복", "기쁘", "즐겁", "설레", "유쾌", "환희", "흥겹", "신나"},
    "Sadness": {"슬프", "우울", "허전", "침울", "눈물", "속상", "서운"},
    "Love": {"사랑", "애정", "좋아", "다정", "연모"},
    "Anger": {"화나", "분노", "짜증", "억울", "성나", "격분"},
}
# 부사 가중치 사전
MODIFIERS = { 
    "정말": 1.5, "진짜": 1.5, "너무": 1.5, "완전": 1.5, "매우": 1.5, "엄청": 1.5,
    "살짝": 0.7, "조금": 0.7, "약간": 0.7,
    "안": -1.0, "못": -1.0, "별로": -1.0,
}
# ★★★ 비속어 및 격한 표현 사전 추가 ★★★
# 이 단어들은 '분노' 감정에 매우 높은 가중치(2.0)를 부여합니다.
SWEAR_LEXICON = {
    "존나": 2.0, "씨발": 2.0, "개빡치네": 2.0, "지랄": 2.0, "염병": 2.0,
    "새끼": 1.8, "미친": 1.8, "개돼지": 2.0, "시발": 2.0
}


# 형태소 분석기 준비
mecab = MeCab()

# --- 2. 분석 함수 정의 (비속어 분석 로직 추가) ---
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
        
        morphs = mecab.morphs(sentence)
        for i, word in enumerate(morphs):
            # 1. 비속어/격한 표현 먼저 확인 (가장 높은 가중치)
            if word in SWEAR_LEXICON:
                time_scores[current_time]["Anger"] += SWEAR_LEXICON[word]
                continue # 비속어는 다른 감정과 중복 계산하지 않음

            # 2. 일반 감성 단어 확인
            for emotion, lexicon in EMOTION_LEXICON.items():
                if word in lexicon:
                    modifier_weight = 1.0
                    if i > 0 and morphs[i-1] in MODIFIERS:
                        modifier_weight = MODIFIERS[morphs[i-1]]
                    
                    time_scores[current_time][emotion] += (1.0 * modifier_weight)
    
    return time_scores

# --- 3. Streamlit UI 구성 ---
st.set_page_config(layout="wide")
st.title("📊 하루 감정 분석 리포트")

# 예시 문장에 비속어 추가
diary = st.text_area("오늘의 일기를 시간의 흐름에 따라 작성해보세요:", 
                     value="아침에 출근하는데 차가 존나 막혀서 씨발 개빡쳤다. 그래도 점심에 맛있는 파스타를 먹어서 기분은 좀 풀렸다. 저녁에는 친구와 즐거운 시간을 보냈다.",
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
                st.write("일기에서 감정 표현을 찾지 못했어요. 조금 더 자세히 써주시면 분석해 드릴게요! 😊")
            else:
                col1, col2 = st.columns([1, 1.2])

                with col1:
                    st.dataframe(df_scores.style.format("{:.2f}").background_gradient(cmap='viridis', axis=1))
                    st.caption("각 시간대별 감정 점수입니다. 점수가 높을수록 해당 감정이 강하게 나타났습니다.")

                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    df_scores.plot(kind='bar', ax=ax, width=0.8)
                    ax.set_title("시간대별 감정 변화 그래프")
                    ax.set_ylabel("감정 점수")
                    ax.set_xticklabels(df_scores.index, rotation=0)
                    ax.legend(title="감정")
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.caption("시간에 따라 내 감정이 어떻게 변했는지 한눈에 확인해보세요.")