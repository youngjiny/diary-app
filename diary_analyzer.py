# diary_analyzer.py (v5.1 - 공식 GSheets 연동 최종본)

import streamlit as st
# from streamlit_gsheets import GSheetsConnection  <- ⭐️ 이 줄을 완전히 삭제합니다.
import re
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import joblib
import random

# --- 1. 기본 설정 ---
MODEL_PATH = Path("sentiment_model.pkl")
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")

# ... (폰트 설정, EMOTIONS, TIMES 등 이전과 동일)
try:
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
except FileNotFoundError:
    st.warning("Malgun Gothic 폰트를 찾을 수 없어 그래프의 한글이 깨질 수 있습니다.")

EMOTIONS = ["기쁨", "슬픔", "분노", "우울", "사랑"]
TIMES = ["아침", "점심", "저녁"]
TIME_KEYWORDS = { "아침": ["아침", "오전", "출근", "일어나서"], "점심": ["점심", "낮", "점심시간"], "저녁": ["저녁", "오후", "퇴근", "밤", "새벽", "자기 전", "꿈"],}


# --- 2. 핵심 기능 함수 ---
@st.cache_resource
def load_ml_resources():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError: return None, None

model, vectorizer = load_ml_resources()

# ⭐️ Google Sheets 연결 설정 (더 간단하고 올바른 방식)
# Secrets의 [connections.gsheets] 제목과 일치해야 합니다.
conn = st.connection("gsheets", type="streamlit_gsheets.GSheetsConnection")

def save_feedback_to_gsheets(feedback_df):
    """피드백을 Google Sheets에 바로 추가합니다."""
    try:
        existing_data = conn.read(worksheet="Sheet1") # ⭐️ 본인의 시트 이름 확인! (보통 Sheet1)
        feedback_to_save = feedback_df[['text', 'label']]
        updated_df = pd.concat([existing_data, feedback_to_save], ignore_index=True)
        updated_df.drop_duplicates(subset=['text'], keep='last', inplace=True)
        conn.update(worksheet="Sheet1", data=updated_df) # ⭐️ 본인의 시트 이름 확인!
        st.success("소중한 피드백이 Google Sheets에 안전하게 저장되었습니다!")
    except Exception as e:
        st.error(f"피드백 저장 중 오류 발생: {e}")
        st.error("Google Sheets 공유 설정 및 시트 이름을 확인해주세요.")

# ... (analyze_diary_ml, recommend, generate_random_diary 등 나머지 함수는 이전과 동일)
# ... (UI 구성 및 나머지 함수는 이전과 동일)
def analyze_diary_ml(text):
    if not model or not vectorizer: return None, None
    sentences = re.split(r'[.?!]', text); sentences = [s.strip() for s in sentences if s.strip()]
    time_scores = { t: {e: 0 for e in EMOTIONS} for t in TIMES }
    analysis_results = []
    for sentence in sentences:
        current_time = "저녁"
        for time_key, keywords in TIME_KEYWORDS.items():
            if any(keyword in sentence for keyword in keywords): current_time = time_key; break
        text_vector = vectorizer.transform([sentence])
        prediction = model.predict(text_vector)[0]
        if prediction in time_scores[current_time]: time_scores[current_time][prediction] += 1
        analysis_results.append({'sentence': sentence, 'predicted_emotion': prediction, 'predicted_time': current_time})
    return time_scores, analysis_results
def recommend(final_emotion):
    recommendations = {"기쁨": {"책": ["오늘 밤, 세계에서 이 사랑이 사라진다 해도"], "음악": ["윤하 - 사건의 지평선"], "영화": ["탑건: 매버릭"]},"슬픔": {"책": ["달러구트 꿈 백화점"], "음악": ["김광석 - 서른 즈음에"], "영화": ["코코"]},"분노": {"책": ["역행자"], "음악": ["(여자)아이들 - TOMBOY"], "영화": ["범죄도시2"]},"우울": {"책": ["불편한 편의점"], "음악": ["아이유 - 밤편지"], "영화": ["리틀 포레스트"]},"사랑": {"책": ["나의 해방일지"], "음악": ["성시경 - 너의 모든 순간"], "영화": ["헤어질 결심"]},}
    return recommendations.get(final_emotion, {"책": [], "음악": [], "영화": []})
def generate_random_diary():
    morning = ["아침에 상쾌하게 일어났다.", "출근길 지하철에 사람이 너무 많아 힘들었다.", "일어나자마자 마신 커피가 정말 맛있었다."]
    afternoon = ["점심으로 맛있는 파스타를 먹어서 기분이 좋았다.", "오후 회의가 길어져서 너무 지쳤다.", "동료와 잠시 나눈 수다가 즐거웠다."]
    evening = ["퇴근하고 운동을 하니 개운했다.", "저녁 약속이 갑자기 취소되어 서운했다.", "자기 전에 본 영화가 정말 감동적이었다.", "하루 종일 쌓인 스트레스에 화가 났다."]
    return f"{random.choice(morning)} {random.choice(afternoon)} {random.choice(evening)}"
def handle_random_click():
    st.session_state.diary_text = generate_random_diary()
    st.session_state.analysis_results = None
def handle_analyze_click():
    diary_content = st.session_state.diary_text
    if not diary_content.strip(): st.warning("일기를 입력해주세요!")
    elif model is None or vectorizer is None: st.error("모델이 로드되지 않았습니다.")
    else:
        with st.spinner('AI가 일기를 분석하고 있습니다...'):
            _, results = analyze_diary_ml(diary_content)
            st.session_state.analysis_results = results
st.set_page_config(layout="wide")
st.title("📊 하루 감정 분석 리포트 (v5.1)")
if 'diary_text' not in st.session_state: st.session_state.diary_text = ""
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
col1, col2 = st.columns([3, 1])
with col1:
    st.text_area("오늘의 일기를 시간의 흐름에 따라 작성해보세요:", key='diary_text', height=250)
with col2:
    st.write(" "); st.write(" ")
    st.button("🔄 랜덤 일기 생성", on_click=handle_random_click)
    st.button("🔍 내 하루 감정 분석하기", type="primary", on_click=handle_analyze_click)
if st.session_state.analysis_results:
    scores_data, _ = analyze_diary_ml(st.session_state.diary_text)
    df_scores = pd.DataFrame(scores_data).T
    if df_scores.sum().sum() > 0:
        st.subheader("🕒 시간대별 감정 분석 결과")
        final_emotion = df_scores.sum().idxmax()
        res_col1, res_col2 = st.columns([1.2, 1])
        with res_col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            df_scores.plot(kind='bar', stacked=True, ax=ax, width=0.8, colormap='Pastel1', edgecolor='grey')
            ax.set_title("시간대별 감정 변화 그래프", fontsize=16); ax.set_ylabel("감정 문장 수", fontsize=12)
            ax.set_xticklabels(df_scores.index, rotation=0, fontsize=12)
            ax.legend(title="감정", bbox_to_anchor=(1.02, 1), loc='upper left'); plt.tight_layout()
            st.pyplot(fig)
        with res_col2:
            st.dataframe(df_scores.style.format("{:.0f}").background_gradient(cmap='viridis'))
            st.success(f"오늘 하루를 종합해 보면, **'{final_emotion}'**의 감정이 가장 컸네요!")
        st.divider()
        st.subheader(f"'{final_emotion}' 감정을 위한 오늘의 추천")
        recs = recommend(final_emotion)
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        with rec_col1:
            st.write("📚 **이런 책은 어때요?**")
            for item in recs['책']: st.write(f"- {item}")
        with rec_col2:
            st.write("🎵 **이런 음악도 들어보세요!**")
            for item in recs['음악']: st.write(f"- {item}")
        with rec_col3:
            st.write("🎬 **이런 영화/드라마도 추천해요!**")
            for item in recs['영화']: st.write(f"- {item}")
        st.divider()
        st.subheader("🔍 분석 결과 피드백")
        feedback_data = []
        for i, result in enumerate(st.session_state.analysis_results):
            st.markdown(f"> {result['sentence']}")
            cols = st.columns([1, 1])
            with cols[0]:
                correct_time = st.radio("이 문장의 시간대는?", TIMES, index=TIMES.index(result['predicted_time']), key=f"time_{i}", horizontal=True)
            with cols[1]:
                correct_emotion = st.selectbox("이 문장의 진짜 감정은?", EMOTIONS, index=EMOTIONS.index(result['predicted_emotion']), key=f"emotion_{i}")
            feedback_data.append({'text': result['sentence'], 'label': correct_emotion, 'time': correct_time})
            st.write("---")
        if st.button("피드백 제출하기"):
            changed_feedback = []
            for i, row in enumerate(pd.DataFrame(feedback_data).to_dict('records')):
                original = st.session_state.analysis_results[i]
                if row['label'] != original['predicted_emotion'] or row['time'] != original['predicted_time']:
                    changed_feedback.append({'text': row['text'], 'label': row['label']})
            if changed_feedback:
                final_feedback_df = pd.DataFrame(changed_feedback)
                save_feedback_to_gsheets(final_feedback_df) # 함수 이름 변경 확인
                st.session_state.analysis_results = None; st.rerun()
            else: st.info("수정된 내용이 없네요. AI가 잘 맞췄나 보네요! 😄")
with st.expander("피드백 저장 현황 보기 (Google Sheets)"):
    try:
        df = conn.read(worksheet="Sheet1", usecols=[0, 1], ttl="5m")
        st.write(f"현재 총 **{len(df)}개**의 데이터가 저장되어 있습니다.")
        st.dataframe(df.tail(5))
    except Exception as e:
        st.write("아직 저장된 데이터가 없거나, 시트를 불러오는 데 실패했습니다.")