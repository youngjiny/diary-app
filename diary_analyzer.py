# diary_analyzer.py (v6.0 - gspread 직접 연결 방식)

import streamlit as st
import gspread  # ⭐️ gspread 라이브러리 import
from google.oauth2.service_account import Credentials # ⭐️ 인증 관련 import
import re
import pandas as pd
# ... (이하 다른 import 구문들은 동일)
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import joblib
import random

# --- 1. 기본 설정 ---
# (이전과 동일)
MODEL_PATH = Path("sentiment_model.pkl")
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")

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
# ⭐️ Google Sheets 연결 함수 (gspread 방식으로 변경)
@st.cache_resource
def get_gsheets_connection():
    # Streamlit Secrets에서 인증 정보 불러오기
    # st.secrets["connections"]["gsheets"]는 Secrets 제목이 [connections.gsheets]임을 의미
    creds_dict = st.secrets["connections"]["gsheets"]
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(credentials)
    return client

def save_feedback_to_gsheets(feedback_df):
    """피드백을 gspread를 사용해 Google Sheets에 추가합니다."""
    try:
        client = get_gsheets_connection()
        # ⭐️ 본인의 Google Sheets 파일 이름과 워크시트 이름을 정확히 입력!
        spreadsheet = client.open("diary_app_feedback") 
        worksheet = spreadsheet.worksheet("Sheet1") # 보통 'Sheet1' 또는 '시트1'
        
        # 기존 데이터와 새 피드백 병합 및 중복 제거
        existing_data = pd.DataFrame(worksheet.get_all_records())
        feedback_to_save = feedback_df[['text', 'label']]
        updated_df = pd.concat([existing_data, feedback_to_save], ignore_index=True)
        updated_df.drop_duplicates(subset=['text'], keep='last', inplace=True)
        
        # 전체 데이터를 시트에 덮어쓰기
        worksheet.clear()
        worksheet.update([updated_df.columns.values.tolist()] + updated_df.values.tolist())
        
        st.success("소중한 피드백이 Google Sheets에 안전하게 저장되었습니다!")
    except Exception as e:
        st.error(f"피드백 저장 중 오류 발생: {e}")
        st.error("Google Sheets 공유 설정, 파일 이름, 시트 이름을 확인해주세요.")

# ... (analyze_diary_ml, recommend 등 다른 함수들은 이전과 동일)
@st.cache_resource
def load_ml_resources():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError: return None, None
model, vectorizer = load_ml_resources()
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


# --- UI 코드 ---

# ⭐️ 피드백 저장 현황 보기 (gspread 방식으로 변경)
with st.expander("피드백 저장 현황 보기 (Google Sheets)"):
    try:
        client = get_gsheets_connection()
        spreadsheet = client.open("diary_app_feedback")
        worksheet = spreadsheet.worksheet("Sheet1")
        df = pd.DataFrame(worksheet.get_all_records())
        
        st.write(f"현재 총 **{len(df)}개**의 데이터가 저장되어 있습니다.")
        st.dataframe(df.tail(5))
    except Exception as e:
        st.write("아직 저장된 데이터가 없거나, 시트를 불러오는 데 실패했습니다.")