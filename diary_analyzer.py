# diary_analyzer.py (v6.5 - 디버깅 기능 추가)

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import re
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import joblib
import random
import openai

# --- 1. 기본 설정 ---
MODEL_PATH = Path("sentiment_model.pkl")
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")

try:
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
except FileNotFoundError:
    st.warning("Malgun Gothic 폰트를 찾을 수 없어 그래프의 한글이 깨질 수 있습니다.")

EMOTIONS = ["행복", "사랑", "슬픔", "분노", "힘듦", "놀람"]
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

@st.cache_resource
def get_gsheets_connection():
    try:
        if "connections" in st.secrets and "gsheets" in st.secrets.connections:
            creds_dict = st.secrets["connections"]["gsheets"]
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
            client = gspread.authorize(credentials)
            return client
        else:
            return None
    except Exception:
        return None

def analyze_diary_ml(model, vectorizer, text):
    # ... (이전과 동일)
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
        if prediction in time_scores.get(current_time, {}):
             time_scores[current_time][prediction] += 1
        analysis_results.append({'sentence': sentence, 'predicted_emotion': prediction, 'predicted_time': current_time})
    return time_scores, analysis_results

def recommend(final_emotion):
    # ... (이전과 동일)
    recommendations = {
        "행복": {"책": ["기분을 관리하면 인생이 관리된다"], "음악": ["악뮤 - DINOSAUR"], "영화": ["월터의 상상은 현실이 된다"]},
        "사랑": {"책": ["사랑의 기술"], "음악": ["폴킴 - 모든 날, 모든 순간"], "영화": ["어바웃 타임"]},
        "슬픔": {"책": ["아몬드"], "음악": ["이선희 - 인연"], "영화": ["코코"]},
        "분노": {"책": ["분노의 심리학"], "음악": ["G-DRAGON - 삐딱하게"], "영화": ["성난 사람들 (드라마)"]},
        "힘듦": {"책": ["죽고 싶지만 떡볶이는 먹고 싶어"], "음악": ["옥상달빛 - 수고했어, 오늘도"], "영화": ["리틀 포레스트"]},
        "놀람": {"책": ["데미안"], "음악": ["Queen - Bohemian Rhapsody"], "영화": ["유전"]},
    }
    return recommendations.get(final_emotion, {"책": [], "음악": [], "영화": []})

def save_feedback_to_gsheets(client, feedback_df):
    # ... (이전과 동일)
    try:
        spreadsheet = client.open("diary_app_feedback")
        worksheet = spreadsheet.worksheet("Sheet1")
        existing_data = pd.DataFrame(worksheet.get_all_records())
        feedback_to_save = feedback_df[['text', 'label']]
        updated_df = pd.concat([existing_data, feedback_to_save], ignore_index=True)
        updated_df.drop_duplicates(subset=['text'], keep='last', inplace=True)
        worksheet.clear()
        worksheet.update([updated_df.columns.values.tolist()] + updated_df.values.tolist())
        st.success("소중한 피드백이 Google Sheets에 안전하게 저장되었습니다!")
    except Exception as e:
        st.error(f"피드백 저장 중 오류 발생: {e}")

def generate_simple_diary():
    # ... (이전과 동일)
    morning = ["아침에 상쾌하게 일어났다.", "출근길 지하철에 사람이 너무 많아 힘들었다."]
    afternoon = ["점심으로 맛있는 파스타를 먹어서 기분이 좋았다.", "갑작스러운 소식을 듣고 너무 놀랐다."]
    evening = ["퇴근하고 운동을 하니 개운했다.", "자기 전에 본 영화가 정말 감동적이고 사랑스러웠다."]
    return f"{random.choice(morning)} {random.choice(afternoon)} {random.choice(evening)}"

# ⭐️⭐️⭐️ 1. 'AI 일기 생성' 함수 수정 ⭐️⭐️⭐️
def generate_diary_with_llm():
    """생성 AI를 이용한 새로운 일기 생성 함수 (디버깅 기능 추가)"""
    # st.secrets에 키가 있는지 먼저 확인
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API 키가 Secrets에 설정되지 않았습니다! '디버깅 정보 보기'를 확인해주세요.")
        return None # 오류가 있으면 함수를 중단

    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        emotion_list = ["행복", "사랑", "슬픔", "분노", "힘듦", "놀람"]
        selected_emotions = random.sample(emotion_list, 2)
        
        prompt = (
            f"당신은 사용자의 감정을 잘 표현하는 일기 작성 전문가입니다. "
            f"'{selected_emotions[0]}'과(와) '{selected_emotions[1]}'의 감정이 자연스럽게 드러나는 "
            f"3~4 문장 길이의 일기를 한 편 작성해주세요. "
            f"답변은 다른 부가 설명 없이 오직 일기 내용만 포함해야 합니다."
        )
        
        response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        diary_content = response.choices[0].message.content
        return diary_content.strip()
    except Exception as e:
        st.error(f"AI 일기 생성 중 API 오류 발생: {e}")
        return None

def handle_random_click():
    with st.spinner("AI가 새로운 일기를 창작하고 있습니다..."):
        new_diary = generate_diary_with_llm()
        if new_diary: # 생성에 성공했을 때만 내용을 업데이트
            st.session_state.diary_text = new_diary
    st.session_state.analysis_results = None

def handle_analyze_click(model, vectorizer):
    # ... (이전과 동일)
    diary_content = st.session_state.diary_text
    if not diary_content.strip(): st.warning("일기를 입력해주세요!")
    elif model is None or vectorizer is None: st.error("모델이 로드되지 않았습니다.")
    else:
        with st.spinner('AI가 일기를 분석하고 있습니다...'):
            _, results = analyze_diary_ml(model, vectorizer, diary_content)
            st.session_state.analysis_results = results

# --- 4. Streamlit UI 구성 ---
st.set_page_config(layout="wide")
st.title("📊 하루 감정 분석 리포트 (v6.5)")
# ... (이하 UI 코드 대부분 동일)

model, vectorizer = load_ml_resources()
if 'diary_text' not in st.session_state: st.session_state.diary_text = ""
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
col1, col2 = st.columns([3, 1])
with col1:
    st.text_area("오늘의 일기를 시간의 흐름에 따라 작성해보세요:", key='diary_text', height=250)
with col2:
    st.write(" "); st.write(" ")
    st.button("🔄 AI로 일기 생성", on_click=handle_random_click)
    st.button("🔍 내 하루 감정 분석하기", type="primary", on_click=handle_analyze_click, args=(model, vectorizer))
if st.session_state.analysis_results:
    if model and vectorizer:
        scores_data, _ = analyze_diary_ml(model, vectorizer, st.session_state.diary_text)
        df_scores = pd.DataFrame(scores_data).T
        if df_scores.sum().sum() > 0:
            # ... (시각화 및 추천 UI는 이전과 동일)
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
                st.write("📚 **이런 책은 어때요?**"); [st.write(f"- {item}") for item in recs['책']]
            with rec_col2:
                st.write("🎵 **이런 음악도 들어보세요?**"); [st.write(f"- {item}") for item in recs['음악']]
            with rec_col3:
                st.write("🎬 **이런 영화/드라마도 추천해요!**"); [st.write(f"- {item}") for item in recs['영화']]
            st.divider()
            st.subheader("🔍 분석 결과 피드백")
            feedback_data = []
            for i, result in enumerate(st.session_state.analysis_results):
                st.markdown(f"> {result['sentence']}")
                cols = st.columns([1, 1])
                with cols[0]:
                    correct_time = st.radio("이 문장의 시간대는?", TIMES, index=TIMES.index(result['predicted_time']), key=f"time_{i}", horizontal=True)
                with cols[1]:
                    try: emotion_index = EMOTIONS.index(result['predicted_emotion'])
                    except ValueError: emotion_index = 0
                    correct_emotion = st.selectbox("이 문장의 진짜 감정은?", EMOTIONS, index=emotion_index, key=f"emotion_{i}")
                feedback_data.append({'text': result['sentence'], 'label': correct_emotion, 'time': correct_time})
                st.write("---")
            if st.button("피드백 제출하기"):
                client = get_gsheets_connection()
                if client:
                    changed_feedback = []
                    for i, row in enumerate(pd.DataFrame(feedback_data).to_dict('records')):
                        original = st.session_state.analysis_results[i]
                        if row['label'] != original['predicted_emotion'] or row['time'] != original['predicted_time']:
                            changed_feedback.append({'text': row['text'], 'label': row['label']})
                    if changed_feedback:
                        final_feedback_df = pd.DataFrame(changed_feedback)
                        save_feedback_to_gsheets(client, final_feedback_df)
                        st.session_state.analysis_results = None; st.rerun()
                    else: st.info("수정된 내용이 없네요. AI가 잘 맞췄나 보네요! 😄")
                else:
                    st.error("Google Sheets에 연결할 수 없습니다.")

# ⭐️⭐️⭐️ 2. '디버깅 정보 보기' 섹션 추가 ⭐️⭐️⭐️
with st.expander("⚙️ 디버깅 정보 보기"):
    st.write("현재 Streamlit Secrets에 등록된 키 목록:")
    # st.secrets.keys()를 사용해 모든 최상위 키를 보여줍니다.
    st.write(st.secrets.keys())
    
    st.write("`connections.gsheets` 상세 정보:")
    if "connections" in st.secrets and "gsheets" in st.secrets.connections:
        st.json(st.secrets.connections.gsheets)
    else:
        st.warning("`connections.gsheets` 정보가 Secrets에 없습니다.")
        
    st.write("`OPENAI_API_KEY` 상세 정보:")
    if "OPENAI_API_KEY" in st.secrets:
        # 키의 일부만 보여줘서 유출 방지
        st.write(f"키가 등록되어 있습니다: `{st.secrets.OPENAI_API_KEY[:5]}...`")
    else:
        st.warning("`OPENAI_API_KEY`가 Secrets에 없습니다.")