# diary_analyzer.py (v7.20 - Recommendations 함수 재도전)

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
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

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
        creds_dict = st.secrets.get("connections", {}).get("gsheets")
        if creds_dict:
            scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
            client = gspread.authorize(credentials)
            return client
        return None
    except Exception:
        return None

@st.cache_resource
def get_spotify_client():
    spotify_creds = st.secrets.get("spotify", {})
    client_id = spotify_creds.get("client_id")
    client_secret = spotify_creds.get("client_secret")
    if not client_id or not client_secret:
        return None
    try:
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        return sp
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_all_data_from_gsheets(_client):
    try:
        spreadsheet = _client.open("diary_app_feedback")
        worksheet = spreadsheet.worksheet("Sheet1")
        df = pd.DataFrame(worksheet.get_all_records())
        return df
    except Exception as e:
        st.error(f"Google Sheets 데이터 로딩 오류: {e}")
        return pd.DataFrame()

def analyze_diary_ml(model, vectorizer, text):
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

@st.cache_data(ttl=3600)
def get_spotify_playlist_recommendations(emotion):
    sp_client = get_spotify_client()
    if not sp_client: return ["Spotify 연결 실패"]
    try:
        playlist_ids = {
            "행복": "1kaEr7seXIYcPflw2M60eA", "사랑": "2KKLfSejuxil1vZvzdVgB4",
            "슬픔": "3tAeVAtMWHzaGOXMGoRhTb", "분노": "22O1tfJ7fSjIo2FdxtJU1",
            "힘듦": "68HSylU5xKtDVYiago9RDw", "놀람": "3sHzse5FGtcafd8dY0mO8h",
        }
        playlist_id = playlist_ids.get(emotion)
        if not playlist_id: return ["추천할 플레이리스트가 없어요."]
        results = sp_client.playlist_items(playlist_id, limit=50)
        tracks = [item['track'] for item in results['items'] if item['track']]
        if not tracks: return ["플레이리스트에 노래가 없어요."]
        random_tracks = random.sample(tracks, min(3, len(tracks)))
        return [f"{track['name']} - {track['artists'][0]['name']}" for track in random_tracks]
    except Exception as e:
        return [f"Spotify 추천 오류: {e}"]

# ⭐️⭐️⭐️ AI 추천 함수를 'recommendations' 방식으로 재수정 ⭐️⭐️⭐️
@st.cache_data(ttl=3600)
def get_spotify_ai_recommendations(emotion):
    sp_client = get_spotify_client()
    if not sp_client:
        return ["Spotify 연결에 실패했습니다."]
    try:
        # 감정별 '씨앗 장르'와 '음악 특징' 설정
        params = {
            "행복": {"seed_genres": ["k-pop", "dance-pop"], "target_valence": 0.8, "target_energy": 0.8},
            "사랑": {"seed_genres": ["k-pop", "r-n-b"], "target_valence": 0.7, "target_energy": 0.5},
            "슬픔": {"seed_genres": ["ballad", "k-indie"], "target_valence": 0.2, "target_energy": 0.3},
            "분노": {"seed_genres": ["rock", "hard-rock"], "target_valence": 0.3, "target_energy": 0.9},
            "힘듦": {"seed_genres": ["ambient", "acoustic"], "target_valence": 0.4, "target_energy": 0.2},
            "놀람": {"seed_genres": ["synth-pop", "funk"], "target_valence": 0.6, "target_energy": 0.8},
        }

        selected_params = params.get(emotion)
        if not selected_params:
            return ["AI가 추천할 파라미터를 찾지 못했어요."]

        results = sp_client.recommendations(limit=20, country="KR", **selected_params)
        
        tracks = results['tracks']
        if not tracks:
            return ["AI가 추천할 노래를 찾지 못했어요."]

        random_tracks = random.sample(tracks, min(3, len(tracks)))
        return [f"{track['name']} - {track['artists'][0]['name']}" for track in random_tracks]

    except Exception as e:
        return [f"Spotify AI 추천 오류: {e}"]

def recommend(final_emotion, method):
    if method == 'AI 자동 추천':
        music_recs = get_spotify_ai_recommendations(final_emotion)
    else:
        music_recs = get_spotify_playlist_recommendations(final_emotion)
    
    recommendations = { "행복": {"책": ["기분을 관리하면 인생이 관리된다"], "영화": ["월터의 상상은 현실이 된다"]}, "사랑": {"책": ["사랑의 기술"], "영화": ["어바웃 타임"]}, "슬픔": {"책": ["아몬드"], "영화": ["코코"]}, "분노": {"책": ["분노의 심리학"], "영화": ["성난 사람들 (드라마)"]}, "힘듦": {"책": ["죽고 싶지만 떡볶이는 먹고 싶어"], "영화": ["리틀 포레스트"]}, "놀람": {"책": ["데미안"], "영화": ["유전"]},}
    recs = recommendations.get(final_emotion, {"책": [], "영화": []})
    recs['음악'] = music_recs
    return recs

def save_feedback_to_gsheets(client, feedback_df):
    try:
        spreadsheet = client.open("diary_app_feedback")
        worksheet = spreadsheet.worksheet("Sheet1")
        rows_to_add = feedback_df[['text', 'label']].values.tolist()
        worksheet.append_rows(rows_to_add, value_input_option='USER_ENTERED')
        st.success("소중한 피드백이 Google Sheets에 안전하게 저장되었습니다!")
        st.cache_data.clear()
    except Exception as e:
        st.error(f"피드백 저장 중 오류 발생: {e}")

def generate_random_diary():
    morning_starts = [ "아침 일찍 일어나 상쾌하게 하루를 시작했다.", "늦잠을 자서 허둥지둥 출근 준비를 했다." ]
    midday_events = [ "점심으로 먹은 파스타가 정말 맛있어서 기분이 좋았다.", "동료에게 칭찬을 들어서 뿌듯했다.", "오후 회의가 너무 길어져서 진이 빠졌다.", "사소한 실수 때문에 팀장님께 지적을 받아서 속상했다." ]
    evening_conclusions = [ "퇴근 후 운동을 하고 나니 몸은 힘들었지만 기분은 상쾌했다.", "자기 전 본 영화가 너무 감동적이어서 여운이 남는다." ]
    diary_parts = []
    diary_parts.append(random.choice(morning_starts))
    num_midday_events = random.randint(1, 2)
    selected_midday_events = random.sample(midday_events, num_midday_events)
    diary_parts.extend(selected_midday_events)
    diary_parts.append(random.choice(evening_conclusions))
    return " ".join(diary_parts)

def handle_random_click():
    st.session_state.diary_text = generate_random_diary()
    st.session_state.analysis_results = None

def handle_analyze_click(model, vectorizer):
    diary_content = st.session_state.diary_text
    if not diary_content.strip():
        st.warning("일기를 입력해주세요!")
    elif model is None or vectorizer is None:
        st.error("모델이 로드되지 않았습니다.")
    else:
        with st.spinner('AI가 일기를 분석하고 있습니다...'):
            _, results = analyze_diary_ml(model, vectorizer, diary_content)
            st.session_state.analysis_results = results
            
st.set_page_config(layout="wide")
st.title("📊 하루 감정 분석 리포트 (v7.20)")

with st.expander("⚙️ 시스템 상태 확인"):
    if st.secrets.get("connections", {}).get("gsheets"): st.success("✅ Google Sheets 인증 정보가 확인되었습니다.")
    else: st.error("❗️ Google Sheets 인증 정보('connections.gsheets')를 찾을 수 없습니다.")
    if st.secrets.get("spotify", {}).get("client_id") and st.secrets.get("spotify", {}).get("client_secret"): st.success("✅ Spotify 인증 정보가 확인되었습니다.")
    else: st.error("❗️ Spotify 인증 정보('[spotify]' 섹션)를 찾을 수 없거나 키 이름이 틀렸습니다.")
    model, vectorizer = load_ml_resources()
    if model and vectorizer: st.success("✅ AI 모델 파일이 성공적으로 로드되었습니다.")
    else: st.error("❗️ AI 모델 파일('sentiment_model.pkl')을 찾을 수 없습니다.")
st.divider()

if 'diary_text' not in st.session_state: st.session_state.diary_text = ""
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'rec_method' not in st.session_state: st.session_state.rec_method = '내 플레이리스트'

col1, col2 = st.columns([3, 1])
with col1:
    st.text_area("오늘의 일기를 시간의 흐름에 따라 작성해보세요:", key='diary_text', height=250)
with col2:
    st.write(" "); st.write(" ")
    st.radio("음악 추천 방식 선택", ('내 플레이리스트', 'AI 자동 추천'), key='rec_method', horizontal=True)
    st.button("🔄 랜덤 일기 생성", on_click=handle_random_click)
    st.button("🔍 내 하루 감정 분석하기", type="primary", on_click=handle_analyze_click, args=(model, vectorizer))

if st.session_state.analysis_results:
    if model and vectorizer:
        scores_data, _ = analyze_diary_ml(model, vectorizer, st.session_state.diary_text)
        df_scores = pd.DataFrame(scores_data).T
        if df_scores.sum().sum() > 0:
            final_emotion = df_scores.sum().idxmax()
            st.subheader("🕒 시간대별 감정 분석 결과")
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
            recs = recommend(final_emotion, st.session_state.rec_method)
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            with rec_col1:
                st.write("📚 **이런 책은 어때요?**")
                for item in recs['책']: st.write(f"- {item}")
            with rec_col2:
                st.write("🎵 **이런 음악도 들어보세요?**")
                for item in recs['음악']: st.write(f"- {item}")
            with rec_col3:
                st.write("🎬 **이런 영화/드라마도 추천해요?**")
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
                else: st.error("Google Sheets에 연결할 수 없습니다.")
st.divider()
with st.expander("피드백 저장 현황 보기 (Google Sheets)"):
    client = get_gsheets_connection()
    if client:
        df = fetch_all_data_from_gsheets(client)
        if not df.empty:
            st.dataframe(df.tail())
            st.info(f"현재 총 **{len(df)}개**의 데이터가 저장되어 있습니다. (1분마다 갱신)")
        else:
            st.write("아직 저장된 데이터가 없습니다.")
    else:
        st.error("Google Sheets에 연결할 수 없습니다. Secrets 설정을 다시 확인해주세요.")