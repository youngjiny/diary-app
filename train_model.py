# train_model.py (Google Sheets 자동 연동 버전)

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from soynlp.tokenizer import LTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# ⭐️ 1. Google Sheets에서 데이터를 직접 불러오는 함수 추가
def load_data_from_gsheets():
    """Google Sheets에서 피드백 데이터를 직접 읽어와 DataFrame으로 반환합니다."""
    print("Google Sheets에서 최신 데이터 로딩 중...")
    try:
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        # 로컬에 저장된 인증키 파일 사용
        creds = Credentials.from_service_account_file('gsheets_credentials.json', scopes=scope)
        client = gspread.authorize(creds)
        
        spreadsheet = client.open("diary_app_feedback")
        worksheet = spreadsheet.worksheet("Sheet1")
        
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        print("데이터 로딩 완료!")
        return df
    except FileNotFoundError:
        print("오류: 'gsheets_credentials.json' 파일을 찾을 수 없습니다.")
        print("1단계 가이드에 따라 인증키 파일을 프로젝트 폴더에 준비해주세요.")
        return None
    except Exception as e:
        print(f"Google Sheets 데이터 로딩 중 오류 발생: {e}")
        return None

# --- 모델 훈련 시작 ---
print("--- 모델 훈련 시작 ---")

# ⭐️ 2. CSV를 읽는 대신, 위에서 만든 함수를 호출
data = load_data_from_gsheets()

# 데이터 로딩에 성공했는지 확인
if data is None or data.empty:
    print("데이터를 불러오지 못해 훈련을 중단합니다.")
    exit()

print(f"총 {len(data)}개의 데이터로 훈련을 시작합니다.")

# (이하 훈련 로직은 이전과 동일)
if len(data) < 50:
    X_train = data['text']
    y_train = data['label']
    print("데이터 양이 적어, 전체 데이터를 훈련에 사용합니다.")
else:
    X = data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tokenizer = LTokenizer()
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize, min_df=1, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

model = LogisticRegression(random_state=42)
model.fit(X_train_tfidf, y_train)

if len(data) >= 50:
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ 모델 정확도: {accuracy:.4f}")

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("--- 훈련 완료! ---")
print("📄 sentiment_model.pkl (모델) 와 📄 tfidf_vectorizer.pkl (벡터라이저) 파일이 새로 생성되었습니다.")