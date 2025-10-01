# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from soynlp.tokenizer import LTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

MASTER_DATA_FILE = "master_sentiment_data.csv"

print("--- 모델 훈련 시작 ---")

if not os.path.exists(MASTER_DATA_FILE):
    print(f"오류: '{MASTER_DATA_FILE}'을 찾을 수 없습니다.")
    print("먼저 generate_data.py를 실행하여 기본 데이터를 생성해주세요.")
    exit()

data = pd.read_csv(MASTER_DATA_FILE, encoding='cp949')
print(f"'{MASTER_DATA_FILE}'에서 {len(data)}개의 데이터로 훈련을 시작합니다.")

# 데이터가 비어 있는지 확인
if data.empty:
    print("오류: 데이터 파일이 비어있습니다.")
    exit()

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tokenizer = LTokenizer()
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=tokenizer.tokenize,
    min_df=2,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ 모델 정확도: {accuracy:.4f}")

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("--- 훈련 완료! ---")
print("📄 sentiment_model.pkl (모델) 와 📄 tfidf_vectorizer.pkl (벡터라이저) 파일이 생성되었습니다.")