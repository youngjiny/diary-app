# generate_data.py

import csv
import random

TARGET_COUNT = 5000
FILE_NAME = "master_sentiment_data.csv"

# --- 문장 재료 확장 ---
common_actions = ["커피를 마셨다", "산책을 했다", "음악을 들었다", "영화를 봤다", "책을 읽었다", "운동을 했다", "친구와 통화했다"]
common_places = ["집에서", "회사에서", "카페에서", "공원에서", "지하철에서"]
common_people = ["동료가", "친구가", "가족이", "연인이"]

# --- 5가지 감정 템플릿 ---
templates = {
    "기쁨": [
        f"오늘 {random.choice(common_actions)} 정말 즐거웠다.",
        f"{random.choice(common_places)} 뜻밖에 좋은 소식을 들어서 하루 종일 기분이 좋았다.",
        f"{random.choice(common_people)} 칭찬해줘서 날아갈 것 같아.",
        "오랫동안 노력했던 일이 드디어 결실을 맺어서 너무 행복하다."
    ],
    "슬픔": [
        f"기대했던 약속이 취소되어서 너무 서운하고 슬펐다.",
        f"{random.choice(common_people)}와(과)의 추억이 떠올라 마음이 아팠다.",
        f"{random.choice(common_places)} 문득 혼자라는 생각에 눈물이 났다.",
        "아끼던 물건을 잃어버려서 너무 속상하다."
    ],
    "분노": [
        "상사가 말도 안 되는 일로 트집을 잡아서 화가 머리 끝까지 났다.",
        f"{random.choice(common_places)} 무례한 사람을 만나서 기분이 상했다.",
        "계획이 계속 틀어져서 너무 짜증나고 화가 난다.",
        "내 의견이 완전히 무시당해서 억울하고 분노가 치밀었다."
    ],
    "우울": [
        "요즘 계속 무기력하고 아무것도 하기 싫은 기분이다.",
        "날씨 때문인지 하루 종일 기분이 가라앉고 우울했다.",
        "미래에 대한 불안감 때문에 마음이 답답하다.",
        f"혼자 {random.choice(common_actions)} 계속 우울한 생각만 들었다."
    ],
    "사랑": [
        f"{random.choice(common_people)} 나를 위해 작은 선물을 준비해서 감동받았다.",
        "힘들 때 곁에서 위로해주는 사람이 있어서 정말 든든하고 사랑을 느꼈다.",
        "우리의 소중한 순간들을 떠올리니 사랑스러운 마음이 가득 찼다.",
        "따뜻한 눈빛과 다정한 말 한마디에 사랑받고 있음을 느꼈다."
    ]
}

print(f"--- {TARGET_COUNT}개의 신규 마스터 데이터 생성 시작 ---")
try:
    with open(FILE_NAME, 'w', newline='', encoding='cp949') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label']) # 헤더 작성

        for _ in range(TARGET_COUNT):
            label = random.choice(list(templates.keys()))
            sentence = random.choice(templates[label])
            writer.writerow([sentence, label])
    print(f"✅ 성공! '{FILE_NAME}' 파일에 {TARGET_COUNT}개의 데이터가 생성되었습니다.")
except Exception as e:
    print(f"오류 발생: {e}")