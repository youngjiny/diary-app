import pandas as pd
import os

# 합칠 파일과 결과 파일 이름 정의
base_data_file = 'sentiment_data.csv'
feedback_file = 'feedback_data.csv'
total_data_file = 'total_data.csv'

print("--- 데이터 병합 시작 ---")

# 기본 데이터 로드
if os.path.exists(base_data_file):
    base_df = pd.read_csv(base_data_file, encoding='cp949')
    print(f"'{base_data_file}'에서 {len(base_df)}개 데이터 로드 완료.")
else:
    print(f"오류: '{base_data_file}'을 찾을 수 없습니다.")
    base_df = pd.DataFrame() # 빈 데이터프레임

# 피드백 데이터 로드
if os.path.exists(feedback_file):
    feedback_df = pd.read_csv(feedback_file, encoding='cp949')
    print(f"'{feedback_file}'에서 {len(feedback_df)}개 데이터 로드 완료.")

    # 두 데이터를 합침
    total_df = pd.concat([base_df, feedback_df], ignore_index=True)

    # 중복된 문장이 있으면, 가장 마지막에 들어온 피드백(최신 정보)을 유지
    total_df.drop_duplicates(subset=['text'], keep='last', inplace=True)

    print("데이터 병합 및 중복 제거 완료.")
else:
    print(f"'{feedback_file}'이 없어 병합을 건너뜁니다.")
    total_df = base_df

# 최종 데이터를 새 파일로 저장
total_df.to_csv(total_data_file, index=False, encoding='cp949')
print(f"✅ 성공! 총 {len(total_df)}개의 데이터가 '{total_data_file}'에 저장되었습니다.")