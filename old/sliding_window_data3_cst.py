import pandas as pd
import numpy as np

def create_sliding_windows(dataframe, target_column, input_sequence_length, output_sequence_length, stride):
    """
    슬라이딩 윈도우를 사용하여 시계열 데이터셋을 생성합니다.

    Args:
        dataframe (pd.DataFrame): 시계열 데이터가 포함된 DataFrame.
                                  'timestamp' 컬럼이 datetime 형식으로 정렬되어 있어야 합니다.
        target_column (str): 예측하고자 하는 대상 컬럼명 (예: 'DATA3').
        input_sequence_length (int): 입력 시퀀스의 길이 (과거 데이터 포인트 수).
        output_sequence_length (int): 출력 시퀀스의 길이 (미래 예측 데이터 포인트 수).
        stride (int): 윈도우를 이동시킬 간격 (데이터 포인트 수).

    Returns:
        tuple: (X, y)
            X (np.array): 학습 입력 데이터 (과거 시퀀스). 형태: (샘플 수, input_sequence_length, 1)
            y (np.array): 학습 타겟 데이터 (미래 시퀀스). 형태: (샘플 수, output_sequence_length, 1)
    """
    data = dataframe[target_column].values
    X, y = [], []

    # 전체 시퀀스 길이 (과거 + 미래)
    total_sequence_length = input_sequence_length + output_sequence_length

    # 윈도우 생성 루프
    for i in range(0, len(data) - total_sequence_length + 1, stride):
        # 입력 시퀀스: 현재 위치부터 input_sequence_length 만큼의 과거 데이터
        input_seq = data[i : i + input_sequence_length]

        # 출력 시퀀스: input_seq 바로 다음부터 output_sequence_length 만큼의 미래 데이터
        output_seq = data[i + input_sequence_length : i + total_sequence_length]

        X.append(input_seq)
        y.append(output_seq)

    return np.array(X), np.array(y)

# --- 예시 데이터 생성 (실제 데이터셋으로 대체해야 함) ---
# 실제로는 계측기 수집 정보 데이터셋을 로드하고 전처리해야 합니다.
# 이 예시는 5분 간격으로 2년치 데이터가 있다고 가정합니다.
# 2020-09-26 00:00부터 2021-01-06 23:55까지의 실제 데이터 범위에 맞춰 대략 102일치 데이터로 생성.
# 102일 * 288포인트/일 = 29376 포인트
# 예측기간이 너무 길어지면 정확도가 급감함. 그러므로 1주일 가량 정도 예측을 위해, sliding윈도우도 1주일로 산정.
df = pd.read_csv(r'D:\ind_data_valid\CST_OP_COLLECTION.csv')

print("더미 데이터셋의 일부:")
print(df.head())
print(f"더미 데이터셋 총 길이: {len(df)} 포인트\n")


# --- 슬라이딩 윈도우 파라미터 설정 ---
# 1주일 = 7일 * 24시간 * 60분/시간 / 5분/포인트 = 2016 포인트
POINTS_PER_DAY = 288 # 24 * 60 / 5

input_sequence_length = 7 * POINTS_PER_DAY  # 과거 1주일 데이터 (2016 포인트)
output_sequence_length = 7 * POINTS_PER_DAY # 미래 1주일 예측 (2016 포인트)
stride = 1 * POINTS_PER_DAY                  # 윈도우를 1일(288포인트)씩 이동

print(f"입력 시퀀스 길이: {input_sequence_length} 포인트 (약 {input_sequence_length / POINTS_PER_DAY}일)")
print(f"출력 시퀀스 길이: {output_sequence_length} 포인트 (약 {output_sequence_length / POINTS_PER_DAY}일)")
print(f"슬라이딩 스텝: {stride} 포인트 (약 {stride / POINTS_PER_DAY}일)\n")

# --- 슬라이딩 윈도우 데이터셋 생성 ---
X_train, y_train = create_sliding_windows(
    dataframe=df,
    target_column='DATA3',
    input_sequence_length=input_sequence_length,
    output_sequence_length=output_sequence_length,
    stride=stride
)

print(f"생성된 학습 입력(X_train) 데이터 형태: {X_train.shape}")
print(f"생성된 학습 타겟(y_train) 데이터 형태: {y_train.shape}")

# X_train과 y_train은 각각 (샘플 수, 시퀀스 길이) 형태입니다.
# LSTM, GRU 등 시퀀스 모델에 넣기 위해 (샘플 수, 시퀀스 길이, 피처 수)로 reshape
# 여기서는 'DATA3' 하나만 예측하므로 피처 수는 1입니다.
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)

print(f"\nReshape 후 학습 입력(X_train) 데이터 형태: {X_train.shape}")
print(f"Reshape 후 학습 타겟(y_train) 데이터 형태: {y_train.shape}")

# 첫 번째 샘플 확인
print("\n첫 번째 X_train 샘플 (과거 1주일 A-상 전압):")
print(X_train[0, :5].flatten(), "...", X_train[0, -5:].flatten()) # 앞뒤 5개만 출력
print("\n첫 번째 y_train 샘플 (미래 1주일 A-상 전압 예측 목표):")
print(y_train[0, :5].flatten(), "...", y_train[0, -5:].flatten()) # 앞뒤 5개만 출력

# CSV 파일로 저장
#    - Excel이나 텍스트 편집기에서 직접 내용을 확인하고 싶을 때 사용합니다.
#    - 3차원 배열 (샘플 수, 시퀀스 길이, 1)을 2차원 배열 (샘플 수, 시퀀스 길이)로 변환하여 저장합니다.
X_train_reshaped_for_csv = X_train.reshape(X_train.shape[0], X_train.shape[1])
y_train_reshaped_for_csv = y_train.reshape(y_train.shape[0], y_train.shape[1])

output_dir = r'D:\ind_data_valid\slided_train_data'
np.savetxt(f'{output_dir}\X_train_data3.csv', X_train_reshaped_for_csv, delimiter=",")
np.savetxt(f'{output_dir}\y_train_data3.csv', y_train_reshaped_for_csv, delimiter=",")
print(f"\nCSV 파일 저장 완료:")
print(f"  X_train_data3.csv (형태: {X_train_reshaped_for_csv.shape}) - 각 행이 하나의 입력 시퀀스")
print(f"  y_train_data3.csv (형태: {y_train_reshaped_for_csv.shape}) - 각 행이 하나의 출력 시퀀스")
