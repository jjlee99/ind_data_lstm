# -*- coding: utf-8 -*-
"""
가스 사용량 수요 예측을 위한 시계열 분석 모델 (LSTM 기반)
- 목적: 향후 일정 시간 간격의 가스 사용량 예측 (회귀)
- 프레임워크: PyTorch
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt # Plotly로 대체
import os # 폴더 생성을 위해 추가
from datetime import datetime
# import seaborn as sns # plotly 대체
import plotly.graph_objects as go
import plotly.express as px
import time

# 검증 윈도우별 평균 MAE를 저장하기 위한 변수 (스크립트 상단 또는 주요 로직 시작 전 초기화)
avg_mae_val_windows = float('nan')
start_time = time.time()  # 시작 시간 기록
# -------------------- 설정 변수 (사용자 지정 가능) --------------------
SELECTED_EQUIP_NO = '82501456832002'  # 분석할 장비 번호 선택, 82501456231912(가장 많은 데이터셋)
TARGET_COLUMN = 'DATA3'             # 예측할 데이터 컬럼 (예: 'DATA1', 'DATA2', ..., 'DATA23')
DATA_FILE_PATH = "/workspace/data/CST_OP_COLLECTION.csv" # 데이터 파일 경로

# -------------------- 데이터셋 클래스 정의 --------------------
# # Seaborn 테마 설정 (matplotlib 임포트 후, plt 사용 전에 위치하는 것이 일반적)
# sns.set_theme(style="whitegrid")

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# -------------------- LSTM 모델 정의 --------------------
class GasDemandLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_sequence_length, num_layers=2):
        super(GasDemandLSTM, self).__init__()
        self.output_sequence_length = output_sequence_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # LSTM의 마지막 hidden state를 받아 output_sequence_length 만큼의 출력을 생성
        self.linear = nn.Linear(hidden_size, output_sequence_length)

    def forward(self, x):
        # x shape: (batch_size, input_sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, input_sequence_length, hidden_size)
        
        # 마지막 타임 스텝의 LSTM 출력을 사용
        last_time_step_out = lstm_out[:, -1, :] # shape: (batch_size, hidden_size)
        prediction = self.linear(last_time_step_out) # shape: (batch_size, output_sequence_length)
        
        # 타겟 y의 형태 (batch_size, output_sequence_length, 1)에 맞추기 위해 unsqueeze
        return prediction.unsqueeze(-1)

# -------------------- 슬라이딩 윈도우 생성 함수 (다중 출력용) --------------------
def create_sliding_windows(data_series, input_sequence_length, output_sequence_length, stride):
    """
    슬라이딩 윈도우를 사용하여 시계열 데이터셋을 생성합니다. (다중 스텝 예측용)

    Args:
        data_series (pd.Series or np.array): 시계열 데이터 (1D 또는 2D (n_points, 1 feature)).
        input_sequence_length (int): 입력 시퀀스의 길이.
        output_sequence_length (int): 출력 시퀀스의 길이 (예측 기간).
        stride (int): 윈도우 이동 간격.

    Returns:
        tuple: (X, y)
            X (np.array): 입력 데이터. 형태: (샘플 수, input_sequence_length, 1)
            y (np.array): 타겟 데이터. 형태: (샘플 수, output_sequence_length, 1)
    """
    if isinstance(data_series, pd.Series):
        data = data_series.values
    else:
        data = data_series

    if data.ndim == 1:
        data = data.reshape(-1, 1) # (n_points, 1 feature) 형태로 변환

    X_list, y_list = [], []
    total_data_length = len(data)
    total_sample_length = input_sequence_length + output_sequence_length

    for i in range(0, total_data_length - total_sample_length + 1, stride):
        input_seq = data[i : i + input_sequence_length]
        output_seq = data[i + input_sequence_length : i + input_sequence_length + output_sequence_length]
        X_list.append(input_seq)
        y_list.append(output_seq)

    if not X_list:
        return np.array([]).reshape(0, input_sequence_length, 1), np.array([]).reshape(0, output_sequence_length, 1)

    return np.array(X_list), np.array(y_list)

# -------------------- 데이터 불러오기 및 전처리 --------------------
print(f"데이터 파일 로드 중: {DATA_FILE_PATH}")
df_full_load = pd.read_csv(DATA_FILE_PATH)

# EQUIP_NO 컬럼을 문자열로 변환 (데이터 파일에 따라 숫자형일 수 있음)
if 'EQUIP_NO' not in df_full_load.columns:
    raise ValueError(f"'{DATA_FILE_PATH}' 파일에 'EQUIP_NO' 컬럼이 없습니다.")
df_full_load['EQUIP_NO'] = df_full_load['EQUIP_NO'].astype(str)

print(f"선택된 장비 번호: {SELECTED_EQUIP_NO}")
df_filtered_by_equip = df_full_load[df_full_load['EQUIP_NO'] == SELECTED_EQUIP_NO].copy()

if df_filtered_by_equip.empty:
    raise ValueError(f"선택된 장비 번호 '{SELECTED_EQUIP_NO}'에 해당하는 데이터가 없습니다. 장비 번호를 확인해주세요.")

print(f"장비 '{SELECTED_EQUIP_NO}'에 대한 데이터 {len(df_filtered_by_equip)}건을 찾았습니다.")

# 필터링된 데이터프레임에 대해 시간 관련 처리 수행
df_filtered_by_equip['REPORT_DATETIME'] = pd.to_datetime(
    df_filtered_by_equip['REPORT_DATE'].astype(str) + df_filtered_by_equip['REPORT_TIME'].astype(str).str.zfill(4),
    format="%Y%m%d%H%M"
)
df = df_filtered_by_equip.sort_values("REPORT_DATETIME").reset_index(drop=True) # 정렬 후 인덱스 재설정

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"선택된 장비 '{SELECTED_EQUIP_NO}'의 데이터에 '{TARGET_COLUMN}' 컬럼이 없습니다.")

# -------------------- 시각화 결과 저장 폴더 생성 (조기 생성으로 변경) --------------------
visualization_dir = "visualization"
os.makedirs(visualization_dir, exist_ok=True)


# --- 특정 기간의 불연속적 데이터 전처리 ---
# 사용자가 언급한 문제 구간 및 값 정의
preprocess_info_msg_printed = False

# 전처리 실행 여부 플래그 (필요에 따라 False로 변경하여 이 전처리 단계를 건너뛸 수 있습니다)
ENABLE_SPECIFIC_PREPROCESSING = False

if ENABLE_SPECIFIC_PREPROCESSING and SELECTED_EQUIP_NO == '82501456231912' and TARGET_COLUMN == 'DATA3':
    preprocess_start_str = "2020-10-19 00:00:00"
    preprocess_end_str = "2020-10-26 23:59:59" # 10월 27일 전까지

    preprocess_start_dt = pd.to_datetime(preprocess_start_str)
    preprocess_end_dt = pd.to_datetime(preprocess_end_str)

    # TARGET_COLUMN을 숫자형으로 변환 (오류 발생 시 NaN으로)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    original_nan_mask_full = df[TARGET_COLUMN].isna() # 전처리 전 원래 NaN 위치 기록

    # 전처리 대상 기간 마스크
    period_mask = (df['REPORT_DATETIME'] >= preprocess_start_dt) & \
                  (df['REPORT_DATETIME'] <= preprocess_end_dt)
    
    num_total_affected_points = 0
    if period_mask.any(): # 해당 기간에 데이터가 있는 경우
        print(f"\n[전처리 알림] 장비 '{SELECTED_EQUIP_NO}', 컬럼 '{TARGET_COLUMN}':")
        print(f"  기간 ({preprocess_start_str} ~ {preprocess_end_str}) 내에서")
        print(f"  계단형 또는 플랫한 구간을 자동으로 식별하여 선형 보간 처리합니다.")

        # 해당 기간의 데이터만 추출하여 처리 (주의: 인덱스는 원본 df의 인덱스 유지)
        series_in_period_indices = df[period_mask].index
        
        # 플랫 구간 식별을 위한 파라미터
        flat_diff_threshold = 0.1  # 연속된 값의 최대 절대 차이 (이 값보다 작으면 플랫으로 간주)
                                   # 데이터의 스케일과 변동성을 보고 조정 필요
        min_flat_duration = 6     # 최소 플랫 구간 길이 (포인트 수, 예: 5분 간격 데이터면 30분)

        points_to_mark_nan = pd.Series(False, index=df.index) # 전체 df에 대한 마스크

        i = 0
        while i < len(series_in_period_indices):
            current_original_idx = series_in_period_indices[i]
            
            if pd.isna(df.loc[current_original_idx, TARGET_COLUMN]):
                i += 1
                continue

            current_segment_indices = [current_original_idx]
            j = i + 1
            while j < len(series_in_period_indices):
                next_original_idx = series_in_period_indices[j]
                if pd.isna(df.loc[next_original_idx, TARGET_COLUMN]):
                    break 
                
                # 현재 세그먼트의 마지막 값과 다음 값의 차이로 플랫 여부 판단
                # 또는 세그먼트의 첫 값과 다음 값의 차이로 판단할 수도 있음
                # 여기서는 직전 값과의 차이로 판단
                prev_val_in_segment = df.loc[current_segment_indices[-1], TARGET_COLUMN]
                next_val = df.loc[next_original_idx, TARGET_COLUMN]

                if abs(next_val - prev_val_in_segment) < flat_diff_threshold:
                    current_segment_indices.append(next_original_idx)
                    j += 1
                else:
                    break 
            
            if len(current_segment_indices) >= min_flat_duration:
                points_to_mark_nan.loc[current_segment_indices] = True
                num_total_affected_points += len(current_segment_indices)
            
            i = j # 다음 탐색 시작 위치는 현재 플랫 세그먼트 바로 다음

        if num_total_affected_points > 0:
            df.loc[points_to_mark_nan, TARGET_COLUMN] = np.nan
            print(f"  총 {num_total_affected_points}개의 데이터 포인트를 보간 대상으로 표시했습니다.")
            preprocess_info_msg_printed = True
        else:
            print(f"  자동으로 식별된 계단형/플랫 구간이 없거나 기준 미달입니다.")
            # preprocess_info_msg_printed는 False로 유지 (실제 전처리 안 일어남)
    
    # 전처리가 일어났거나, 원래 NaN이 있었던 경우에만 보간/채우기 실행
    if preprocess_info_msg_printed or original_nan_mask_full.any():
        df[TARGET_COLUMN].interpolate(method='linear', limit_direction='both', inplace=True)
        df[TARGET_COLUMN].fillna(method='bfill', inplace=True)
        df[TARGET_COLUMN].fillna(method='ffill', inplace=True)
        if preprocess_info_msg_printed : # 실제 전처리가 일어난 경우에만 메시지 출력
             print(f"  선형 보간 및 NaN 채우기 완료.")

if preprocess_info_msg_printed:
    print("  특정 기간 불연속 데이터 전처리 완료.\n")

# --- 전처리 후 데이터 시각화 (선택 사항) ---
if preprocess_info_msg_printed or ENABLE_SPECIFIC_PREPROCESSING: # 전처리가 실행되었거나, 실행하도록 설정된 경우
    fig_plotly_preprocess = px.line(df, x='REPORT_DATETIME', y=TARGET_COLUMN,
                                    title=f"장비: {SELECTED_EQUIP_NO} - 전처리 후 {TARGET_COLUMN} 전체 데이터",
                                    labels={'REPORT_DATETIME': "날짜 및 시간", TARGET_COLUMN: f"{TARGET_COLUMN} 값"})
    fig_plotly_preprocess.update_traces(line=dict(color='green', width=1), name=f"Preprocessed {TARGET_COLUMN}", showlegend=True)
    
    file_path_html = os.path.join(visualization_dir, "after_preprocess_data.html")
    fig_plotly_preprocess.write_html(file_path_html)
    print(f"전처리 후 데이터 시각화 저장: {file_path_html}")
    # fig_plotly_preprocess.show()


full_series = df[TARGET_COLUMN].values.reshape(-1, 1)

scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(full_series)
# --- 슬라이딩 윈도우 파라미터 설정 ---
POINTS_PER_DAY = 24 * (60 // 5) # 5분 간격 데이터이므로 하루에 288 포인트

INPUT_SEQ_DAYS = 7    # 과거 7일 데이터 사용
OUTPUT_SEQ_DAYS = 7   # 미래 7일 예측
STRIDE_DAYS = 1       # 1일 간격으로 윈도우 이동

INPUT_SEQ_LENGTH = INPUT_SEQ_DAYS * POINTS_PER_DAY     # 2016 포인트
OUTPUT_SEQ_LENGTH = OUTPUT_SEQ_DAYS * POINTS_PER_DAY   # 2016 포인트
STRIDE_POINTS = STRIDE_DAYS * POINTS_PER_DAY           # 288 포인트
# 참고: STRIDE_POINTS를 작게 설정하면 학습 샘플 수가 늘어나지만, 샘플 간 중복도 커집니다.
# 현재 설정(데이터 약 103일, stride 1일) 시 약 89개의 학습 샘플이 생성됩니다. 데이터가 더 많거나 stride를 줄이면 샘플 수가 늘어납니다.
# 슬라이딩 개수
if len(full_series) < (INPUT_SEQ_LENGTH + OUTPUT_SEQ_LENGTH):
    raise ValueError(
        f"장비 '{SELECTED_EQUIP_NO}'의 '{TARGET_COLUMN}' 데이터가 너무 적어 ({len(full_series)} 포인트) "
        f"윈도우(입력 {INPUT_SEQ_LENGTH}, 출력 {OUTPUT_SEQ_LENGTH})를 생성할 수 없습니다. "
        "데이터 기간을 확인하거나 윈도우 크기를 줄여주세요."
    )

X, y = create_sliding_windows(series_scaled, 
                               input_sequence_length=INPUT_SEQ_LENGTH,
                               output_sequence_length=OUTPUT_SEQ_LENGTH,
                               stride=STRIDE_POINTS)

if X.shape[0] == 0:
    raise ValueError("생성된 윈도우 샘플이 없습니다. 데이터 양이나 윈도우/stride 설정을 확인하세요.")

X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)

batch_size = 16 # 데이터셋 크기에 따라 조정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# -------------------- 학습 루프 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = GasDemandLSTM(input_size=1, hidden_size=64, output_sequence_length=OUTPUT_SEQ_LENGTH, num_layers=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 50 # 예측 기간이 길고 복잡한 패턴 학습을 위해 epoch 증가 (필요시 더 늘리거나 조기 종료 추가)
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for seqs, targets in train_loader:
        seqs, targets = seqs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * seqs.size(0)

    train_loss /= len(train_loader.dataset)
    
    # 검증 손실
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for seqs, targets in val_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            outputs = model(seqs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * seqs.size(0)
    val_loss /= len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

# -------------------- 검증 데이터셋 첫 샘플에 대한 예측 및 결과 시각화 --------------------
model.eval()
all_val_preds_scaled_list = []
all_val_actuals_scaled_list = []

with torch.no_grad():
    for seqs, targets in val_loader:
        seqs = seqs.to(device)
        outputs = model(seqs)
        all_val_preds_scaled_list.append(outputs.cpu().numpy())
        all_val_actuals_scaled_list.append(targets.cpu().numpy())

if not all_val_preds_scaled_list:
    print("검증 데이터가 없습니다. 검증 결과 시각화를 건너뜁니다.")
else:
    # 모든 배치를 하나로 합침
    all_val_preds_scaled = np.concatenate(all_val_preds_scaled_list, axis=0) # (total_val_samples, OUTPUT_SEQ_LENGTH, 1)
    all_val_actuals_scaled = np.concatenate(all_val_actuals_scaled_list, axis=0) # (total_val_samples, OUTPUT_SEQ_LENGTH, 1)

    if all_val_preds_scaled.shape[0] > 0:
        # 첫 번째 검증 샘플을 사용 (shape: (OUTPUT_SEQ_LENGTH, 1))
        val_pred_sample_scaled = all_val_preds_scaled[0]
        val_actual_sample_scaled = all_val_actuals_scaled[0]

        # 스케일 역변환
        val_pred_sample_inverse = scaler.inverse_transform(val_pred_sample_scaled)
        val_actual_sample_inverse = scaler.inverse_transform(val_actual_sample_scaled)

        fig_plotly_val_sample = go.Figure()
        time_steps_val = list(range(len(val_actual_sample_inverse)))
        fig_plotly_val_sample.add_trace(go.Scatter(x=time_steps_val, y=val_actual_sample_inverse.flatten(),
                                           mode='lines', name="Actual (Validation First Sample)"))
        fig_plotly_val_sample.add_trace(go.Scatter(x=time_steps_val, y=val_pred_sample_inverse.flatten(),
                                           mode='lines', name="Predicted (Validation First Sample)"))
        fig_plotly_val_sample.update_layout(title=f"장비: {SELECTED_EQUIP_NO} - {TARGET_COLUMN} 예측 (검증 첫 샘플)",
                                   xaxis_title=f"Time Steps (다음 {OUTPUT_SEQ_DAYS}일)",
                                   yaxis_title=f"{TARGET_COLUMN} 값",
                                   legend_title="Legend")
        
        file_path_html = os.path.join(visualization_dir, "validation_first_sample_forecast.html")
        fig_plotly_val_sample.write_html(file_path_html)
        print(f"검증 데이터 첫 샘플 예측 그래프 저장: {file_path_html}")
        # fig_plotly_val_sample.show()
    else:
        print("검증 데이터 샘플이 없어 시각화할 수 없습니다.")

# -------------------- 전체 검증 기간에 대한 예측 성능 시각화 --------------------
if len(X_val) > 0 and len(y_val) > 0:
    print("\n전체 검증 기간에 대한 예측 성능 시각화 생성 중...")
    fig_plotly_overall_val = go.Figure()

    # 1. 검증 기간 전체의 실제값 플로팅
    #    검증 데이터의 첫 번째 타겟 시퀀스가 시작하는 인덱스
    val_period_actual_start_idx = len(X_train) * STRIDE_POINTS + INPUT_SEQ_LENGTH
    #    검증 데이터의 마지막 타겟 시퀀스가 끝나는 인덱스
    val_period_actual_end_idx = (len(X_train) + len(X_val) - 1) * STRIDE_POINTS + INPUT_SEQ_LENGTH + OUTPUT_SEQ_LENGTH

    if val_period_actual_end_idx <= len(df):
        actual_val_period_timestamps = df['REPORT_DATETIME'].iloc[val_period_actual_start_idx:val_period_actual_end_idx]
        actual_val_period_values = full_series[val_period_actual_start_idx:val_period_actual_end_idx].flatten()
        fig_plotly_overall_val.add_trace(go.Scatter(x=actual_val_period_timestamps, y=actual_val_period_values,
                                           mode='lines', name=f"Actual {TARGET_COLUMN} (Validation Period)",
                                           line=dict(color='blue', width=1.5)))

        # 2. 검증 데이터 각 윈도우에 대한 예측값 플로팅
        model.eval()
        first_prediction_plotted = False
        with torch.no_grad():
            for i in range(len(X_val)):
                val_input_tensor = X_val[i].unsqueeze(0).to(device)
                val_output_scaled = model(val_input_tensor)
                val_output_actual = scaler.inverse_transform(val_output_scaled.cpu().numpy()[0]).flatten()

                # 이 예측에 해당하는 타임스탬프 계산
                pred_start_idx = (len(X_train) + i) * STRIDE_POINTS + INPUT_SEQ_LENGTH
                pred_end_idx = pred_start_idx + OUTPUT_SEQ_LENGTH
                
                if pred_end_idx <= len(df):
                    timestamps_for_pred = df['REPORT_DATETIME'].iloc[pred_start_idx:pred_end_idx]
                    show_legend_flag = not first_prediction_plotted
                    # 모든 예측선(빨간색)이 동일한 이름을 갖도록 하여 범례에서 그룹화합니다.
                    # 첫 번째 예측선에 대해서만 showlegend=True로 설정하여 범례 항목이 한 번만 나타나도록 합니다.
                    trace_name = "Model Predictions (Validation Windows)"

                    fig_plotly_overall_val.add_trace(go.Scatter(x=timestamps_for_pred, y=val_output_actual,
                                                       mode='lines', name=trace_name,
                                                       line=dict(color='red'), opacity=0.6,
                                                       showlegend=show_legend_flag))
                    if not first_prediction_plotted:
                        first_prediction_plotted = True
                else:
                    print(f"  경고: 검증 샘플 {i}의 예측 타임스탬프 범위를 벗어납니다. 일부 예측이 그려지지 않을 수 있습니다.")


        fig_plotly_overall_val.update_layout(title=f"장비: {SELECTED_EQUIP_NO} - {TARGET_COLUMN} 전체 검증 기간 예측 성능",
                                   xaxis_title="날짜 및 시간",
                                   yaxis_title=f"{TARGET_COLUMN} 값",
                                   legend_title="Legend")
        file_path_html = os.path.join(visualization_dir, "overall_validation_performance.html")
        fig_plotly_overall_val.write_html(file_path_html)
        print(f"전체 검증 기간 예측 성능 그래프 저장: {file_path_html}")
        # fig_plotly_overall_val.show()
    else:
        print("전체 검증 기간의 실제값 범위를 계산할 수 없습니다 (데이터 길이 부족). 시각화를 건너뜁니다.")
else:
    print("\n검증 데이터가 없어 전체 검증 기간 예측 성능 시각화를 건너뜁니다.")

# # -------------------- 검증 윈도우별 예측 오차(MAE) 시각화 (별도 그래프 생성 부분은 이제 선택사항 또는 제거 가능) --------------------
# if len(X_val) > 0 and len(y_val) > 0:
#     print("\n검증 윈도우별 예측 오차(MAE) 시각화 생성 중...")
#     model.eval()
#     window_maes = []
#     # window_start_timestamps_for_plot = [] # X축 레이블용 (필요시 사용)

#     with torch.no_grad():
#         # 이 루프는 위 'overall_validation_performance' 그래프 생성 시 MAE 계산과 중복될 수 있습니다.
#         # avg_mae_val_windows 변수는 이미 위에서 계산되었으므로,
#         # 이 섹션은 별도의 MAE 막대 그래프가 여전히 필요할 경우에만 유지합니다.
#         # 만약 overall_validation_performance 그래프의 주석으로 충분하다면 이 섹션은 주석 처리하거나 삭제해도 됩니다.
#         # 여기서는 일단 유지하되, avg_mae_val_windows를 다시 계산하지 않도록 주의합니다.
#         for i in range(len(X_val)):
#             # 현재 검증 윈도우에 대한 모델 예측 가져오기
#             val_input_tensor = X_val[i].unsqueeze(0).to(device)
#             pred_output_scaled = model(val_input_tensor) # Shape: (1, OUTPUT_SEQ_LENGTH, 1)
#             # 스케일 역변환 (예측값)
#             pred_output_actual = scaler.inverse_transform(pred_output_scaled.cpu().numpy()[0]).flatten() # Shape: (OUTPUT_SEQ_LENGTH,)

#             # 현재 검증 윈도우에 대한 실제 타겟값 가져오기
#             actual_output_scaled = y_val[i].cpu().numpy() # Shape: (OUTPUT_SEQ_LENGTH, 1)
#             # 스케일 역변환 (실제값)
#             actual_output_actual = scaler.inverse_transform(actual_output_scaled).flatten() # Shape: (OUTPUT_SEQ_LENGTH,)

#             # 이 윈도우에 대한 MAE 계산
#             mae = np.mean(np.abs(pred_output_actual - actual_output_actual))
#             window_maes.append(mae)

#     if window_maes: # MAE 계산이 수행된 경우
#         # avg_mae_val_windows는 이미 위에서 계산되었을 수 있으므로, 여기서는 로컬 변수로 사용하거나,
#         # 전역 변수를 업데이트하는 경우 중복 계산이 되지 않도록 주의해야 합니다.
#         print(f"  평균 MAE (검증 윈도우별): {avg_mae_val_windows:.4f}")

#         fig_mae_per_window = go.Figure()
        
#         x_axis_data = list(range(len(window_maes)))
#         x_axis_title = "검증 윈도우 인덱스"

#         fig_mae_per_window.add_trace(go.Bar(
#             x=x_axis_data,
#             y=window_maes,
#             name="윈도우별 MAE"
#         ))
#         fig_mae_per_window.add_hline(
#             y=avg_mae_val_windows,
#             line_dash="dash",
#             line_color="red",
#             annotation_text=f"평균 MAE: {avg_mae_val_windows:.4f}",
#             annotation_position="bottom right"
#         )
#         fig_mae_per_window.update_layout(
#             title=f"장비: {SELECTED_EQUIP_NO} - {TARGET_COLUMN} 검증 윈도우별 예측 MAE",
#             xaxis_title=x_axis_title,
#             yaxis_title="MAE (Mean Absolute Error)",
#             legend_title="Legend"
#         )
#         file_path_html = os.path.join(visualization_dir, "validation_window_mae_performance.html")
#         fig_mae_per_window.write_html(file_path_html)
#         print(f"  검증 윈도우별 MAE 그래프 저장: {file_path_html}")
# else:
#     print("\n검증 데이터가 없어 윈도우별 MAE 시각화를 건너뜁니다.") # avg_mae_val_windows는 float('nan')으로 유지됨
# -------------------- 최종 1주일 미래 예측 --------------------
# 전체 데이터셋의 마지막 INPUT_SEQ_LENGTH 만큼을 입력으로 사용
last_sequence_scaled = series_scaled[-INPUT_SEQ_LENGTH:]
last_sequence_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(device) # (1, INPUT_SEQ_LENGTH, 1)

model.eval()
with torch.no_grad():
    future_prediction_scaled = model(last_sequence_tensor) # (1, OUTPUT_SEQ_LENGTH, 1)

# CPU로 옮기고 numpy 배열로 변환 후, 스케일 역변환
future_prediction_scaled_np = future_prediction_scaled.cpu().numpy()[0, :, :] # (OUTPUT_SEQ_LENGTH, 1)
future_prediction_actual = scaler.inverse_transform(future_prediction_scaled_np)

# 예측된 기간에 대한 타임스탬프 생성
last_timestamp = df['REPORT_DATETIME'].iloc[-1] # 데이터의 마지막 시간
future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=5), # 데이터 간격이 5분이므로
                                  periods=OUTPUT_SEQ_LENGTH,
                                  freq='5min') # 데이터 간격에 맞춰 '5min'으로 설정

fig_plotly_future = go.Figure()
# 과거 데이터 일부와 함께 예측 결과 표시
history_to_plot = min(len(full_series), INPUT_SEQ_LENGTH + OUTPUT_SEQ_LENGTH + STRIDE_POINTS) # 표시할 과거 데이터 길이 조정 (예측에 사용된 기간 + 약간 더)
fig_plotly_future.add_trace(go.Scatter(x=df['REPORT_DATETIME'].iloc[-history_to_plot:], 
                                     y=full_series[-history_to_plot:].flatten(),
                                     mode='lines', name=f"Historical {TARGET_COLUMN}",
                                     line=dict(width=1), opacity=0.7))

# 만약 검증 데이터가 있다면, 마지막 검증 윈도우에 대한 예측을 함께 표시
if len(y_val) > 0:
    # 마지막 검증 윈도우의 입력과 실제 타겟
    last_val_input_tensor = X_val[-1].unsqueeze(0).to(device)

    # 마지막 검증 윈도우에 대한 예측
    with torch.no_grad():
        last_val_pred_scaled = model(last_val_input_tensor) # (1, OUTPUT_SEQ_LENGTH, 1)
    last_val_pred_actual = scaler.inverse_transform(last_val_pred_scaled.cpu().numpy()[0]).flatten()

    # 마지막 검증 윈도우의 타겟(y)에 해당하는 원본 데이터의 인덱스 계산
    idx_of_last_val_window_in_X = len(X_train) + len(y_val) - 1
    
    start_idx_target_for_last_val = idx_of_last_val_window_in_X * STRIDE_POINTS + INPUT_SEQ_LENGTH
    end_idx_target_for_last_val = start_idx_target_for_last_val + OUTPUT_SEQ_LENGTH
    
    if end_idx_target_for_last_val <= len(df):
        timestamps_for_last_val_target = df['REPORT_DATETIME'].iloc[start_idx_target_for_last_val:end_idx_target_for_last_val]
        fig_plotly_future.add_trace(go.Scatter(x=timestamps_for_last_val_target, y=last_val_pred_actual,
                                           mode='lines', name=f"Predicted {TARGET_COLUMN} (Last Validation Window)",
                                           line=dict(color='orange', dash='dash')))
    else:
        print("마지막 검증 윈도우의 타임스탬프를 계산할 수 없습니다 (데이터 길이 부족).")

fig_plotly_future.add_trace(go.Scatter(x=future_timestamps, y=future_prediction_actual.flatten(),
                                     mode='lines', name=f"Predicted {TARGET_COLUMN} (Forecast Next {OUTPUT_SEQ_DAYS} Days)",
                                     line=dict(color='red', width=2)))

fig_plotly_future.update_layout(title=f"장비: {SELECTED_EQUIP_NO} - {TARGET_COLUMN} 값: 과거, 최근 검증 예측 및 향후 {OUTPUT_SEQ_DAYS}일 예측",
                                xaxis_title="날짜 및 시간",
                                yaxis_title=f"{TARGET_COLUMN} 값",
                                legend_title="Legend")

file_path_html = os.path.join(visualization_dir, "future_1week_forecast.html")
fig_plotly_future.write_html(file_path_html)
# fig_plotly_future.show()

print(f"\n향후 {OUTPUT_SEQ_DAYS}일 ({OUTPUT_SEQ_LENGTH} 포인트) 예측 완료 및 그래프 저장: {file_path_html}")
print("첫 5개 예측값:", future_prediction_actual.flatten()[:5])
print("마지막 5개 예측값:", future_prediction_actual.flatten()[-5:])

end_time = time.time()  # 종료 시간 기록
elapsed_time = end_time - start_time  # 경과 시간 계산

print(f"Execution time: {elapsed_time} seconds")

# 학습 루프의 마지막 train_loss와 val_loss를 저장하기 위한 변수 초기화
final_train_loss = float('nan')
final_val_loss = float('nan')
actual_epochs_run = 0

if epochs > 0: # 학습이 실행된 경우에만 값 할당
    if 'train_loss' in locals(): # 학습 루프 내에서 train_loss가 할당되었는지 확인
        final_train_loss = train_loss
    if 'val_loss' in locals(): # 학습 루프 내에서 val_loss가 할당되었는지 확인
        final_val_loss = val_loss
    if 'epoch' in locals(): # epoch 변수가 학습 루프에서 정의되었는지 확인
        actual_epochs_run = epoch + 1

# -------------------- 학습 정보 저장 --------------------
info_file_path = os.path.join(os.getcwd(), "training_info.txt")
with open(info_file_path, "w", encoding='utf-8') as f:
    f.write("=========== Data & Preprocessing Info ===========\n")
    f.write(f"Selected Equipment No: {SELECTED_EQUIP_NO}\n")
    f.write(f"Target Column: {TARGET_COLUMN}\n")
    f.write(f"Data File Path: {DATA_FILE_PATH}\n")
    f.write(f"Specific Preprocessing Enabled: {ENABLE_SPECIFIC_PREPROCESSING}\n")
    
    if preprocess_info_msg_printed: # 특정 전처리가 실제로 수행되었을 때 관련 정보 기록
        f.write(f"  Preprocessing Period: {preprocess_start_str} to {preprocess_end_str}\n")
        f.write(f"  Flat Diff Threshold: {flat_diff_threshold}\n")
        f.write(f"  Min Flat Duration: {min_flat_duration}\n")
        f.write(f"  Points marked for interpolation: {num_total_affected_points}\n")
    elif ENABLE_SPECIFIC_PREPROCESSING and SELECTED_EQUIP_NO == '82501456231912' and TARGET_COLUMN == 'DATA3':
        # 전처리를 시도했으나, 조건에 맞는 구간이 없거나 해당 기간에 데이터가 없었던 경우
        f.write(f"  Preprocessing attempted for period {preprocess_start_str} to {preprocess_end_str} but no sections met criteria or no data in period.\n")

    f.write(f"Data points for selected equipment after initial load: {len(df_filtered_by_equip)}\n")
    f.write(f"Data points after datetime conversion and sorting (df): {len(df)}\n")
    f.write(f"Data points in full_series (target column values for model): {len(full_series)}\n")
    f.write(f"Data points in series_scaled (after scaling): {len(series_scaled)}\n")

    f.write("\n=========== Sliding Window Info ===========\n")
    f.write(f"Points per day (for window calculation): {POINTS_PER_DAY}\n")
    f.write(f"Input Sequence Days: {INPUT_SEQ_DAYS}\n")
    f.write(f"Output Sequence Days: {OUTPUT_SEQ_DAYS}\n")
    f.write(f"Stride Days: {STRIDE_DAYS}\n")
    f.write(f"Input Sequence Length (points): {INPUT_SEQ_LENGTH}\n")
    f.write(f"Output Sequence Length (points): {OUTPUT_SEQ_LENGTH}\n")
    f.write(f"Stride (points): {STRIDE_POINTS}\n")
    f.write(f"Total generated window samples (X.shape[0]): {X.shape[0]}\n")
    f.write(f"Training samples (X_train.shape[0]): {X_train.shape[0]}\n")
    f.write(f"Validation samples (X_val.shape[0]): {X_val.shape[0]}\n")

    f.write("\n=========== Model Hyperparameters ===========\n")
    f.write(f"LSTM Input Size: {model.lstm.input_size}\n")
    f.write(f"LSTM Hidden Size: {model.lstm.hidden_size}\n")
    f.write(f"LSTM Num Layers: {model.lstm.num_layers}\n")
    
    f.write("\n=========== Training Parameters ===========\n")
    f.write(f"Device: {device}\n")
    f.write(f"Actual Epochs Run: {actual_epochs_run}\n")
    f.write(f"Target Epochs: {epochs}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Learning Rate: {optimizer.param_groups[0]['lr']}\n")
    f.write(f"Criterion: {criterion.__class__.__name__}\n")
    f.write(f"Optimizer: {optimizer.__class__.__name__}\n")
    
    f.write("\n=========== Training Results ===========\n")
    f.write(f"Final Training Loss: {final_train_loss:.6f}\n") 
    f.write(f"Final Validation Loss (MSE): {final_val_loss:.6f}\n") # MSE임을 명시
    f.write(f"Average MAE per Validation Window: {avg_mae_val_windows:.6f}\n") # 추가된 정보
    f.write(f"Total Execution Time: {elapsed_time:.2f} seconds\n")

print(f"학습 정보 저장 완료: {info_file_path}")
