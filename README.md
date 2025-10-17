# 🔥 가스 사용량 수요 예측 시스템

LSTM 딥러닝 모델을 활용한 시계열 가스 사용량 예측 프로젝트

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [기술 스택](#-기술-스택)
- [시스템 아키텍처](#-시스템-아키텍처)
- [설치 방법](#-설치-방법)
- [사용 방법](#-사용-방법)
- [모델 설명](#-모델-설명)
- [결과 시각화](#-결과-시각화)
- [성능 지표](#-성능-지표)
- [라이선스](#-라이선스)

## 🎯 프로젝트 개요

본 프로젝트는 **LSTM(Long Short-Term Memory)** 딥러닝 모델을 활용하여 가스 사용량을 예측하는 시계열 분석 시스템입니다. 5분 단위로 수집된 가스 사용 데이터를 기반으로 향후 7일간의 사용량을 예측합니다.

### 핵심 목표
- 과거 7일 데이터를 학습하여 향후 7일 사용량 예측
- 장비별 맞춤형 예측 모델 구축
- 실시간 데이터 전처리 및 이상치 자동 보정
- 직관적인 시각화를 통한 예측 결과 제공

## ✨ 주요 기능

### 1. **데이터 전처리**
- 시계열 데이터 정렬 및 결측치 처리
- 계단형/플랫 구간 자동 감지 및 선형 보간
- MinMax Scaling을 통한 정규화

### 2. **슬라이딩 윈도우 기반 학습**
- 입력: 과거 7일 (2,016 포인트, 5분 간격)
- 출력: 향후 7일 (2,016 포인트)
- Stride: 1일 (288 포인트)

### 3. **LSTM 모델**
- 2-layer LSTM 아키텍처
- Hidden Size: 64
- 다중 스텝 예측 (Multi-step Forecasting)

### 4. **실시간 시각화**
- Plotly 기반 인터랙티브 차트
- 검증 데이터 예측 성능 분석
- 향후 예측 결과 시각화

## 🛠 기술 스택

### Core Framework
- **PyTorch** - 딥러닝 모델 구축
- **Python 3.8+** - 메인 언어

### Data Processing
- **Pandas** - 데이터 전처리
- **NumPy** - 수치 연산
- **scikit-learn** - 데이터 스케일링

### Visualization
- **Plotly** - 인터랙티브 시각화

### Hardware Acceleration
- **CUDA** - GPU 가속 (선택사항)

## 🏗 시스템 아키텍처

```
┌─────────────────┐
│  Raw CSV Data   │
│  (5분 단위)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Preprocess │
│  - 결측치 처리   │
│  - 이상치 보정   │
│  - 정규화        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sliding Window  │
│  Past 7 days    │
│  → Future 7 days│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LSTM Model    │
│  (2-layer, 64)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prediction &   │
│  Visualization  │
└─────────────────┘
```

## 📦 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/gas-demand-forecast.git
cd gas-demand-forecast
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 필수 패키지 설치
```bash
pip install torch pandas numpy scikit-learn plotly
```

### 4. 데이터 준비
- CSV 파일을 `/workspace/data/` 디렉토리에 배치
- 파일명: `CST_OP_COLLECTION.csv`

## 🚀 사용 방법

### 기본 실행
```bash
python gas_demand_forecast.py
```

### 설정 변경
스크립트 상단의 설정 변수를 수정하여 커스터마이징할 수 있습니다:

```python
# 분석할 장비 번호
SELECTED_EQUIP_NO = '82501456832002'

# 예측할 데이터 컬럼
TARGET_COLUMN = 'DATA3'

# 데이터 파일 경로
DATA_FILE_PATH = "/workspace/data/CST_OP_COLLECTION.csv"

# 특정 기간 전처리 활성화 여부
ENABLE_SPECIFIC_PREPROCESSING = False
```

### 하이퍼파라미터 조정
```python
# 모델 파라미터
hidden_size = 64
num_layers = 2

# 학습 파라미터
epochs = 50
batch_size = 16
learning_rate = 0.001
```

## 🧠 모델 설명

### LSTM 아키텍처

```python
GasDemandLSTM(
  (lstm): LSTM(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    batch_first=True
  )
  (linear): Linear(in_features=64, out_features=2016)
)
```

### 입력/출력 형태
- **입력**: `(batch_size, 2016, 1)` - 과거 7일 데이터
- **출력**: `(batch_size, 2016, 1)` - 향후 7일 예측

### 손실 함수
- **MSE Loss** (Mean Squared Error)
- 검증 지표: **MAE** (Mean Absolute Error)

### 최적화 알고리즘
- **Adam Optimizer**
- Learning Rate: 0.001

## 📊 결과 시각화

모델 실행 후 `visualization/` 디렉토리에 다음 결과물이 생성됩니다:

### 1. `after_preprocess_data.html`
- 전처리 후 전체 데이터 시각화
- 이상치 보정 확인

### 2. `validation_first_sample_forecast.html`
- 검증 데이터 첫 샘플에 대한 예측 결과
- 실제값 vs 예측값 비교

### 3. `overall_validation_performance.html`
- 전체 검증 기간 동안의 예측 성능
- 모든 검증 윈도우의 예측 오버레이

### 4. `future_1week_forecast.html`
- 과거 데이터 + 최근 검증 예측 + 향후 7일 예측
- 종합적인 시계열 예측 결과

### 예시 시각화

```
┌────────────────────────────────────────┐
│  실제값 (파란선)                        │
│  예측값 (빨간선)                        │
│                                        │
│      ╱╲    ╱╲                         │
│     ╱  ╲  ╱  ╲    예측 구간          │
│    ╱    ╲╱    ╲╱╲                    │
│   ╱              ╲                    │
│  ╱                ╲                   │
└────────────────────────────────────────┘
    과거 7일        향후 7일
```

## 📈 성능 지표

### 학습 정보 (`training_info.txt`)

실행 완료 후 다음 정보가 자동 저장됩니다:

```
=========== Training Results ===========
Final Training Loss: 0.012345
Final Validation Loss (MSE): 0.023456
Average MAE per Validation Window: 1.234567
Total Execution Time: 123.45 seconds
```

### 전형적인 성능
- **검증 MSE**: 0.02 ~ 0.05 (정규화된 스케일)
- **검증 MAE**: 실제 스케일에서 계산됨
- **실행 시간**: 약 2~5분 (CPU 기준)

## 📁 프로젝트 구조

```
gas-demand-forecast/
│
├── gas_demand_forecast.py    # 메인 스크립트
├── README.md                  # 프로젝트 문서
├── requirements.txt           # 의존성 패키지
│
├── data/                      # 데이터 디렉토리
│   └── CST_OP_COLLECTION.csv
│
├── visualization/             # 시각화 결과물
│   ├── after_preprocess_data.html
│   ├── validation_first_sample_forecast.html
│   ├── overall_validation_performance.html
│   └── future_1week_forecast.html
│
└── training_info.txt          # 학습 정보 로그
```

## 🔧 트러블슈팅

### CUDA 관련 오류
```bash
# CPU 모드로 강제 실행
export CUDA_VISIBLE_DEVICES=""
python gas_demand_forecast.py
```

### 메모리 부족
```python
# 배치 사이즈 줄이기
batch_size = 8  # 기본값: 16
```

### 데이터 부족 오류
```python
# 윈도우 크기 조정
INPUT_SEQ_DAYS = 3   # 기본값: 7
OUTPUT_SEQ_DAYS = 3  # 기본값: 7
```

## 🤝 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 👨‍💻 작성자

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 감사의 글

- PyTorch 커뮤니티
- Plotly 개발팀
- 오픈소스 기여자 여러분

---

⭐️ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!