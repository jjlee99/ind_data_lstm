
폴더 및 현재 디렉토리 파일 설명
data: CST_OP_COLLECTION.csv,IOT_CTRL_THERMO_202506100935_3month.csv # 두가지 데이터를 포함함. (cst로 시작하는 계측기 수집 정보 데이터, thermo인 열화상 센서 데이터)
models : forecast.py # 데이터 전처리부터, 딥러닝 모델 구축, 시각화 까지 포함되어 있는 파일
old : analyze.ipynb, forecase_valid.ipynb etc # forecase 실행 파일을 개발하기 위한 분석 파일, thermo 분석 파일 등 포함
visualization : future_1week_forecast, overall_validation_performance, validation_first_sample_forecast.html # 3가지 시각화로, 측정 기간 이후 일주일 동안의 예측 시각화, 전체 검증 데이터의 성능 시각화, 첫째 슬라이딩 윈도우의 성능 시각화임.
그외 : 가상환경 구축방법, 권한 문제 해결 등 오류에 대한 사항들 메모장으로 기입해둠.

실증 시나리오는 lstm 모델 활용 시계열 분석을 통한 계측기 a상 전압 수요 예측