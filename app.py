# 실제로 데이터를 분석하는 곳
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리
from flask import Flask, request, jsonify
from datetime import datetime, timezone
from zoneinfo import ZoneInfo




app = Flask(__name__)


# 날짜 형식 조정 ---------------------------------------------------------------------------------------------------------------

def convert_timestamp_to_date(timestamp):
    """밀리초 단위의 타임스탬프를 'Asia/Seoul' 시간대를 기준으로 'YYYY-MM-DD HH:MM:SS' 형태의 문자열로 변환합니다."""
    dt_local = datetime.fromtimestamp(int(timestamp) / 1000, tz=ZoneInfo('Asia/Seoul'))
    return dt_local.strftime('%Y-%m-%d %H:%M:%S')

# 결과 내보내기 ---------------------------------------------------------------------------------------------------------------

@app.route('/postLog/<usercode>', methods=['POST'])
def handle_post_log(usercode):

    # 가정: 'real.csv' 파일에 분석할 실제 데이터가 있으며,
    # 여기서는 이미 준비된 10개의 데이터를 사용한다고 가정합니다.

    # 데이터 불러오기 및 전처리 (10개 데이터 예시)
    # df = pd.read_csv("./data/user_code_total/real.csv")

    # 'Content-Type'을 검증하지 않고 요청 데이터를 JSON으로 처리
    data = request.get_json(force=True)
      
    df = pd.DataFrame(data)

    
    # 'RegisterDate' 열에 있는 타임스탬프를 지정된 형식의 문자열로 변환
    # df['RegisterDate'] = df['RegisterDate'].apply(convert_timestamp_to_date)

    df['regidate'] = pd.to_datetime(df['regidate'])
    df.sort_values(by='regidate', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    # # 'registerDate' 컬럼이 있다면, 해당 컬럼의 타임스탬프를 날짜 형식으로 변환
    # if 'registerDate' in df.columns:
    #     df['registerDate'] = df['registerDate'].apply(lambda x: convert_timestamp_to_date(x))
    
    
    
    df = df.tail(10)  # 최근 10개 데이터만 사용
    

    
    #  파일 이름 정의
    file_name = f"dataSaveTest_{usercode}.csv"  # 예를 들어 {usercode}, REST API로부터 받은 값

    # 파일 저장
    try:
        with open(file_name, 'x') as f:
            # 파일이 새로 생성되었으므로, 헤더와 함께 데이터를 저장
            df.to_csv(file_name, index=False, mode='w', header=True)
    except FileExistsError:
        # 파일이 이미 존재하면, 헤더 없이 데이터를 추가
        df.to_csv(file_name, index=False, mode='a', header=False)

    
    # 파일을 불러옵니다.
    df = pd.read_csv(file_name)


    # 심박수 데이터 추출 및 스케일링
    heartbeat_data = df['heartbeat'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    heartbeat_data_scaled = scaler.fit_transform(heartbeat_data)

    # PyTorch 텐서로 변환 및 차원 조정
    X = torch.Tensor(heartbeat_data_scaled).unsqueeze(2)  # [1, 10, 1] 형태로 변환

    # LSTM 기반 오토인코더 모델 정의 (변경 없음)
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super(Autoencoder, self).__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
            self.linear = nn.Linear(input_dim, input_dim)

        def forward(self, x):
            _, (hn, _) = self.encoder(x)
            hn = hn[-1].unsqueeze(0)
            repeated_hn = hn.repeat(1, x.size(1), 1)
            decoded, _ = self.decoder(repeated_hn)
            decoded = decoded.contiguous().view(-1, x.size(2))
            decoded = self.linear(decoded)
            return decoded.view(x.size(0), x.size(1), -1)

    # 모델 불러오기 (가정: 학습된 모델이 'best_model.pth'에 저장되어 있음)
    model_path = f"C:/Users/user/KeepMeSafe/AI_Project/trainedModel/best_model_user_code_{usercode}.pth"
    model = Autoencoder(input_dim=1, hidden_dim=50, num_layers=2)
    # map_location을 사용하여 CPU에서 모델 상태 사전을 로드
    model_state = torch.load(model_path, map_location=torch.device('cpu'))

    # 모델 상태 사전을 사용하여 모델에 가중치 로드
    model.load_state_dict(model_state)

    
    # model.load_state_dict(torch.load(model_path))
    model.eval()
    
    


    # 데이터에 대한 예측 수행 및 재구성 오차 계산
    criterion = torch.nn.MSELoss()  # 평균 제곱 오차 손실 함수
    with torch.no_grad():
        outputs = model(X)
        reconstruction_errors = criterion(outputs, X).item()

    print(f'재구성 오차: {reconstruction_errors}')

    # 재구성 오차를 기반으로 필요한 분석 수행
    # 예를 들어, 특정 임계값을 기반으로 이상치 판단 등

    # 재구성 오차 계산
    criterion = torch.nn.MSELoss()
    reconstruction_errors = []

    with torch.no_grad():
        outputs = model(X)
        for i in range(X.size(0)):
            loss = criterion(outputs[i], X[i]).item()
            reconstruction_errors.append(loss)

    # 임계값 설정 (이전에 정의한 방식을 그대로 사용)
    mean_reconstruction_error = np.mean(reconstruction_errors)
    std_reconstruction_error = np.std(reconstruction_errors)
    threshold = mean_reconstruction_error + 3 * std_reconstruction_error

    # # 재구성 오차 시각화
    # plt.hist(reconstruction_errors, bins=50, alpha=0.7, color='blue', label='Reconstruction errors')
    # plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label='Threshold')
    # plt.title('Histogram of Reconstruction Errors')
    # plt.xlabel('Reconstruction error')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.show()

    # # 재구성 오차를 기반으로 이상치를 식별합니다.
    # anomalies = [i for i, error in enumerate(reconstruction_errors) if error > threshold]

    # # 결과 DataFrame 초기화
    # results = pd.DataFrame({
    #     'status': [1] * len(df),  # 초기 상태를 'SAFE'로 설정
    #     'worker_id': df['usercode'].values  # 'UserCode'를 'worker_id'로 사용
    # })

    
    # SAFE : 1
    # CAUTION : 2


    # # 이상치에 대한 상태를 'CAUTION'으로 업데이트
    # for idx in anomalies:
    #     if idx < len(results):  # 인덱스가 결과 DataFrame의 범위 내에 있는지 확인
    #         results.at[idx, 'status'] = 2
            
    # # print(results)


    # # 'status' 열에서 'CAUTION'이 하나라도 있는지 검사
    # if 2 in results['status'].values:
    #     overall_status = 2
    # else:
    #     overall_status = 1

    # # JSON 형식으로 결과 반환
    # result = {
    #     'status': overall_status,
    #     'worker_id': int(df['usercode'].values[0])  # 리스트로 감싸 단일 값으로 설정
    # }

    # # print(result)

    # return jsonify(result)
    
    
        # 두번째------------------------------------------------------------------------------------------------
    
    # # 상태 값에 대한 명시적 정의
    # SAFE = 1
    # CAUTION = 2

    # # 이상치를 식별합니다.
    # anomalies = [i for i, error in enumerate(reconstruction_errors) if error > threshold]

    # # 결과 DataFrame 초기화
    # results = pd.DataFrame({
    #     'status': SAFE,  # 초기 상태를 'SAFE'로 설정
    #     'worker_id': df['usercode']  # 'UserCode'를 'worker_id'로 사용
    # })

    # # 이상치에 대한 상태를 'CAUTION'으로 업데이트
    # for idx in anomalies:
    #     # if idx < len(results):  # 인덱스가 결과 DataFrame의 범위 내에 있는지 확인
    #     results.at[idx, 'status'] = CAUTION

    # # 'status' 열에서 CAUTION(2)이 하나라도 있는지 검사
    # overall_status = CAUTION if CAUTION in results['status'].values else SAFE

    # # JSON 형식으로 결과 반환
    # result = {
    #     'status': overall_status,
    #     'worker_id': int(df['usercode'].iloc[-1])  # 마지막 worker_id 반환, 모든 worker_id를 반환하는 것이 의도된 경우 수정 필요
    # }

    # # 결과 출력 (Flask 등의 웹 프레임워크 사용 시)
    # return jsonify(result)


        # 세번째 --------------------------------------------------------------------------------------------------------------------------

    # # 상태 값에 대한 명시적 정의
    # SAFE = 1
    # CAUTION = 2

    # # 이상치를 식별합니다.
    # anomalies = [i for i, error in enumerate(reconstruction_errors) if error > threshold]

    # # 결과 DataFrame 초기화
    # results = pd.DataFrame({
    #     'status': [SAFE] * len(df),  # 모든 초기 상태를 'SAFE'로 설정
    #     'worker_id': df['usercode']  # 'UserCode'를 'worker_id'로 사용
    # })

    # # 이상치에 대한 상태를 'CAUTION'으로 업데이트
    # for idx in anomalies:
    #     results.at[idx, 'status'] = CAUTION

    # # 'status' 열에서 CAUTION(2)이 3개 이상 있는지 검사
    # caution_count = sum(results['status'] == CAUTION)

    # if caution_count > 9:
    #     overall_status = CAUTION
    # else:
    #     overall_status = SAFE

    # # JSON 형식으로 결과 반환
    # result = {
    #     'status': overall_status,
    #     'worker_id': int(df['usercode'].iloc[-1])  # 마지막 worker_id 반환
    # }

    # # 결과 출력 (Flask 등의 웹 프레임워크 사용 시)
    # print(f"result : {result}")
    # print(f"results : {results}")
    # return jsonify(result)
    
    

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)