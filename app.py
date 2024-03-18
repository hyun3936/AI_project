from datetime import datetime
import pytz
import json
import csv
from flask import Flask, request, jsonify
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import os
from Model import Autoencoder
from DataImport import savedata

app = Flask(__name__)

# 날짜 형식 조정 ---------------------------------------------------------------------------------------------------------------

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def convert_timestamp_to_date(timestamp):
    """밀리초 단위의 타임스탬프를 'Asia/Seoul' 시간대를 기준으로 'YYYY-MM-DD HH:MM:SS' 형태의 문자열로 변환합니다."""
    dt_local = datetime.fromtimestamp(int(timestamp) / 1000, tz=ZoneInfo('Asia/Seoul'))
    return dt_local.strftime('%Y-%m-%d %H:%M:%S')

# 결과 내보내기 ---------------------------------------------------------------------------------------------------------------

@app.route('/postLog', methods=['POST'])
def handle_post_log():
    
    # data 호출
    savedata(data)
 
# 데이터 전처리 및 모델 클래스 정의는 여기에 포함됩니다.
    
    heartbeat_data = savedata.df['heartbeat'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    heartbeat_data_scaled = scaler.fit_transform(heartbeat_data)
    
    # 시계열 데이터 변환 함수
    def create_dataset(data, time_steps=1):
        X = []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), 0])
        return np.array(X)
    time_steps = 3
    X = create_dataset(heartbeat_data_scaled, time_steps)
    
    # numpy 배열을 PyTorch 텐서로 변환 및 차원 조정
    X = torch.Tensor(X).unsqueeze(2)
    
    # DataLoader 생성
    batch_size = 16
    dataset = TensorDataset(X, X)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
# 모델 인스턴스 생성
    model = Autoencoder(input_dim=1, hidden_dim=50, num_layers=2) 
    criterion = torch.nn.MSELoss()  # 평균 제곱 오차 손실 함수

# 모델을 평가 모드로 설정하고, 재구성 오차를 계산하여 이상치 탐지 기준을 설정합니다.
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, inputs).item()
        reconstruction_errors.append(loss)

# 이상치 탐지를 위한 임계값을 설정합니다.
    mean_reconstruction_error = np.mean(reconstruction_errors)
    std_reconstruction_error = np.std(reconstruction_errors)
    threshold = mean_reconstruction_error + 2 * std_reconstruction_error

# 재구성 오차를 기반으로 이상치를 식별합니다.
    anomalies = [i for i, error in enumerate(reconstruction_errors) if error > threshold]
        
    # 이상치로 식별된 샘플 인덱스 (예시)
    anomalies = [0]  # 첫 번째 샘플이 이상치로 가정
    
    # 이상치 인덱스 보정 (time_steps 고려)
    time_steps = 10
    anomalies_corrected = [x + time_steps for x in anomalies]
    
    # 결과 DataFrame 초기화
    results = pd.DataFrame({
        'status': ['SAFE'] * len(savedata.df),  # 초기 상태를 'SAFE'로 설정
        'worker_id': savedata.df['usercode'].values  # 'UserCode'를 'worker_id'로 사용
    })
    
    # 이상치에 대한 상태를 'CAUTION'으로 업데이트
    for idx in anomalies_corrected:
        if idx < len(results):  # 인덱스가 결과 DataFrame의 범위 내에 있는지 확인
            results.at[idx, 'status'] = 'CAUTION'
    

    return jsonify(results)


# 끝 ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)