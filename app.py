from datetime import datetime
import pytz
import json
import csv
from flask import Flask, request, jsonify
import pandas as pd
import requests

app = Flask(__name__)

# 날짜 형식 조정 ---------------------------------------------------------------------------------------------------------------

def convert_timestamp_to_date(timestamp):
    """밀리초 단위의 타임스탬프를 특정 시간대를 기준으로 'YYYY-MM-DD HH:MM:SS' 형태의 문자열로 변환합니다."""
    # 서울 시간대를 지정
    timezone = pytz.timezone('Asia/Seoul')
    # UTC 기준의 datetime 객체를 생성
    dt_utc = datetime.utcfromtimestamp(int(timestamp) / 1000.0)
    # 서울 시간대로 변환
    dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(timezone)
    # 지정된 형식의 문자열로 변환
    return dt_local.strftime('%Y-%m-%d %H:%M:%S')

# 데이터 받기 ---------------------------------------------------------------------------------------------------------------

@app.route('/postLog', methods=['POST'])
def handle_post_log():
    data = request.json
    
     # 'registerDate' 필드가 있고 타임스탬프 형식인 경우, 날짜 형식으로 변환합니다.
    if isinstance(data, dict):  # 단일 객체 처리
        if 'registerDate' in data:
            data['registerDate'] = convert_timestamp_to_date(data['registerDate'])
    elif isinstance(data, list):  # 리스트 내의 각 객체 처리
        for item in data:
            if 'registerDate' in item:
                item['registerDate'] = convert_timestamp_to_date(item['registerDate'])   
    
    
    file_name = "dataSaveTest5.csv"
    fieldnames = ["id","usercode", "latitude", "longitude", "heartRate", "temperature", "outTemp", "registerDate"]
    
    try:
        with open(file_name, 'x', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    except FileExistsError:
        pass # 파일이 이미 존재하면 아무것도 하지 않음
    
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 만약 data가 리스트라면, 각 항목을 순회하며 씁니다.
        if isinstance(data, list):
            for item in data:
                writer.writerow(item)
        else:
            writer.writerow(data)
    
    return jsonify({"message": "데이터를 성공적으로 받았습니다."}), 200

# 결과 내보내기 ---------------------------------------------------------------------------------------------------------------

@app.route('/sendData', methods=['GET'])
def send_data():
    # 모델로부터 결과 값을 생성합니다. 여기서는 예시 데이터를 사용합니다.
   
    data = pd.read_csv("./data/user_code_total/dataSaveTest4.csv")
    
    df = pd.DataFrame(data)
    
    # 이상치로 식별된 샘플 인덱스 (예시)
    anomalies = [0]  # 첫 번째 샘플이 이상치로 가정
    
    # 이상치 인덱스 보정 (time_steps 고려)
    time_steps = 3
    anomalies_corrected = [x + time_steps for x in anomalies]
    
    # 결과 DataFrame 초기화
    results = pd.DataFrame({
        'status': ['SAFE'] * len(df),  # 초기 상태를 'SAFE'로 설정
        'worker_id': df['UserCode'].values  # 'UserCode'를 'worker_id'로 사용
    })
    
    # 이상치에 대한 상태를 'CAUTION'으로 업데이트
    for idx in anomalies_corrected:
        if idx < len(results):  # 인덱스가 결과 DataFrame의 범위 내에 있는지 확인
            results.at[idx, 'status'] = 'CAUTION'
    
    # 스프링 부트 서버의 주소와 엔드포인트
    url = 'http://localhost:8080/receiveData'
        
    # 스프링 부트 서버에 POST 요청을 보냅니다.
    response = requests.post(url, json=results.to_dict(orient='records'))
    
    # 스프링 부트 서버로부터의 응답을 반환합니다.
    return jsonify(response.json()), response.status_code



# 끝 ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
