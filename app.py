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
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


app = Flask(__name__)

# 날짜 형식 조정 ---------------------------------------------------------------------------------------------------------------

def convert_timestamp_to_date(timestamp):
    """밀리초 단위의 타임스탬프를 'Asia/Seoul' 시간대를 기준으로 'YYYY-MM-DD HH:MM:SS' 형태의 문자열로 변환합니다."""
    dt_local = datetime.fromtimestamp(int(timestamp) / 1000, tz=ZoneInfo('Asia/Seoul'))
    return dt_local.strftime('%Y-%m-%d %H:%M:%S')

# 결과 내보내기 ---------------------------------------------------------------------------------------------------------------

@app.route('/postLog', methods=['POST'])
def handle_post_log():
    


# 끝 ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)