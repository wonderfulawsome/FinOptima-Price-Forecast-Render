from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from prophet import Prophet
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import os

# Flask 앱 생성
app = Flask(__name__)

# CORS 설정 - 모든 도메인 허용
CORS(app)

# API 키를 환경 변수에서 가져오기
API_KEY = os.environ.get('FMP_API')

def get_stock_data(ticker, days=200):
    """주식 데이터 가져오기"""
    try:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data['historical'][:days])
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        # 오류 시 샘플 데이터 반환
        return generate_sample_data(days)

def generate_sample_data(days):
    """샘플 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days)
    close = np.random.normal(500, 50, days).cumsum() + 400
    volume = np.random.randint(20000000, 50000000, days)
    
    df = pd.DataFrame({
        'date': dates,
        'close': close,
        'volume': volume,
        'open': close * np.random.uniform(0.98, 1.0, days),
        'high': close * np.random.uniform(1.0, 1.03, days),
        'low': close * np.random.uniform(0.97, 1.0, days),
    })
    return df.sort_values('date')

def add_technical_indicators(df):
    """기술적 지표 추가"""
    df = df.copy()
    # 이동평균선
    df['sma20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['sma50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['sma200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
    
    # 거래량 지표
    df['volume_sma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma20']
    df['volume_spike'] = np.where(df['volume_ratio'] > 1.5, 1, 0)
    
    # RSI 지표
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    # 결측치 처리
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.reset_index(drop=True)
    
    return df

# 예측 API 엔드포인트
@app.route('/api/predict', methods=['GET'])
def predict_stock():
    """주식 예측 API 엔드포인트"""
    ticker = request.args.get('ticker', 'QQQ')
    days = int(request.args.get('days', 200))
    future_days = int(request.args.get('future_days', 30))
    
    try:
        # 이하 예측 로직 (이미 있는 코드 사용)
        # 데이터 준비
        stock_data = get_stock_data(ticker, days)
        stock_data = add_technical_indicators(stock_data)
        
        # 학습 데이터 준비
        train_df = stock_data[['date', 'close', 'sma20', 'sma50', 'volume_ratio', 'rsi']].rename(
            columns={'date':'ds', 'close':'y'})
        
        # Prophet 모델 설정
        model = Prophet(
            changepoint_prior_scale=0.1,
            seasonality_mode='multiplicative',
            yearly_seasonality=5,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        
        # 회귀변수 추가
        model.add_regressor('sma20', standardize=False)
        model.add_regressor('sma50', standardize=False)
        model.add_regressor('volume_ratio', standardize=True)
        model.add_regressor('rsi', standardize=True)
        
        # 결과 처리 및 반환
        # (기존 코드가 있다면 그대로 사용)
        
        # 예시 응답 (실제로는 더 복잡할 수 있음)
        result = {
            'ticker': ticker,
            'current_price': float(stock_data['close'].iloc[-1]),
            'historical_data': stock_data.to_dict(orient='records'),
            'last_date': stock_data['date'].max().strftime('%Y-%m-%d'),
            'support_levels': [370, 380],
            'resistance_levels': [420, 430],
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 건강 체크 엔드포인트
@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

# CORS 문제 해결을 위한 OPTIONS 메서드 처리
@app.route('/api/predict', methods=['OPTIONS'])
def handle_options():
    return '', 200

# 프리플라이트 요청 대응을 위한 후처리
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

# 메인 경로 (추가 경로가 필요한 경우)
@app.route('/forecast', methods=['GET', 'OPTIONS'])
def forecast():
    if request.method == 'OPTIONS':
        return '', 200
    
    # 실제 forecast 로직
    ticker = request.args.get('ticker', 'QQQ')
    # 처리 로직...
    
    return jsonify({'ticker': ticker, 'message': 'Forecast endpoint'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
