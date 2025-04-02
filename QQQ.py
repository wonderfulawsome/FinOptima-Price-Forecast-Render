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
CORS(app, supports_credentials=True)

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

def identify_support_resistance(df, price_column='close', window=10):
    """이동평균선과 피벗 포인트 기반 지지/저항 영역 식별"""
    df = df.copy()
    
    # 최근 고점/저점 식별
    df['pivot_high'] = df[price_column].rolling(window=window, center=True).apply(
        lambda x: 1 if x.iloc[window//2] == max(x) else 0, raw=False)
    df['pivot_low'] = df[price_column].rolling(window=window, center=True).apply(
        lambda x: 1 if x.iloc[window//2] == min(x) else 0, raw=False)
    
    # 최근 N개의 피벗 고점/저점 선택
    n_pivots = 5
    resistance_levels = df[df['pivot_high'] == 1].tail(n_pivots)[price_column].tolist()
    support_levels = df[df['pivot_low'] == 1].tail(n_pivots)[price_column].tolist()
    
    # 주요 이동평균선도 지지/저항선으로 고려
    last_ma_values = [df['sma20'].iloc[-1], df['sma50'].iloc[-1], df['sma200'].iloc[-1]]
    
    # 유효한 지지/저항선 식별 (현재 가격으로부터 ±15% 이내)
    current_price = df[price_column].iloc[-1]
    price_range = current_price * 0.15
    
    valid_resistance = [level for level in resistance_levels if level > current_price and 
                       level < current_price + price_range]
    valid_support = [level for level in support_levels if level < current_price and 
                    level > current_price - price_range]
    
    # 이동평균선 추가
    for ma in last_ma_values:
        if ma > current_price and ma < current_price + price_range:
            valid_resistance.append(ma)
        elif ma < current_price and ma > current_price - price_range:
            valid_support.append(ma)
    
    # 중복 제거 및 정렬
    valid_resistance = sorted(list(set(valid_resistance)))
    valid_support = sorted(list(set(valid_support)))
    
    return valid_support, valid_resistance

# 예측 API 엔드포인트
@app.route('/api/predict', methods=['GET'])
def predict_api():
    """주식 예측 API 엔드포인트"""
    ticker = request.args.get('ticker', 'QQQ')
    days = int(request.args.get('days', 200))
    future_days = int(request.args.get('future_days', 30))
    
    try:
        result = predict_stock(ticker, days, future_days)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_stock(ticker, days=200, future_days=30):
    """주식 예측 함수"""
    # 데이터 준비
    stock_data = get_stock_data(ticker, days)
    stock_data = add_technical_indicators(stock_data)
    
    # 지지/저항선 계산
    support_levels, resistance_levels = identify_support_resistance(stock_data)
    
    # 학습 데이터 준비 - NaN 값 확인 및 제거
    train_df = stock_data[['date', 'close', 'sma20', 'sma50', 'volume_ratio', 'rsi']].rename(
        columns={'date':'ds', 'close':'y'})
    
    # NaN 값 제거
    train_df = train_df.fillna(method='ffill').fillna(method='bfill')
    
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
    
    # 모델 학습
    model.fit(train_df)
    
    # 미래 데이터 생성
    future = model.make_future_dataframe(periods=future_days)
    
    # 미래 회귀변수 값 설정
    last_date = stock_data['date'].max()
    last_sma20 = stock_data.loc[stock_data['date'] == last_date, 'sma20'].values[0]
    last_sma50 = stock_data.loc[stock_data['date'] == last_date, 'sma50'].values[0]
    last_volume_ratio = stock_data.loc[stock_data['date'] == last_date, 'volume_ratio'].values[0]
    last_rsi = stock_data.loc[stock_data['date'] == last_date, 'rsi'].values[0]
    
    # 과거 값 복사
    past_dates = stock_data['date']
    for col in ['sma20', 'sma50', 'volume_ratio', 'rsi']:
        # 과거 값은 실제 데이터에서 가져오기
        for date, val in zip(past_dates, stock_data[col]):
            future.loc[future['ds'] == date, col] = val
    
    # 미래 값은 시뮬레이션
    future_dates = future[future['ds'] > last_date]['ds']
    
    # 미래 값 시뮬레이션
    for i, date in enumerate(future_dates):
        # 이동평균선 시뮬레이션 (완만한 변화)
        future.loc[future['ds'] == date, 'sma20'] = last_sma20 * (1 + np.random.normal(0, 0.005) * (i+1))
        future.loc[future['ds'] == date, 'sma50'] = last_sma50 * (1 + np.random.normal(0, 0.003) * (i+1))
        
        # RSI 시뮬레이션 (평균 회귀 특성)
        if i == 0:
            future.loc[future['ds'] == date, 'rsi'] = last_rsi
        else:
            prev_rsi = future.loc[future['ds'] == future_dates.iloc[i-1], 'rsi'].values[0]
            new_rsi = prev_rsi + np.random.normal(0, 3)
            if new_rsi > 70: new_rsi -= np.random.uniform(2, 5)  # 과매수 시 조정
            if new_rsi < 30: new_rsi += np.random.uniform(2, 5)  # 과매도 시 조정
            future.loc[future['ds'] == date, 'rsi'] = max(0, min(100, new_rsi))
        
        # 거래량 시뮬레이션
        future.loc[future['ds'] == date, 'volume_ratio'] = max(0.5, np.random.normal(1, 0.2))
    
    # 결측치 최종 처리
    future = future.fillna(method='ffill')
    
    # 예측 수행
    forecast = model.predict(future)
    
    # 예측 결과 변동성 추가
    for i in range(1, len(forecast)):
        if forecast['ds'].iloc[i] <= last_date:
            continue  # 과거 데이터는 수정하지 않음
        
        # 기본 변동성
        base_volatility = np.random.normal(0, 0.01)
        
        # RSI 영향
        rsi = future.loc[future['ds'] == forecast['ds'].iloc[i], 'rsi'].values[0]
        if rsi > 70:  # 과매수
            rsi_effect = np.random.uniform(-0.02, -0.005)
        elif rsi < 30:  # 과매도
            rsi_effect = np.random.uniform(0.005, 0.02)
        else:
            rsi_effect = 0
        
        # 거래량 영향
        volume_effect = 0
        volume_ratio = future.loc[future['ds'] == forecast['ds'].iloc[i], 'volume_ratio'].values[0]
        if volume_ratio > 1.5:  # 거래량 스파이크
            volume_effect = np.random.uniform(-0.02, 0.02)
        
        # 이동평균선 영향
        sma_effect = 0
        price = forecast.loc[i, 'yhat']
        sma20 = future.loc[future['ds'] == forecast['ds'].iloc[i], 'sma20'].values[0]
        sma50 = future.loc[future['ds'] == forecast['ds'].iloc[i], 'sma50'].values[0]
        
        if price < sma20 * 0.98:  # 지지선 효과
            sma_effect = np.random.uniform(0.005, 0.015)
        elif price > sma20 * 1.02:  # 저항선 효과
            sma_effect = np.random.uniform(-0.015, -0.005)
        
        # 총 효과 적용
        total_effect = base_volatility + rsi_effect + volume_effect + sma_effect
        forecast.loc[i, 'yhat'] *= (1 + total_effect)
        
        # 신뢰 구간 조정
        forecast.loc[i, 'yhat_lower'] = forecast.loc[i, 'yhat'] * 0.95
        forecast.loc[i, 'yhat_upper'] = forecast.loc[i, 'yhat'] * 1.05

    # 미래 예측 요약
    final_price = forecast[forecast['ds'] > last_date]['yhat'].iloc[-1]
    current_price = stock_data['close'].iloc[-1]
    expected_return = ((final_price / current_price) - 1) * 100
    
    # RSI 상태 분석
    current_rsi = stock_data['rsi'].iloc[-1]
    rsi_status = "과매수" if current_rsi > 70 else "과매도" if current_rsi < 30 else "중립"
    
    # 이동평균선 상태 분석
    ma_status = "상승세" if (current_price > stock_data['sma20'].iloc[-1] and 
                          stock_data['sma20'].iloc[-1] > stock_data['sma50'].iloc[-1]) else "하락세"
    
    # 거래량 분석
    recent_vol_avg = stock_data['volume_ratio'].iloc[-5:].mean()
    vol_status = "증가 추세" if recent_vol_avg > 1.2 else "감소 추세" if recent_vol_avg < 0.8 else "보통"
    
    # 투자 제안
    if expected_return > 5:
        recommendation = "강력 매수"
    elif expected_return > 2:
        recommendation = "매수"
    elif expected_return > -2:
        recommendation = "중립"
    elif expected_return > -5:
        recommendation = "매도"
    else:
        recommendation = "강력 매도"

    # 결과 데이터 생성
    result = {
        'ticker': ticker,
        'historical_data': stock_data.to_dict(orient='records'),
        'forecast_data': forecast.to_dict(orient='records'),
        'last_date': last_date.strftime('%Y-%m-%d'),
        'current_price': float(current_price),
        'final_price': float(final_price),
        'expected_return': float(expected_return),
        'rsi': {
            'current': float(current_rsi),
            'status': rsi_status
        },
        'trend': ma_status,
        'volume': {
            'status': vol_status,
            'recent_avg': float(recent_vol_avg)
        },
        'recommendation': recommendation,
        'support_levels': support_levels,
        'resistance_levels': resistance_levels
    }
    
    return result

# forecast 엔드포인트
@app.route('/forecast', methods=['GET'])
def forecast():
    """기술적 지표 기반 주가 예측 엔드포인트"""
    ticker = request.args.get('ticker', 'QQQ')
    days = int(request.args.get('days', 200))
    future_days = int(request.args.get('future_days', 30))
    
    try:
        result = predict_stock(ticker, days, future_days)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 건강 체크 엔드포인트
@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

# 기본 경로 (루트 경로)
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'app': 'Stock Technical Analysis API',
        'version': '1.0',
        'endpoints': [
            {'path': '/forecast', 'method': 'GET', 'description': '주가 예측 API'},
            {'path': '/api/predict', 'method': 'GET', 'description': '주가 예측 API (별칭)'},
            {'path': '/healthcheck', 'method': 'GET', 'description': '서버 상태 확인'}
        ]
    })

# OPTIONS 요청 처리 - 모든 경로에 대해
@app.route('/forecast', methods=['OPTIONS'])
@app.route('/api/predict', methods=['OPTIONS'])
def handle_options():
    return '', 204  # 204 No Content

# 모든 응답에 CORS 헤더 추가
@app.after_request
def add_cors_headers(response):
    # 허용할 오리진 - 모든 도메인 허용
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    # 허용할 헤더 - 필요한 모든 헤더 포함
    response.headers.add('Access-Control-Allow-Headers', 
                       'Content-Type, Authorization, X-Requested-With, Accept, Origin, Cache-Control, Pragma')
    
    # 허용할 메서드
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    
    # preflight 요청 캐싱 (24시간)
    response.headers.add('Access-Control-Max-Age', '86400')
    
    # 인증 정보 포함 허용
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    
    # 브라우저 캐싱 최적화를 위한 Vary 헤더
    response.headers.add('Vary', 'Origin')
    
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
