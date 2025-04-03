import os
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify
from flask_cors import CORS
from prophet import Prophet
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# 랜덤 고정
np.random.seed(42)

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("FMP_API", "DEMO_KEY")

def get_stock_data(ticker, days=200):
    try:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={API_KEY}"
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data['historical'][:days])
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
        close = np.random.normal(500, 50, days).cumsum() + 400
        volume = np.random.randint(2e7, 5e7, days)
        df = pd.DataFrame({'date': dates, 'close': close, 'volume': volume})
        return df.sort_values('date')

def add_technical_indicators(df):
    df = df.copy()
    df['sma20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['sma50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['sma200'] = SMAIndicator(close=df['close'], window=5).sma_indicator()
    df['volume_sma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma20']
    df['volume_spike'] = np.where(df['volume_ratio'] > 1.5, 1, 0)
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df = df.fillna(method='ffill').fillna(method='bfill').reset_index(drop=True)
    return df

@app.route("/forecast/<ticker>", methods=["GET"])
def forecast(ticker):
    # 실제 데이터와 기술적 지표 생성
    df = get_stock_data(ticker, 200)
    df = add_technical_indicators(df)
    
    # Prophet 모델 학습 데이터 준비
    train_df = df[['date','close','sma20','sma50','volume_ratio','rsi']].rename(columns={'date':'ds','close':'y'})
    train_df = train_df.fillna(method='ffill').fillna(method='bfill')
    
    # Prophet 모델 초기화 및 회귀변수 추가
    model = Prophet(
        changepoint_prior_scale=0.1,
        seasonality_mode='multiplicative',
        yearly_seasonality=5,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.add_regressor('sma20', standardize=False)
    model.add_regressor('sma50', standardize=False)
    model.add_regressor('volume_ratio', standardize=True)
    model.add_regressor('rsi', standardize=True)
    model.fit(train_df)
    
    # 순차적으로 미래 30일 예측 (이전 예측 주가를 포함하여 이동평균 재계산)
    forecast_days = 30
    forecast_list = []
    current_data = df.copy()  # 실제 데이터
    last_date = current_data['date'].max()
    # 여기서는 volume_ratio는 1.0, rsi는 마지막 값을 그대로 사용 (필요시 더 세밀한 시뮬레이션 가능)
    last_rsi = current_data['rsi'].iloc[-1]
    
    for i in range(forecast_days):
        next_date = last_date + pd.Timedelta(days=1)
        # 최근 20일, 50일의 종가를 사용해 이동평균 계산
        if len(current_data) >= 20:
            next_sma20 = current_data['close'].iloc[-20:].mean()
        else:
            next_sma20 = current_data['close'].mean()
        if len(current_data) >= 50:
            next_sma50 = current_data['close'].iloc[-50:].mean()
        else:
            next_sma50 = current_data['close'].mean()
        next_volume_ratio = 1.0
        next_rsi = last_rsi
        
        # 다음 날 예측을 위한 회귀변수 DataFrame 구성
        next_reg = pd.DataFrame({
            'ds': [next_date],
            'sma20': [next_sma20],
            'sma50': [next_sma50],
            'volume_ratio': [next_volume_ratio],
            'rsi': [next_rsi]
        })
        
        # 하루 단위 예측
        pred = model.predict(next_reg)
        next_yhat = pred['yhat'].iloc[0]
        
        forecast_list.append({
            "ds": next_date.strftime("%Y-%m-%d"),
            "yhat": next_yhat,
            "yhat_lower": pred['yhat_lower'].iloc[0],
            "yhat_upper": pred['yhat_upper'].iloc[0],
            "sma20": next_sma20,
            "sma50": next_sma50,
            "volume_spike": 1 if next_volume_ratio > 1.5 else 0,
            "rsi": next_rsi
        })
        
        # 예측된 종가를 current_data에 추가해 다음 이동평균 계산에 반영
        new_row = {
            'date': next_date,
            'close': next_yhat,
            'sma20': next_sma20,
            'sma50': next_sma50,
            'sma200': current_data['close'].iloc[-5:].mean() if len(current_data) >= 5 else current_data['close'].mean(),
            'volume_spike': 0,
            'rsi': next_rsi
        }
        current_data = current_data.append(new_row, ignore_index=True)
        last_date = next_date
    
    # 지지와 저항선 산출 (기존 방식 유지)
    current_price = df['close'].iloc[-1]
    sma_now = [df['sma20'].iloc[-1], df['sma50'].iloc[-1], df['sma200'].iloc[-1]]
    support_list = [x for x in sma_now if x < current_price]
    resist_list  = [x for x in sma_now if x > current_price]
    window = 10
    for i in range(window, len(df)-window):
        if df['close'].iloc[i] == max(df['close'].iloc[i-window:i+window]):
            if df['close'].iloc[i] > current_price:
                resist_list.append(df['close'].iloc[i])
        if df['close'].iloc[i] == min(df['close'].iloc[i-window:i+window]):
            if df['close'].iloc[i] < current_price:
                support_list.append(df['close'].iloc[i])
    volume_spike_dates = df.loc[df['volume_spike']==1,'date'].dt.strftime("%Y-%m-%d").tolist()
    
    real_data = []
    for _, row in df.iterrows():
        real_data.append({
            "ds": row['date'].strftime("%Y-%m-%d"),
            "close": row['close'],
            "sma20": row['sma20'],
            "sma50": row['sma50'],
            "sma200": row['sma200'],
            "volume_spike": int(row['volume_spike']),
            "rsi": row['rsi']
        })
    
    return jsonify({
        "real": real_data,
        "predicted": forecast_list,
        "support": sorted(support_list),
        "resistance": sorted(resist_list),
        "forecastStart": df['date'].max().strftime("%Y-%m-%d"),
        "volumeSpikes": volume_spike_dates
    })

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
