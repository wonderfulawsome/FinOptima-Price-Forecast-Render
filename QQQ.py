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
    
    # Prophet 모델 학습 데이터 준비 (이동평균 regressors 제거)
    train_df = df[['date','close','volume_ratio','rsi']].rename(columns={'date':'ds','close':'y'})
    train_df = train_df.fillna(method='ffill').fillna(method='bfill')
    
    # Prophet 모델 초기화 및 regressors 추가 (volume_ratio와 rsi만 사용)
    model = Prophet(
        changepoint_prior_scale=0.1,
        seasonality_mode='multiplicative',
        yearly_seasonality=5,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.add_regressor('volume_ratio', standardize=True)
    model.add_regressor('rsi', standardize=True)
    model.fit(train_df)
    
    # 미래 30일 예측 준비
    forecast_days = 30
    last_date = df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
    # forecast용 regressors: volume_ratio는 1.0, rsi는 마지막 값 그대로 사용
    last_rsi = df['rsi'].iloc[-1]
    future_df = pd.DataFrame({
        'ds': future_dates,
        'volume_ratio': [1.0] * forecast_days,
        'rsi': [last_rsi] * forecast_days
    })
    
    # Prophet 예측
    forecast_result = model.predict(future_df)
    forecasted = forecast_result[['ds','yhat','yhat_lower','yhat_upper']].set_index('ds')
    
    # 이동평균 계산을 위해 전체 가격 시리즈 생성 (과거 close + 예측 yhat)
    historical = df.set_index('date')['close']
    all_prices = pd.concat([historical, forecasted['yhat']])
    sma20_all = all_prices.rolling(20).mean()
    sma50_all = all_prices.rolling(50).mean()
    
    forecast_list = []
    for d in future_dates:
        forecast_list.append({
            "ds": d.strftime("%Y-%m-%d"),
            "yhat": forecasted.loc[d, 'yhat'],
            "yhat_lower": forecasted.loc[d, 'yhat_lower'],
            "yhat_upper": forecasted.loc[d, 'yhat_upper'],
            "sma20": sma20_all.loc[d] if not pd.isna(sma20_all.loc[d]) else None,
            "sma50": sma50_all.loc[d] if not pd.isna(sma50_all.loc[d]) else None,
            "volume_spike": 0,  # volume_ratio=1.0이므로 spike 없음
            "rsi": last_rsi
        })
    
    # 지지와 저항선 산출 (과거 데이터 기준)
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
        "forecastStart": last_date.strftime("%Y-%m-%d"),
        "volumeSpikes": volume_spike_dates
    })

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
