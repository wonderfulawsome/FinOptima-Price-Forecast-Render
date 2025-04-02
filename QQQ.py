# main.py
import os
import io
import base64
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

app = Flask(__name__)
CORS(app)  # CORS 허용

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
        # API 불가 시 샘플 데이터
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
        close = np.random.normal(500, 50, days).cumsum() + 400
        volume = np.random.randint(2e7, 5e7, days)
        df = pd.DataFrame({'date': dates, 'close': close, 'volume': volume})
        return df.sort_values('date')

def add_technical_indicators(df):
    df = df.copy()
    df['sma20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['sma50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['sma200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
    df['volume_sma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma20']
    df['volume_spike'] = np.where(df['volume_ratio'] > 1.5, 1, 0)
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df = df.fillna(method='ffill').fillna(method='bfill').reset_index(drop=True)
    return df

@app.route("/forecast/<ticker>", methods=["GET"])
def forecast(ticker):
    # 데이터 준비
    df = get_stock_data(ticker, 200)
    df = add_technical_indicators(df)

    # Prophet 학습 준비
    train_df = df[['date','close','sma20','sma50','volume_ratio','rsi']].rename(columns={'date':'ds','close':'y'})
    train_df = train_df.fillna(method='ffill').fillna(method='bfill')
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

    # 모델 학습
    model.fit(train_df)

    # 미래 데이터프레임
    future = model.make_future_dataframe(periods=30)
    last_date = df['date'].max()
    last_sma20 = df.loc[df['date'] == last_date, 'sma20'].values[0]
    last_sma50 = df.loc[df['date'] == last_date, 'sma50'].values[0]
    last_rsi   = df.loc[df['date'] == last_date, 'rsi'].values[0]

    # 과거 값 복사
    for col in ['sma20','sma50','volume_ratio','rsi']:
        for d, v in zip(df['date'], df[col]):
            future.loc[future['ds'] == d, col] = v

    # 미래 값 시뮬레이션
    f_dates = future[future['ds'] > last_date]['ds']
    for i, d in enumerate(f_dates):
        # 이동평균
        future.loc[future['ds'] == d, 'sma20'] = last_sma20 * (1 + np.random.normal(0, 0.005)*(i+1))
        future.loc[future['ds'] == d, 'sma50'] = last_sma50 * (1 + np.random.normal(0, 0.003)*(i+1))
        # RSI
        if i == 0:
            future.loc[future['ds'] == d, 'rsi'] = last_rsi
        else:
            prev_rsi = future.loc[future['ds'] == f_dates.iloc[i-1], 'rsi'].values[0]
            new_rsi = prev_rsi + np.random.normal(0, 3)
            if new_rsi > 70: new_rsi -= np.random.uniform(2, 5)
            if new_rsi < 30: new_rsi += np.random.uniform(2, 5)
            future.loc[future['ds'] == d, 'rsi'] = max(0, min(100, new_rsi))
        # 거래량
        future.loc[future['ds'] == d, 'volume_ratio'] = max(0.5, np.random.normal(1, 0.2))

    future = future.fillna(method='ffill')
    forecast = model.predict(future)

    # 예측값 후처리 (변동성 반영)
    for i in range(1, len(forecast)):
        if forecast['ds'].iloc[i] <= last_date:
            continue
        base_vol = np.random.normal(0, 0.01)
        rsi_val = future.loc[i, 'rsi']
        if rsi_val > 70:
            rsi_eff = np.random.uniform(-0.02, -0.005)
        elif rsi_val < 30:
            rsi_eff = np.random.uniform(0.005, 0.02)
        else:
            rsi_eff = 0

        vol_eff = 0
        if future.loc[i, 'volume_ratio'] > 1.5:
            vol_eff = np.random.uniform(-0.02, 0.02)

        sma_eff = 0
        price = forecast.loc[i, 'yhat']
        sma20 = future.loc[i, 'sma20']
        if price < sma20 * 0.98:
            sma_eff = np.random.uniform(0.005, 0.015)
        elif price > sma20 * 1.02:
            sma_eff = np.random.uniform(-0.015, -0.005)

        total_eff = base_vol + rsi_eff + vol_eff + sma_eff
        forecast.loc[i, 'yhat'] *= (1 + total_eff)
        forecast.loc[i, 'yhat_lower'] = forecast.loc[i, 'yhat'] * 0.95
        forecast.loc[i, 'yhat_upper'] = forecast.loc[i, 'yhat'] * 1.05

    # ----------------------------------------
    # 시각화
    plt.figure(figsize=(16,10))
    # 실제 주가
    plt.plot(df['date'], df['close'], 'k-', label='Real')
    # 이동평균선
    plt.plot(df['date'], df['sma20'], 'g-', alpha=0.5, label='SMA20')
    plt.plot(df['date'], df['sma50'], 'b-', alpha=0.5, label='SMA50')
    plt.plot(df['date'], df['sma200'], 'r-', alpha=0.5, label='SMA200')

    # 예측
    mask = forecast['ds'] > last_date
    plt.plot(forecast.loc[mask,'ds'], forecast.loc[mask,'yhat'], 'r--', label='Predicted')
    plt.fill_between(forecast.loc[mask,'ds'],
                     forecast.loc[mask,'yhat_lower'],
                     forecast.loc[mask,'yhat_upper'],
                     color='red', alpha=0.2)

    # 지지선 / 저항선 (이동평균 + 고점/저점)
    current_price = df['close'].iloc[-1]
    support_levels = [lvl for lvl in [df['sma20'].iloc[-1], df['sma50'].iloc[-1], df['sma200'].iloc[-1]]
                      if lvl < current_price]
    resistance_levels = [lvl for lvl in [df['sma20'].iloc[-1], df['sma50'].iloc[-1], df['sma200'].iloc[-1]]
                         if lvl > current_price]

    # 최근 고점/저점 추가 (window=10 범위에서 국소 최대/최소)
    window = 10
    for i in range(window, len(df) - window):
        # 국소 최대
        if df['close'].iloc[i] == max(df['close'].iloc[i-window:i+window]):
            if df['close'].iloc[i] > current_price:
                resistance_levels.append(df['close'].iloc[i])
        # 국소 최소
        if df['close'].iloc[i] == min(df['close'].iloc[i-window:i+window]):
            if df['close'].iloc[i] < current_price:
                support_levels.append(df['close'].iloc[i])

    for idx, lvl in enumerate(support_levels):
        plt.axhline(y=lvl, color='green', linestyle='--', alpha=0.4,
                    label='Support' if idx == 0 else "_")

    for idx, lvl in enumerate(resistance_levels):
        plt.axhline(y=lvl, color='red', linestyle='--', alpha=0.4,
                    label='Resistance' if idx == 0 else "_")

    # 거래량 스파이크 표시
    volume_spikes = df[df['volume_spike'] == 1]
    for i, date in enumerate(volume_spikes['date']):
        plt.axvline(x=date, color='purple', alpha=0.2,
                    label='Volume Spike' if i == 0 else "_")

    # 예측 시작 시점 표시
    plt.axvline(x=last_date, color='black', linestyle='--', label='Start')

    plt.title(f'{ticker} Price Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ----------------------------------------
    # 그래프를 메모리에 저장 -> base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode()
    plt.close()

    return jsonify({"image": base64_img})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
