import os
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify
from flask_cors import CORS
from prophet import Prophet
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

np.random.seed(42)

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("FMP_API", "DEMO_KEY")

def get_stock_data(ticker, days=200):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data['historical'][:days])
    df['date'] = pd.to_datetime(df['date'])
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

def forecast_exogenous(df, col, periods):
    temp = df[['date', col]].rename(columns={'date': 'ds', col: 'y'})
    m = Prophet()
    m.fit(temp)
    future_exog = m.make_future_dataframe(periods=periods, freq='D')
    forecast_exog = m.predict(future_exog)
    return forecast_exog[['ds', 'yhat']]

@app.route("/forecast/<ticker>", methods=["GET"])
def forecast(ticker):
    df = get_stock_data(ticker, 200)
    df = add_technical_indicators(df)

    # 학습 데이터 준비
    train_df = df[['date', 'close', 'sma20', 'sma50', 'volume_ratio', 'rsi']].rename(columns={'date': 'ds', 'close': 'y'})
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
    model.fit(train_df)

    future = model.make_future_dataframe(periods=30)
    last_date = df['date'].max()

    # 외생변수 Prophet 모델로 미래 예측
    for col in ['sma20', 'sma50', 'volume_ratio', 'rsi']:
        forecast_exog = forecast_exogenous(df, col, 30)
        # 과거는 실제값, 미래는 예측값 사용
        future[col] = future['ds'].apply(
            lambda d: df.loc[df['date'] == d, col].values[0] if d <= last_date
            else forecast_exog.loc[forecast_exog['ds'] == d, 'yhat'].values[0]
        )

    forecast_df = model.predict(future)

    current_price = df['close'].iloc[-1]
    sma_now = [df['sma20'].iloc[-1], df['sma50'].iloc[-1], df['sma200'].iloc[-1]]
    support_list = [x for x in sma_now if x < current_price]
    resist_list  = [x for x in sma_now if x > current_price]

    window = 10
    for i in range(window, len(df) - window):
        if df['close'].iloc[i] == max(df['close'].iloc[i-window:i+window]):
            if df['close'].iloc[i] > current_price:
                resist_list.append(df['close'].iloc[i])
        if df['close'].iloc[i] == min(df['close'].iloc[i-window:i+window]):
            if df['close'].iloc[i] < current_price:
                support_list.append(df['close'].iloc[i])

    volume_spike_dates = df.loc[df['volume_spike'] == 1, 'date'].dt.strftime("%Y-%m-%d").tolist()

    real_rows = df[['date', 'close', 'sma20', 'sma50', 'sma200', 'volume_spike', 'rsi']]
    pred_rows = forecast_df[forecast_df['ds'] > last_date].copy()

    real_data = []
    for _, row in real_rows.iterrows():
        real_data.append({
            "ds": row['date'].strftime("%Y-%m-%d"),
            "close": row['close'],
            "sma20": row['sma20'],
            "sma50": row['sma50'],
            "sma200": row['sma200'],
            "volume_spike": int(row['volume_spike']),
            "rsi": row['rsi']
        })

    pred_data = []
    for _, row in pred_rows.iterrows():
        dstr = row['ds'].strftime("%Y-%m-%d")
        pred_data.append({
            "ds": dstr,
            "yhat": row['yhat'],
            "yhat_lower": row['yhat_lower'],
            "yhat_upper": row['yhat_upper'],
            "sma20": future.loc[future['ds'] == row['ds'], 'sma20'].values[0],
            "sma50": future.loc[future['ds'] == row['ds'], 'sma50'].values[0],
            "volume_spike": 1 if future.loc[future['ds'] == row['ds'], 'volume_ratio'].values[0] > 1.5 else 0,
            "rsi": future.loc[future['ds'] == row['ds'], 'rsi'].values[0]
        })

    return jsonify({
        "real": real_data,
        "predicted": pred_data,
        "support": sorted(support_list),
        "resistance": sorted(resist_list),
        "forecastStart": last_date.strftime("%Y-%m-%d"),
        "volumeSpikes": volume_spike_dates
    })

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
