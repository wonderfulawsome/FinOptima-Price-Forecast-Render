import os
import datetime
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("FMP_API_KEY")

def generate_forecast(ticker):
    # URL에서 timeseries 파라미터 제거
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"FMP API error: {response.text}")
    data = response.json()
    historical = data.get("historical", [])
    if not historical:
        raise Exception("No historical data found.")
    
    # DataFrame 생성 및 정리
    df = pd.DataFrame(historical)
    df = df[['date', 'close']].rename(columns={'close': 'price'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 100일치 데이터로 필터링
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=100)
    df = df[df['date'] >= cutoff_date].reset_index(drop=True)
    
    # Prophet 모델 학습용 데이터 준비
    df_prophet = df.rename(columns={"date": "ds", "price": "y"})
    df_prophet = df_prophet.sort_values('ds')
    
    model = Prophet()
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=7, freq='B')
    forecast = model.predict(future)
    
    forecast_df = forecast[['ds', 'yhat']]
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    forecast_df = forecast_df.sort_values('ds')
    
    history_end = df_prophet['ds'].max()
    actual = df_prophet.copy()
    actual['type'] = 'actual'
    
    forecast_part = forecast_df[forecast_df['ds'] > history_end].copy()
    forecast_part = forecast_part.rename(columns={"yhat": "y"})
    forecast_part['type'] = 'forecast'
    
    merged = pd.concat([actual[['ds', 'y', 'type']], forecast_part[['ds', 'y', 'type']]])
    merged['ds'] = pd.to_datetime(merged['ds'])
    merged = merged.sort_values('ds')
    merged['ds'] = merged['ds'].dt.strftime('%Y-%m-%d')
    
    return [
        {"date": row['ds'], "price": round(row['y'], 2), "type": row['type']}
        for _, row in merged.iterrows()
    ]

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    ticker = data.get("ticker", "DIA")
    try:
        forecast_data = generate_forecast(ticker)
        return jsonify({"forecast": forecast_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return jsonify({"message": "JSON Forecast API is running"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
