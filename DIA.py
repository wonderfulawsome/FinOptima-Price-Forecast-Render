import os
import datetime
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# API 키를 직접 입력합니다.
API_KEY = "523e3qcGaXMikqx3nh4mOcdo8Kr9gxjY"

def get_data(ticker):
    # FMP API 엔드포인트: serietype=line로 단순 종가 데이터 요청
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={API_KEY}&serietype=line"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"FMP API error: {response.text}")
    
    data_json = response.json()
    historical = data_json.get("historical", [])
    if not historical:
        raise Exception("No historical data found from FMP API.")
    
    df = pd.DataFrame(historical)
    if "date" not in df.columns or "close" not in df.columns:
        raise Exception("Unexpected data format returned from FMP API.")
    
    df = df[['date', 'close']].rename(columns={'close': 'price'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)
    df = df[df['date'] >= cutoff_date].reset_index(drop=True)
    
    if df.empty:
        raise Exception("No data available after filtering by cutoff_date.")
    
    print("Data after filtering by cutoff_date:")
    print(df.head())
    
    return df

def generate_forecast(ticker):
    df = get_data(ticker)
    df_prophet = df.rename(columns={"date": "ds", "price": "y"})
    df_prophet = df_prophet.sort_values('ds')
    
    print(f"Training data records: {len(df_prophet)}")
    if len(df_prophet) < 2:
        raise Exception("Not enough data to generate forecast.")
    
    model = Prophet()
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=7, freq='B')
    forecast = model.predict(future)
    
    forecast_df = forecast[['ds', 'yhat']]
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    forecast_df = forecast_df.sort_values('ds')
    
    history_end = df_prophet['ds'].max()
    # 최근 30일치 실제 데이터
    recent_history = df_prophet[df_prophet['ds'] > history_end - pd.Timedelta(days=30)]
    recent_history['type'] = 'actual'
    
    prediction_part = forecast_df[forecast_df['ds'] > history_end].copy()
    prediction_part = prediction_part.rename(columns={"yhat": "y"})
    prediction_part['type'] = 'forecast'
    
    merged = pd.concat([recent_history[['ds', 'y', 'type']],
                        prediction_part[['ds', 'y', 'type']]])
    merged['ds'] = pd.to_datetime(merged['ds'])
    merged = merged.sort_values('ds')
    merged['ds'] = merged['ds'].dt.strftime('%Y-%m-%d')
    
    print("Merged forecast data:")
    print(merged.head())
    
    return [
        {"date": row['ds'], "price": round(row['y'], 2), "type": row['type']}
        for _, row in merged.iterrows()
    ]

@app.route('/forecast', methods=['POST'])
def forecast_endpoint():
    data = request.get_json()
    ticker = data.get("ticker", "DIA")
    try:
        forecast_data = generate_forecast(ticker)
        return jsonify({"forecast": forecast_data})
    except Exception as e:
        print("Error generating forecast:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return jsonify({"message": "JSON Forecast API is running"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
