import os
import datetime
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
from alpha_vantage.timeseries import TimeSeries

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

def get_cached_data(ticker):
    cache_file = f"cache_{ticker}.csv"
    if os.path.exists(cache_file):
        modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.datetime.now() - modified_time < datetime.timedelta(hours=24):
            return pd.read_csv(cache_file, parse_dates=['date'])

    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
    data = data[['4. close']].rename(columns={'4. close': 'price'})
    data.index = pd.to_datetime(data.index)

    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)
    df = data[data.index >= cutoff_date].reset_index()
    df = df.rename(columns={"index": "date"})

    df.to_csv(cache_file, index=False)
    return df

def generate_forecast(ticker):
    df = get_cached_data(ticker)
    df_prophet = df.rename(columns={"date": "ds", "price": "y"})
    # 날짜 오름차순 정렬
    df_prophet = df_prophet.sort_values('ds')

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=7, freq='B')
    forecast = model.predict(future)

    forecast_df = forecast[['ds', 'yhat']]
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    # 날짜 정렬
    forecast_df = forecast_df.sort_values('ds')

    history_end = df_prophet['ds'].max()
    recent_history = df_prophet[df_prophet['ds'] > history_end - pd.Timedelta(days=30)]
    recent_history['type'] = 'actual'

    prediction_part = forecast_df[forecast_df['ds'] > history_end].copy()
    prediction_part = prediction_part.rename(columns={"yhat": "y"})
    prediction_part['type'] = 'forecast'

    merged = pd.concat([
        recent_history[['ds', 'y', 'type']],
        prediction_part[['ds', 'y', 'type']]
    ])
    # 날짜 오름차순 정렬 
    merged['ds'] = pd.to_datetime(merged['ds'])
    merged = merged.sort_values('ds')
    # 문자열 변환
    merged['ds'] = merged['ds'].dt.strftime('%Y-%m-%d')

    return [
        {"date": row['ds'], "price": round(row['y'], 2), "type": row['type']}
        for _, row in merged.iterrows()
    ]

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    ticker = data.get("ticker", "QQQ")
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
