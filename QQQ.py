import os
import io
import datetime
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
from alpha_vantage.timeseries import TimeSeries

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")  # Render 환경변수에서 키 사용

def get_cached_data(ticker):
    cache_file = f"cache_{ticker}.csv"
    if os.path.exists(cache_file):
        modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.datetime.now() - modified_time < datetime.timedelta(hours=24):
            return pd.read_csv(cache_file, parse_dates=['date'])

    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=ticker, outputsize='compact')  # compact = 최근 100일
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

    model = Prophet(seasonality_mode='multiplicative')
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=7, freq='B')  # 7일 예측
    forecast = model.predict(future)

    result_df = forecast[['ds', 'yhat']].tail(7)
    result_df['ds'] = result_df['ds'].dt.strftime('%Y-%m-%d')

    result = [{"date": row['ds'], "price": round(row['yhat'], 2)} for _, row in result_df.iterrows()]
    return result

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
