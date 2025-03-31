import os
import io
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from prophet import Prophet
from alpha_vantage.timeseries import TimeSeries

app = Flask(__name__)
CORS(app)

API_KEY = 'NE5PBQ8LFYJNMMGJ'  # 실제 Alpha Vantage API 키 입력

# 캐시된 데이터 불러오기 또는 새로 받아오기
def get_cached_data(ticker):
    cache_file = f"cache_{ticker}.csv"

    if os.path.exists(cache_file):
        modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.datetime.now() - modified_time < datetime.timedelta(hours=24):
            return pd.read_csv(cache_file, parse_dates=['date'])

    # 새로 받아오기
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=ticker, outputsize='full')
    data = data[['4. close']].rename(columns={'4. close': 'price'})
    data.index = pd.to_datetime(data.index)

    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=365 * 5)
    df = data[data.index >= cutoff_date].reset_index()
    df = df.rename(columns={"index": "date"})

    df.to_csv(cache_file, index=False)
    return df

# Prophet 예측 및 이미지 반환
def get_forecast_plot(ticker):
    df = get_cached_data(ticker)
    df_prophet = df.rename(columns={"date": "ds", "price": "y"})

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=7, freq='B')
    forecast = model.predict(future)

    fig = model.plot(forecast)
    plt.title(f"{ticker} 7-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# 예측 API
@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    ticker = data.get("ticker", "QQQ")
    try:
        image_buf = get_forecast_plot(ticker)
        return send_file(image_buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 기본 루트 요청 (Render용)
@app.route('/')
def index():
    return jsonify({"message": "Forecast API is running"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render용 포트
    app.run(host="0.0.0.0", port=port, debug=True)
