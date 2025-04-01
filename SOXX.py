import os
import io
import datetime
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
from alpha_vantage.timeseries import TimeSeries
from sqlalchemy import create_engine, Column, String, Text, DateTime, Table, MetaData, select, insert, update

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")  # Render에서 설정한 PostgreSQL URL

# SQLAlchemy 엔진 및 메타데이터 설정
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# 캐시 데이터를 저장할 테이블 정의
cache_table = Table(
    'cache_data', metadata,
    Column('ticker', String, primary_key=True),
    Column('data', Text),
    Column('updated_at', DateTime)
)

metadata.create_all(engine)

def get_cached_data(ticker):
    conn = engine.connect()
    s = select([cache_table]).where(cache_table.c.ticker == ticker)
    result = conn.execute(s).fetchone()
    # 캐시가 있고, 24시간 이내이면 사용
    if result:
        updated_at = result['updated_at']
        if datetime.datetime.now() - updated_at < datetime.timedelta(hours=24):
            csv_data = result['data']
            df = pd.read_csv(io.StringIO(csv_data), parse_dates=['date'])
            conn.close()
            return df

    # 캐시가 없거나 24시간 이상 지난 경우, API에서 데이터 가져오기
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
    data = data[['4. close']].rename(columns={'4. close': 'price'})
    data.index = pd.to_datetime(data.index)

    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)
    df = data[data.index >= cutoff_date].reset_index()
    df = df.rename(columns={"index": "date"})

    # DataFrame을 CSV 문자열로 변환하여 DB에 저장
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    now = datetime.datetime.now()
    if result:
        upd = update(cache_table).where(cache_table.c.ticker == ticker).values(data=csv_data, updated_at=now)
        conn.execute(upd)
    else:
        ins = insert(cache_table).values(ticker=ticker, data=csv_data, updated_at=now)
        conn.execute(ins)
    conn.close()
    return df

def generate_forecast(ticker):
    df = get_cached_data(ticker)
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
    recent_history = df_prophet[df_prophet['ds'] > history_end - pd.Timedelta(days=30)].copy()
    recent_history['type'] = 'actual'

    prediction_part = forecast_df[forecast_df['ds'] > history_end].copy()
    prediction_part = prediction_part.rename(columns={"yhat": "y"})
    prediction_part['type'] = 'forecast'

    merged = pd.concat([
        recent_history[['ds', 'y', 'type']],
        prediction_part[['ds', 'y', 'type']]
    ])
    merged['ds'] = pd.to_datetime(merged['ds'])
    merged = merged.sort_values('ds')
    merged['ds'] = merged['ds'].dt.strftime('%Y-%m-%d')

    return [
        {"date": row['ds'], "price": round(row['y'], 2), "type": row['type']}
        for _, row in merged.iterrows()
    ]

@app.route('/forecast', methods=['POST'])
def forecast_route():
    data = request.get_json()
    ticker = data.get("ticker", "SOXX")
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
