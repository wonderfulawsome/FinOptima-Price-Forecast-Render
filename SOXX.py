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

# API 호출 상태를 저장할 테이블 정의
api_status_table = Table(
    'api_status', metadata,
    Column('id', String, primary_key=True),
    Column('last_call_date', DateTime)
)

metadata.create_all(engine)

# 오늘 API 호출이 가능한지 확인
def can_call_api_today():
    today = datetime.datetime.now().date()
    
    with engine.begin() as conn:
        s = select([api_status_table]).where(api_status_table.c.id == 'alpha_vantage')
        result = conn.execute(s).fetchone()
        
        # 상태 기록이 없으면 초기화 (첫 실행)
        if not result:
            ins = insert(api_status_table).values(
                id='alpha_vantage',
                last_call_date=datetime.datetime.now() - datetime.timedelta(days=1)  # 어제로 설정
            )
            conn.execute(ins)
            return True  # API 호출 가능
        
        # 마지막 호출 날짜가 오늘이 아니면 호출 가능
        last_call_date = result['last_call_date'].date()
        return last_call_date < today

# API 호출 후 상태 업데이트
def update_api_call_status():
    with engine.begin() as conn:
        upd = update(api_status_table).where(api_status_table.c.id == 'alpha_vantage').values(
            last_call_date=datetime.datetime.now()
        )
        conn.execute(upd)

def get_cached_data(ticker):
    # 캐시 확인
    with engine.begin() as conn:
        s = select([cache_table]).where(cache_table.c.ticker == ticker)
        result = conn.execute(s).fetchone()
    
    # 캐시가 있으면 사용할지 결정
    if result:
        updated_at = result['updated_at']
        is_cache_fresh = (datetime.datetime.now() - updated_at) < datetime.timedelta(hours=24)
        
        # 1. 캐시가 신선하거나
        # 2. 오늘 이미 API를 호출했으면 캐시 사용
        if is_cache_fresh or not can_call_api_today():
            csv_data = result['data']
            return pd.read_csv(io.StringIO(csv_data), parse_dates=['date'])
    
    # API 호출 가능한지 확인
    if can_call_api_today():
        try:
            # API 호출
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
            
            # 캐시 업데이트 또는 삽입
            with engine.begin() as conn:
                if result:
                    upd = update(cache_table).where(cache_table.c.ticker == ticker).values(data=csv_data, updated_at=now)
                    conn.execute(upd)
                else:
                    ins = insert(cache_table).values(ticker=ticker, data=csv_data, updated_at=now)
                    conn.execute(ins)
            
            # API 호출 성공 시 상태 업데이트
            update_api_call_status()
            
            return df
        except Exception as e:
            # API 호출 실패 시 캐시 사용 (캐시가 있는 경우)
            if result:
                csv_data = result['data']
                return pd.read_csv(io.StringIO(csv_data), parse_dates=['date'])
            else:
                # 캐시도 없고 API 호출도 실패한 경우
                raise Exception(f"Failed to fetch data for {ticker}: {str(e)}")
    else:
        # API 호출 불가능하고 캐시가 없는 경우
        if not result:
            raise Exception(f"No cache data for {ticker} and daily API call limit reached.")
        
        # 캐시가 있지만 오래된 경우 (이미 위에서 처리됨)
        csv_data = result['data']
        return pd.read_csv(io.StringIO(csv_data), parse_dates=['date'])

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
