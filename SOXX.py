import os
import io
import datetime
import pandas as pd
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
from alpha_vantage.timeseries import TimeSeries
from sqlalchemy import create_engine, Column, String, Text, DateTime, Table, MetaData, select, insert, update, inspect

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")  # Render에서 설정한 PostgreSQL URL

if not DATABASE_URL:
    logger.error("DATABASE_URL 환경변수가 설정되지 않았습니다.")
    DATABASE_URL = "sqlite:///temp.db"  # 기본값으로 SQLite 사용
    logger.info(f"기본 데이터베이스로 설정: {DATABASE_URL}")

try:
    # SQLAlchemy 엔진 및 메타데이터 설정
    engine = create_engine(DATABASE_URL)
    # 데이터베이스 연결 테스트
    with engine.connect() as conn:
        logger.info("데이터베이스 연결 성공!")
    
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

    # 테이블이 이미 존재하는지 확인
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    logger.info(f"기존 테이블: {existing_tables}")
    
    # 테이블 생성 (존재하지 않으면)
    metadata.create_all(engine)
    logger.info("테이블 생성 완료!")

except Exception as e:
    logger.error(f"데이터베이스 초기화 오류: {str(e)}")
    raise

# 오늘 API 호출이 가능한지 확인
def can_call_api_today():
    today = datetime.datetime.now().date()
    
    try:
        with engine.connect() as conn:
            s = select([api_status_table]).where(api_status_table.c.id == 'alpha_vantage')
            result = conn.execute(s).fetchone()
            
            # 상태 기록이 없으면 초기화 (첫 실행)
            if not result:
                logger.info("API 호출 상태 레코드가 없습니다. 새로운 레코드를 생성합니다.")
                ins = insert(api_status_table).values(
                    id='alpha_vantage',
                    last_call_date=datetime.datetime.now() - datetime.timedelta(days=1)  # 어제로 설정
                )
                conn.execute(ins)
                conn.commit()  # 명시적 커밋
                return True  # API 호출 가능
            
            # 마지막 호출 날짜가 오늘이 아니면 호출 가능
            last_call_date = result['last_call_date'].date()
            can_call = last_call_date < today
            logger.info(f"마지막 API 호출 날짜: {last_call_date}, 오늘 호출 가능: {can_call}")
            return can_call
    except Exception as e:
        logger.error(f"API 호출 상태 확인 중 오류: {str(e)}")
        # 오류 발생 시 보수적으로 API 호출 불가능으로 처리
        return False

# API 호출 후 상태 업데이트
def update_api_call_status():
    try:
        with engine.begin() as conn:
            now = datetime.datetime.now()
            upd = update(api_status_table).where(api_status_table.c.id == 'alpha_vantage').values(
                last_call_date=now
            )
            conn.execute(upd)
            logger.info(f"API 호출 상태 업데이트 완료: {now}")
    except Exception as e:
        logger.error(f"API 호출 상태 업데이트 중 오류: {str(e)}")

def get_cached_data(ticker):
    logger.info(f"{ticker}에 대한 데이터 요청 시작")
    
    # 캐시 확인
    try:
        with engine.connect() as conn:
            s = select([cache_table]).where(cache_table.c.ticker == ticker)
            result = conn.execute(s).fetchone()
        
        if result:
            logger.info(f"{ticker}에 대한 캐시 데이터 발견")
            updated_at = result['updated_at']
            is_cache_fresh = (datetime.datetime.now() - updated_at) < datetime.timedelta(hours=24)
            
            if is_cache_fresh:
                logger.info(f"{ticker} 캐시가 신선함 (24시간 이내)")
            else:
                logger.info(f"{ticker} 캐시가 오래됨 (24시간 이상)")
            
            # 1. 캐시가 신선하거나
            # 2. 오늘 이미 API를 호출했으면 캐시 사용
            if is_cache_fresh or not can_call_api_today():
                csv_data = result['data']
                logger.info(f"{ticker} 캐시 데이터 사용")
                return pd.read_csv(io.StringIO(csv_data), parse_dates=['date'])
        else:
            logger.info(f"{ticker}에 대한 캐시 데이터 없음")
        
        # API 호출 가능한지 확인
        if can_call_api_today():
            logger.info(f"{ticker}에 대한 API 호출 시작")
            try:
                # API 호출
                ts = TimeSeries(key=API_KEY, output_format='pandas')
                data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
                logger.info(f"{ticker}에 대한 API 호출 성공")
                
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
                        logger.info(f"{ticker} 캐시 데이터 업데이트")
                        upd = update(cache_table).where(cache_table.c.ticker == ticker).values(data=csv_data, updated_at=now)
                        conn.execute(upd)
                    else:
                        logger.info(f"{ticker} 캐시 데이터 새로 생성")
                        ins = insert(cache_table).values(ticker=ticker, data=csv_data, updated_at=now)
                        conn.execute(ins)
                
                # API 호출 성공 시 상태 업데이트
                update_api_call_status()
                
                return df
            except Exception as e:
                logger.error(f"{ticker}에 대한 API 호출 실패: {str(e)}")
                # API 호출 실패 시 캐시 사용 (캐시가 있는 경우)
                if result:
                    csv_data = result['data']
                    logger.info(f"API 호출 실패, {ticker}의 기존 캐시 데이터 사용")
                    return pd.read_csv(io.StringIO(csv_data), parse_dates=['date'])
                else:
                    # 캐시도 없고 API 호출도 실패한 경우
                    raise Exception(f"Failed to fetch data for {ticker}: {str(e)}")
        else:
            logger.info("오늘의 API 호출 제한에 도달")
            # API 호출 불가능하고 캐시가 없는 경우
            if not result:
                msg = f"{ticker}에 대한 캐시 데이터가 없고, 일일 API 호출 제한에 도달했습니다."
                logger.error(msg)
                raise Exception(msg)
            
            # 캐시가 있지만 오래된 경우 (이미 위에서 처리됨)
            csv_data = result['data']
            logger.info(f"API 호출 제한으로 인해 {ticker}의 오래된 캐시 데이터 사용")
            return pd.read_csv(io.StringIO(csv_data), parse_dates=['date'])
    except Exception as e:
        logger.error(f"{ticker} 데이터 가져오기 중 오류: {str(e)}")
        raise

def generate_forecast(ticker):
    df = get_cached_data(ticker)
    logger.info(f"{ticker} 데이터 로드 완료, 예측 시작")
    
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

    logger.info(f"{ticker} 예측 완료")
    return [
        {"date": row['ds'], "price": round(row['y'], 2), "type": row['type']}
        for _, row in merged.iterrows()
    ]

@app.route('/forecast', methods=['POST'])
def forecast_route():
    data = request.get_json()
    ticker = data.get("ticker", "SOXX")
    logger.info(f"{ticker}에 대한 예측 요청 받음")
    try:
        forecast_data = generate_forecast(ticker)
        return jsonify({"forecast": forecast_data})
    except Exception as e:
        logger.error(f"예측 처리 중 오류: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug', methods=['GET'])
def debug_route():
    """데이터베이스 상태 디버깅을 위한 엔드포인트"""
    try:
        db_info = {
            "db_url": DATABASE_URL.replace(":@", ":***@"),  # 비밀번호 가림
            "tables": []
        }
        
        # 테이블 목록 확인
        inspector = inspect(engine)
        db_info["tables"] = inspector.get_table_names()
        
        # API 상태 확인
        with engine.connect() as conn:
            api_status = conn.execute(select([api_status_table])).fetchall()
            db_info["api_status"] = [dict(row) for row in api_status] if api_status else "No records"
            
            # 캐시 상태 확인 (티커 목록만)
            cache_status = conn.execute(select([cache_table.c.ticker, cache_table.c.updated_at])).fetchall()
            db_info["cache_entries"] = [
                {"ticker": row.ticker, "updated_at": row.updated_at.strftime("%Y-%m-%d %H:%M:%S")} 
                for row in cache_status
            ] if cache_status else "No records"
        
        return jsonify(db_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return jsonify({"message": "JSON Forecast API is running"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
