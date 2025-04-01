import os
import datetime
import requests
import pandas as pd
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet

app = Flask(__name__)
CORS(app)

# API 키 가져오기 - 환경 변수에 설정되어 있어야 함
API_KEY = os.environ.get("FMP_API_KEY")
if not API_KEY:
    print("WARNING: FMP_API_KEY 환경 변수가 설정되지 않았습니다.")

def generate_forecast(ticker):
    """
    특정 주식의 과거 데이터를 가져와 미래 가격 예측
    """
    try:
        print(f"Generating forecast for {ticker} at {datetime.datetime.now()}")
        
        # FMP API에서 과거 주가 데이터 가져오기
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={API_KEY}"
        
        # API 요청에 헤더 추가
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache"
        }
        
        print(f"Requesting data from FMP API: {url}")
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"FMP API error: {response.status_code}, {response.text}")
            raise Exception(f"FMP API error: Status {response.status_code}")
        
        data = response.json()
        historical = data.get("historical", [])
        
        if not historical:
            print("No historical data found from FMP API")
            raise Exception("No historical data found for this ticker.")
        
        print(f"Received {len(historical)} historical data points")
        
        # DataFrame 생성 및 정리
        df = pd.DataFrame(historical)
        df = df[['date', 'close']].rename(columns={'close': 'price'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 100일치 데이터로 필터링
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=100)
        df = df[df['date'] >= cutoff_date].reset_index(drop=True)
        
        if len(df) < 30:
            print(f"Insufficient data: only {len(df)} days available")
            raise Exception("Insufficient historical data for prediction.")
        
        # Prophet 모델 학습용 데이터 준비
        df_prophet = df.rename(columns={"date": "ds", "price": "y"})
        df_prophet = df_prophet.sort_values('ds')
        
        print("Training Prophet model...")
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.fit(df_prophet)
        
        # 향후 7일(영업일) 예측
        future = model.make_future_dataframe(periods=7, freq='B')
        forecast = model.predict(future)
        
        forecast_df = forecast[['ds', 'yhat']]
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        forecast_df = forecast_df.sort_values('ds')
        
        # 실제 데이터와 예측 데이터 병합
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
        
        print(f"Forecast completed successfully with {len(forecast_part)} predicted days")
        
        # JSON 형식으로 변환
        return [
            {"date": row['ds'], "price": round(row['y'], 2), "type": row['type']}
            for _, row in merged.iterrows()
        ]
        
    except Exception as e:
        print(f"Error in generate_forecast: {str(e)}")
        print(traceback.format_exc())  # 상세한 오류 추적 정보 출력
        raise

@app.route('/forecast', methods=['POST'])
def forecast():
    """
    주가 예측 API 엔드포인트
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        ticker = data.get("ticker")
        
        if not ticker:
            return jsonify({"error": "Ticker symbol is required"}), 400
            
        print(f"Received forecast request for ticker: {ticker}")
        
        forecast_data = generate_forecast(ticker)
        
        response = jsonify({"forecast": forecast_data})
        
        # 캐시 방지 헤더 추가
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return response
        
    except Exception as e:
        print(f"Error in forecast endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    """
    서버 상태 확인용 엔드포인트
    """
    response = jsonify({
        "message": "JSON Forecast API is running", 
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    # 캐시 방지 헤더 추가
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
