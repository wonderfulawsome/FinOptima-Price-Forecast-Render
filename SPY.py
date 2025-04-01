import os
import datetime
import io
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, Response, render_template_string
from prophet import Prophet
from alpha_vantage.timeseries import TimeSeries

app = Flask(__name__)

# Alpha Vantage API 키 (실제 FMP 키가 아니라 Alpha Vantage 키를 사용합니다)
API_KEY = '523e3qcGaXMikqx3nh4mOcdo8Kr9gxjY'

@app.route('/')
def index():
    # 메인 페이지에서 플롯 이미지를 포함한 간단한 HTML 페이지를 반환합니다.
    html = '''
    <html>
        <head>
            <title>Prophet Forecast Plot</title>
        </head>
        <body style="background: #000; color: #fff; text-align: center;">
            <h1>Prophet 7-Day Forecast for SPY</h1>
            <img src="/plot" alt="Forecast Plot">
        </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/plot')
def plot():
    # Alpha Vantage API를 통해 SPY 데이터 가져오기
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol='SPY', outputsize='full')
    data = data[['4. close']].rename(columns={'4. close': 'price'})
    data.index = pd.to_datetime(data.index)
    
    # 오늘 날짜 기준 500일 전 (약 1.5년, 주석과 맞게 조정 가능)
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=500)
    df = data[data.index >= cutoff_date]
    df = df.reset_index()
    
    # Prophet 모델 학습을 위한 데이터 준비
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df_prophet = df.rename(columns={"date": "ds", "price": "y"})
    
    # Prophet 모델 학습
    model = Prophet()
    model.fit(df_prophet)
    
    # 향후 7일 (영업일 기준) 예측
    future = model.make_future_dataframe(periods=7, freq='B')
    forecast = model.predict(future)
    
    # 실제 데이터와 예측 데이터 분리
    forecast = forecast.set_index('ds')
    history_end = df_prophet['ds'].max()
    forecast_part = forecast.loc[forecast.index > history_end]
    actual_part = df_prophet.set_index('ds')
    
    # 플롯 생성
    plt.figure(figsize=(12, 6))
    plt.plot(actual_part.index, actual_part['y'], color='black', label='Actual Price')
    plt.plot(forecast_part.index, forecast_part['yhat'], color='blue', label='Forecast')
    plt.plot([actual_part.index[-1], forecast_part.index[0]],
             [actual_part['y'].iloc[-1], forecast_part['yhat'].iloc[0]],
             color='blue', linestyle='--')
    plt.fill_between(forecast_part.index,
                     forecast_part['yhat_lower'],
                     forecast_part['yhat_upper'],
                     color='blue', alpha=0.2, label='Confidence Interval')
    plt.title("Prophet 7-Day Forecast (Connected)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # 플롯을 PNG 이미지로 저장하여 반환
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return Response(img.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
