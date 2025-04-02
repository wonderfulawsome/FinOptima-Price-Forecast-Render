# 베이스 이미지로 Python 3.9 선택
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 캐싱 개선을 위해 먼저 requirements.txt만 복사
COPY requirements.txt .

# 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 모든 ETF 관련 Python 파일 복사
COPY DIA.py IBIT.py QQQ.py SOXX.py SPY.py ./

# API 키를 환경 변수로 설정 (Render에서 설정)
# ENV FMP_API=your_api_key_here

# Gunicorn 포트 설정 (Render는 PORT 환경 변수를 제공)
ENV PORT=8000

# 비 root 유저 생성 및 전환 (보안 강화)
RUN adduser --disabled-password --gecos "" appuser
RUN chown -R appuser:appuser /app
USER appuser

# Gunicorn으로 QQQ.py 실행 (기본 엔트리 포인트로 설정)
# 다른 ETF가 메인 진입점이라면 해당 파일명으로 변경
CMD gunicorn --bind 0.0.0.0:$PORT QQQ:app
