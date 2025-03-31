# Python 3.10 이미지 사용
FROM python:3.10

WORKDIR /app

# 필수 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Flask 앱 실행 (QQQ.py로 변경)
COPY . .
CMD ["python", "QQQ.py"]
