FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app/src

COPY requirements.txt requirements-cpu.txt ./
RUN pip install --no-cache-dir -r requirements-cpu.txt

COPY . .

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-10000}"]