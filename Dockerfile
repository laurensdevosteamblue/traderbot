FROM python:3.14-slim

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY traderbot ./traderbot
COPY .streamlit ./.streamlit

CMD ["python", "-m", "traderbot", "run"]
