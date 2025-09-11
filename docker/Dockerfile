FROM python:3.10-slim

WORKDIR /app

COPY ./docker/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

COPY ./src ./src
COPY ./model ./model
COPY ./data ./data

EXPOSE 5700

CMD ["sh", "-c", "cd src; fastapi dev web_server.py --host 0.0.0.0 --port 5700"]
