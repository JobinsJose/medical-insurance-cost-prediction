FROM python:3.11-slim

WORKDIR /app/APP

COPY App/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "App/app.py"]