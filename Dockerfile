FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py crop_yield_model.py crop_yield.csv index.html ./

ENV HOST=0.0.0.0
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
