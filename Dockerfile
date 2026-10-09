# Calls a Hugging Face Space for SAM 3, so the block itself needs no GPU and no model weights
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY label.py parameters.json LICENSE ./

ENTRYPOINT ["python3", "-u", "label.py"]
