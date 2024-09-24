FROM python:3.10
RUN pip install git+https://github.com/huggingface/diffusers.git
RUN pip install --no-cache-dir accelerate invisible_watermark transformers==4.42.4 gradio sentencepiece protobuf
COPY app.py app.py
CMD ["python3", "app.py"]
