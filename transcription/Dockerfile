# Compatible with Jetpack 6.2
FROM nvcr.io/nvidia/pytorch:25.02-py3-igpu
WORKDIR /app
RUN apt update && apt install -y --no-install-recommends libportaudiocpp0 && apt clean \
    pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir torch transformers sounddevice
COPY speech_recognition.py .
ENV HF_HOME="/huggingface/"
ENTRYPOINT ["python", "speech_recognition.py"]