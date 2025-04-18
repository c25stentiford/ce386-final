FROM nvcr.io/nvidia/pytorch:25.02-py3-igpu
WORKDIR /app
RUN apt update && \
    apt install -y --no-install-recommends libportaudiocpp0 libportaudio2 && \
    apt clean
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir transformers==4.49.0 accelerate==1.5.2 sounddevice
COPY speech_recognition.py .
ENV HF_HOME="/huggingface/"
ENTRYPOINT ["python", "speech_recognition.py"]
