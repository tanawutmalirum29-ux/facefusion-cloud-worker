FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ARG FACEFUSION_VERSION=3.9.2
ENV FACEFUSION_VERSION=${FACEFUSION_VERSION}
ENV FACEFUSION_ROOT=/facefusion
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /facefusion
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python-is-python3 python3-pip git curl ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${FACEFUSION_VERSION} https://github.com/facefusion/facefusion.git . \
    && python install.py cuda@12 --skip-conda

COPY requirements.txt /tmp/facefusion-worker-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/facefusion-worker-requirements.txt
COPY cloud_api.py /facefusion/cloud_api.py

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "cloud_api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
