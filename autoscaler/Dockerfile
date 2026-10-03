# Autoscaler + control-room dashboard image.
#
#   docker build -t k8s-autoscaler/autoscaler:latest -f autoscaler/Dockerfile autoscaler
#   docker run --rm -p 8080:8080 k8s-autoscaler/autoscaler:latest              # demo mode
#   docker run ... k8s-autoscaler/autoscaler:latest --mode live --shadow        # needs cluster access
#
# In-cluster it runs as the `autoscaler` ServiceAccount (see k8s/03-autoscaler/).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# CPU-only PyTorch (the default PyPI wheel on x86-64 drags in ~3 GB of CUDA libs).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install torch --index-url ${TORCH_INDEX} \
 && pip install -r requirements.txt

COPY src ./src
COPY models ./models
COPY results ./results

# non-root
RUN useradd --uid 10001 --create-home autoscaler
USER 10001

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=2)" || exit 1

ENTRYPOINT ["python", "-m", "src.dashboard.server"]
CMD ["--mode", "demo", "--port", "8080"]
