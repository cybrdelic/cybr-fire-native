FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=4 \
    LIBGL_ALWAYS_SOFTWARE=1 CYBR_GL_BACKEND=egl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libegl-mesa0 libgl1-mesa-dri libglx-mesa0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --no-compile -r requirements.txt
COPY . .
ENTRYPOINT ["python", "-m", "cybrfire"]
CMD ["device"]
