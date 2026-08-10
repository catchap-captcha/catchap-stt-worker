# CatChap STT 워커 — faster-whisper on GPU(Tesla T4). 앱 4대와 같은 Docker 스택으로 통일(A안).
#
# 베이스: CUDA 12.2 런타임 + cuDNN8(faster-whisper/CTranslate2가 요구). nvidia-container-toolkit
# 을 깐 호스트에서 `docker run --gpus all`로 T4를 컨테이너에 연결한다. nvcc(툴킷)는 불필요 —
# 추론은 런타임 라이브러리만 쓴다.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# ffmpeg — faster-whisper가 다양한 컨테이너(mp4/webm)에서 오디오를 디코딩하는 데 사용.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY main.py .

# 기본값 — GPU(T4) 기준. 호스트에서 -e로 덮어쓴다(예: 모델 크기·토큰).
ENV STT_MODEL=large-v3 \
    STT_DEVICE=cuda \
    STT_COMPUTE=float16

EXPOSE 8100
# 워커는 사내망(백엔드)만 접근 — 외부 노출 금지(방화벽/보안그룹으로 백엔드 IP만 허용).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
