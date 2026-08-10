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
# ★secrets_loader.py 를 같이 넣는다 — main.py 가 이것을 부른다.
#   빠뜨리면 import 부터 실패해서 파드가 아예 안 뜬다(조용히 지나가지는 않는다).
COPY main.py secrets_loader.py ./

# ★★모델(3.09GB)을 ★이미지에 굽는다 — 2026-08-10 결정
#
# ★왜 — 전에는 쿠버네티스 PVC(디스크)에 캐시로 두었다. 그런데 그 디스크는
#   ★가용영역에 묶인다(kr-central-2-a). GPU 노드풀을 지우고 ★2-b 에 다시 만들면
#   디스크가 안 붙어 파드가 ★영영 Pending 이 된다. 원인이 스토리지 쪽이라
#   GPU 문제로 오해하기도 쉽다.
#
#   ★앱 5개 중 이 워커만 디스크를 갖고 있었다. 없애면 앱 층이 다시 ★완전 무상태가 되고,
#   노드풀을 언제 어디에 다시 만들어도 안 막힌다.
#
# ★왜 emptyDir 이 아니라 이미지인가 — 둘 다 디스크는 없앤다. 다만 emptyDir 은
#   ★파드가 뜰 때마다 HuggingFace 에서 다시 받는다(실측 38.9MB/s → 1.2분).
#     ⚠️HuggingFace 가 죽어 있으면 ★전사를 아예 못 한다 — 런타임 의존이 생긴다
#     ⚠️`main` 을 따라가므로 모델 판이 ★조용히 바뀔 수 있다
#   이미지에 구우면 런타임 의존이 ★0 이고 판이 못 박힌다. 대신 이미지가 6.6→약 9.5GB
#   (받는 시간 58초 → 약 85초). 이 서비스는 자주 안 바뀌므로 그쪽이 낫다.
#
# ★판을 못 박는다 — revision 을 안 주면 `main` 을 따라가 재현이 안 된다.
#   아래 값은 ★2026-08-10 에 실제로 돌던 파드의 캐시에서 읽은 것이다.
ARG STT_MODEL_REPO=Systran/faster-whisper-large-v3
ARG STT_MODEL_REVISION=edaa852ec7e145841d8ffdb056a99866b5f0a478
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${STT_MODEL_REPO}', revision='${STT_MODEL_REVISION}', local_dir='/opt/whisper-model')" \
    && rm -rf /root/.cache/huggingface \
    && ls -l /opt/whisper-model \
    && du -sh /opt/whisper-model \
    && test -s /opt/whisper-model/model.bin \
    && test -s /opt/whisper-model/config.json \
    && test -s /opt/whisper-model/tokenizer.json \
    && test -s /opt/whisper-model/vocabulary.json
# ★위 test 4줄이 있는 이유 — 일부만 받아져도 빌드는 성공해 버린다. 그러면 파드는
#   멀쩡히 뜨고 ★첫 전사에서야 죽는다. 빌드에서 시끄럽게 실패하는 편이 낫다.

# 기본값 — GPU(T4) 기준. 호스트에서 -e로 덮어쓴다(예: 모델 크기·토큰).
# ★STT_MODEL 이 ★경로다. faster-whisper 는 이름 대신 로컬 경로를 받으면
#   HuggingFace 를 ★아예 안 부른다. (이름 `large-v3` 로 두면 캐시를 찾으러 나간다)
ENV STT_MODEL=/opt/whisper-model \
    STT_DEVICE=cuda \
    STT_COMPUTE=float16

EXPOSE 8100
# 워커는 사내망(백엔드)만 접근 — 외부 노출 금지(방화벽/보안그룹으로 백엔드 IP만 허용).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
