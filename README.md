# CatChap STT 워커 (faster-whisper / GPU)

강의 오디오를 타임스탬프 자막으로 전사하는 **자체 호스팅 STT 서비스**. OpenAI Whisper
API(유료·25MB 한계·오디오 외부 전송)를 대체 — Tesla T4 GPU에서 무료로, 대용량도 전사한다.

## 왜 만들었나
- 우리에겐 놀고 있는 **GPU(Tesla T4, 16GB)** 가 있다.
- 로컬 스파이크(`spike-whisper/`)에서 faster-whisper large-v3가 잘 되는 걸 검증했다.
- OpenAI는 분당 $0.006 유료 + 25MB 한계 + 오디오를 외부로 보냄. 자체 호스팅은 **과금 0 ·
  대용량 OK · 오디오가 사내에만** 머문다.

## 구조
```
앱 백엔드 ──(영상/오디오 POST, X-Worker-Token)──▶ 이 워커(:8100) ──▶ T4 GPU 전사
                                              ◀──({segments:[{start,end,text}]})──
```
워커는 오직 '오디오→자막' 한 가지만 한다(사용자·DB·시험을 모름). 반환 형태는 기존
`app/clients/stt_client.py`의 OpenAI 경로와 동일해, 백엔드는 소스에 상관없이 같은 코드로 받는다.

## 엔드포인트
- `GET /health` — 기동 확인(모델 미로드).
- `POST /transcribe` (multipart `file`, query `language=ko`, header `X-Worker-Token`) →
  `{ "segments": [{start, end, text}], "duration", "language" }`. 무음/전사 실패는 422.

---

## ★어디서 도나 (2026-08-10 이전)

**쿠버네티스 클러스터의 GPU 노드풀**에서 돈다. 그 전에는 GPU VM(`10.0.5.57`)에서
`docker run --gpus all` 로 사람이 직접 띄우고 있었다.

```
namespace   catchap
Deployment  stt-worker      GPU 1개를 요청한다(nvidia.com/gpu: 1)
Service     stt-worker:8100 클러스터 안에서만 보인다(ClusterIP)
PVC         stt-model-cache 모델 캐시 2.9GB
매니페스트   catchap-infra  k8s/stt-worker/
```

★**GPU 노드를 어떻게 찾아가나** — `nodeSelector` 로 고르지 않는다.
`nvidia.com/gpu: 1` 을 요청하면 **그 자원을 가진 노드는 하나뿐**이라 스케줄러가 알아서 보낸다.
GPU 를 보이게 해 주는 device plugin 은 `catchap-infra` 의 `k8s/80-nvidia-device-plugin.yaml` 이다.

### 배포
`main` 에 병합하면 **저절로 된다.** GitHub Actions 가 이미지를 만들어 Container Registry 에
올리고, `catchap-infra` 의 매니페스트에서 이미지 태그를 그 커밋 해시로 고친다. ArgoCD 가 반영한다.

⚠️**이미지가 6.6GB 다**(CUDA 12.2 + cuDNN8 베이스가 대부분). 그래서 이 저장소의 배포
워크플로만 캐시를 안 쓰고, 러너 디스크를 먼저 비우고, 시간 제한이 45분이다. 이유는
`.github/workflows/deploy.yml` 주석에 있다.

### 확인
```bash
kubectl -n catchap get pods -l app=stt-worker -o wide
kubectl -n catchap exec deploy/stt-worker -- nvidia-smi        # GPU 가 붙었나
kubectl -n catchap exec deploy/backend-api -- \
  curl -s http://stt-worker:8100/health                        # 백엔드에서 보이나
```

---

## 로컬에서 돌려 보기 (GPU 없이)

```bash
docker build -t catchap-stt-worker .
docker run --rm -p 8100:8100 \
  -e STT_DEVICE=cpu -e STT_COMPUTE=int8 -e STT_MODEL=tiny \
  -e STT_WORKER_TOKEN=dev \
  catchap-stt-worker
curl -s localhost:8100/health
```

★`STT_MODEL=tiny` 로 해야 한다. `large-v3` 는 CPU 에서 실용적이지 않다.

## 설정 (환경변수)

| 이름 | 기본값 | 뜻 |
|---|---|---|
| `STT_MODEL` | `large-v3` | 모델 크기 |
| `STT_DEVICE` | `cuda` | `cuda` 또는 `cpu` |
| `STT_COMPUTE` | `float16` | CPU 면 `int8` |
| `STT_WORKER_TOKEN` | (빈 값) | ★백엔드와 나눠 갖는 열쇠. **빈 값이면 인증을 안 한다** |

⚠️`STT_WORKER_TOKEN` 이 비면 **누구나 이 워커를 쓸 수 있다.** 막히는 게 아니라 **열린다.**

★클러스터에서는 이 값을 **금고(Secrets Manager)에서 직접 읽는다.** `catchap-stt-worker-token`
하나뿐이고, 백엔드도 **같은 시크릿**을 읽어 헤더에 실어 보낸다. 그래서 회전할 때
**한 곳만 바꾸면 양쪽이 같이 따라온다.**

```
K8s Secret  stt-worker-secret   ★금고를 여는 열쇠 두 줄만
                                SECRETS_ACCESS_KEY · SECRETS_SECRET_KEY
금고        catchap-stt-worker-token   ★실제 값은 여기에만
```

⚠️★`secrets_loader.py` 호출은 `main.py` 의 `_TOKEN` 줄보다 **앞이어야 한다.**
뒤로 밀리면 `_TOKEN` 이 빈 값으로 굳고 인증이 통째로 꺼진다. CI 가 이 순서를
실제로 확인한다(가짜 로더를 끼워 `_TOKEN` 이 그 값을 받는지 본다).

## 보안
- 워커는 **백엔드에서만** 접근한다. Service 가 ClusterIP 라 클러스터 밖에서는 안 보이고,
  그 위에 공유 토큰이 한 겹 더 있다. Ingress 에 붙이지 않는다.
- 오디오는 임시 파일로 받고 전사 후 즉시 삭제한다(원본 미보관).
