"""faster-whisper STT 워커 — 강의 오디오를 타임스탬프 자막으로 전사(자체 호스팅 GPU).

왜 이게 있나(팀 학습용): 앱 백엔드가 자막 없는 강의로 AI 문항을 생성할 때 음성 전사가
필요한데, 지금은 OpenAI Whisper API(유료 분당 $0.006·25MB 한계·오디오 외부 전송)를 쓴다.
우리에겐 놀고 있는 Tesla T4 GPU 서버가 있으니, faster-whisper를 여기 올려 전사를 **자사에서
무료로**(과금 0·대용량 OK·오디오 외부 유출 없음) 처리한다. 로컬 스파이크(spike-whisper)에서
faster-whisper large-v3가 잘 되는 걸 이미 검증했고, 그 코드를 이 워커로 이식한 것이다.

구조: 백엔드만 이 워커를 부른다(공유 토큰 X-Worker-Token으로 인증). 워커는 오직 '오디오→자막'
한 가지만 안다(사용자·DB·시험을 모름). 반환 형태는 기존 stt_client.transcribe_video와 동일한
[{start, end, text}]라, 백엔드는 OpenAI든 이 워커든 같은 코드로 결과를 받는다.

공부 키워드: faster-whisper(CTranslate2 기반 Whisper 추론 — CUDA 런타임 wheel 내장이라
nvcc 불필요), VAD(voice activity detection — 무음 구간 건너뛰어 속도·품질↑), 워커/잡 분리
(무거운 작업을 본체에서 떼어 전용 서비스로).
"""
import os
import tempfile

from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from secrets_loader import load_secrets_into_env

# ★★아래 os.environ.get 들보다 ★먼저 금고를 읽어 os.environ 에 넣는다. 자리가 중요하다.
#
#   이 파일은 모듈을 불러오는 ★그 순간 환경변수를 읽어 상수에 담는다. 로더가 그 뒤에
#   돌면 _TOKEN 은 이미 빈 문자열로 굳어 있고, 빈 값이면 아래 transcribe 가 인증을
#   ★통째로 건너뛴다 — 클러스터 안의 아무 파드나 이 워커로 GPU 를 쓸 수 있게 된다.
#   ★막히는 게 아니라 ★열린다. 그래서 조용히 지나간다.
#
#   ⚠️캡차에서 같은 함정을 겪었다(0810) — dataclass 기본값이 class 문 시점에 평가돼서,
#     로더를 settings = Settings() 앞에 두었더니 여전히 옛 값을 읽고 있었다.
#
#   ★이 순서가 뒤집히지 않는지는 CI 가 실제로 확인한다(.github/workflows/ci.yml).
#   ★SECRETS_BACKEND 가 없으면 로더는 아무것도 안 한다 — 로컬·CI 는 그대로 돈다.
load_secrets_into_env()

# 모델·디바이스는 환경변수로 — GPU(T4)면 large-v3/float16, 없으면 CPU(int8) 폴백 가능.
_MODEL_SIZE = os.environ.get("STT_MODEL", "large-v3")
_DEVICE = os.environ.get("STT_DEVICE", "cuda")
_COMPUTE = os.environ.get("STT_COMPUTE", "float16")
# 백엔드만 부르게 하는 공유 토큰(빈 값이면 인증 생략 — 로컬/사내망 전용 기동 시).
_TOKEN = os.environ.get("STT_WORKER_TOKEN", "")

app = FastAPI(title="CatChap STT Worker (faster-whisper)")

# 모델은 무거우니(수 GB) 프로세스당 1회만 로드해 재사용(첫 요청에서 lazy 로드).
# faster_whisper import도 이 안에서 — /health는 모델·라이브러리 없이도 뜨고, HTTP 계층을
# 테스트에서 stub할 수 있다(무거운 CUDA 의존을 import 시점에 강제하지 않는다).
_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(_MODEL_SIZE, device=_DEVICE, compute_type=_COMPUTE)
    return _model


@app.get("/health")
def health() -> dict:
    """기동 확인용 — 모델은 로드하지 않는다(가벼운 헬스체크)."""
    return {"ok": True, "model": _MODEL_SIZE, "device": _DEVICE, "loaded": _model is not None}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = "ko",
    x_worker_token: str = Header(default=""),
) -> dict:
    """영상/오디오 → 전사 세그먼트 [{start, end, text}] (초 단위·시간순).

    stt_client.transcribe_video(OpenAI)와 동일한 반환 형태 — 백엔드가 소스에 상관없이 같은
    코드로 처리한다. 세그먼트가 하나도 없으면(무음 등) 422 — 빈 자막을 성공으로 위장하지 않는다."""
    if _TOKEN and x_worker_token != _TOKEN:
        raise HTTPException(status_code=401, detail="invalid worker token")

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    # 임시 파일로 받아 전사(faster-whisper는 파일 경로를 받는다 — mp4/webm 컨테이너 그대로 OK).
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        segments, info = _get_model().transcribe(path, language=language, vad_filter=True)
        out: list[dict] = []
        for s in segments:  # 제너레이터 — 여기서 실제 전사가 진행된다
            text = (s.text or "").strip()
            if not text:
                continue
            start = max(0.0, float(s.start))
            end = max(start, float(s.end))
            out.append({"start": round(start, 2), "end": round(end, 2), "text": text})
        if not out:
            raise HTTPException(status_code=422, detail="no speech segments (무음이거나 전사 실패)")
        return {
            "segments": out,
            "duration": round(float(info.duration), 1),
            "language": info.language,
        }
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
