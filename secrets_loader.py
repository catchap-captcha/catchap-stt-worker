# -*- coding: utf-8 -*-
"""기동 시 카카오클라우드 Secrets Manager에서 비밀값을 읽어 환경변수로 주입한다.

왜 이렇게 하는가 — 쿠버네티스로 옮기면 비밀값을 둘 수 있는 곳이 두 군데다.

    (가) 앱이 기동 시 Secrets Manager를 읽는다   K8s Secret에는 부트스트랩 키 1개만
    (나) 비밀값을 K8s Secret에 복사해 둔다        런타임 의존이 없지만 두 곳을 동기화해야 함

**(가)를 택했다**(`인프라-캡처/00-아키텍처-설계.md` 2-E). 이유는 두 가지다.

  1. 비밀값을 바꿀 때 K8s를 건드리지 않아도 된다. Secrets Manager에서 새 버전을 만들고
     파드를 재시작하면 끝이다. (나)였다면 매번 Secret 매니페스트도 같이 고쳐야 한다.
  2. K8s Secret은 이름과 달리 **암호화가 아니라 base64**다. etcd에 사실상 평문으로 들어가고,
     클러스터를 볼 수 있는 사람은 다 읽을 수 있다. Secrets Manager는 KMS(kms11)로 암호화되고
     누가 언제 읽었는지 감사 기록이 남는다.

`catchap-behavior-ai` 서비스 계정이 이 목적을 위해 존재한다 — `Secrets Manager 매니저` + `KMS 사용자`
조합이고 kms11 접근 명단에 들어 있다(2026-08-10 신설·복호화 실측 확인).
★역할은 계정을 만든 뒤에 바꿀 수 없다. 이 계정을 재생성하면 kms11 명단에서 빠져
  행동AI 앱이 기동하지 못한다.

★이 파일은 `catchap-backend/app/core/secrets_loader.py` 와 ★같은 코드다.
  새 방식을 만들지 않고 이미 운영에서 도는 것을 그대로 옮겼다.

──────────────────────────────────────────────────────────────────────────
동작
──────────────────────────────────────────────────────────────────────────
`SECRETS_BACKEND` 가 `kakaocloud` 일 때만 동작한다. **기본값은 `none`** 이라
로컬 개발·테스트·기존 배포는 이 파일이 없는 것과 똑같이 동작한다.

    SECRETS_BACKEND=none         (기본) 아무것도 하지 않는다
    SECRETS_BACKEND=kakaocloud   기동 시 읽어서 os.environ 에 넣는다

★이 모듈은 `Settings` 를 쓰지 않고 `os.environ` 을 직접 읽는다. Settings 를 만들기
  **전에** 실행돼야 하기 때문이다(비밀값이 Settings 의 재료다 — 닭과 달걀).

★실패하면 예외를 낸다. 조용히 .env 로 떨어지지 않는다 — 그러면 설정 누락이
  "비밀번호가 틀렸습니다" 같은 엉뚱한 증상으로 둔갑해서 원인을 못 찾는다.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger("catchap.secrets")

# ★기동 순서 때문에 이 모듈의 로그는 화면에 안 나온다 —
#   main.py 는 setup_logging() 을 먼저 부르는데, setup_logging() 자신이
#   get_settings() 를 부르므로 ★로더는 dictConfig 가 적용되기 전에 끝난다.
#   그래서 결과를 여기 남겨 두고, 로그 설정이 끝난 뒤 main.py 가 찍는다.
#   ⚠️이게 없으면 제일 위험한 경우 — SECRETS_BACKEND 가 없어서 로더가
#   조용히 아무것도 안 한 경우 — 를 아무도 모른다.
_LAST: "LoadResult | None" = None


def last_result() -> "LoadResult | None":
    """직전 load_secrets_into_env() 의 결과. 한 번도 안 불렀으면 None."""
    return _LAST

_IAM_DEFAULT = "https://iam.kakaocloud.com/identity/v3"
_SM_DEFAULT = "https://secrets-manager-service.kr-central-2.kakaocloud.com"
_TIMEOUT = 15

# 주입을 허용하는 이름 — 대문자로 시작하는 환경변수 형식만.
# ★`$` 는 끝의 개행도 허용한다 — 개행이 붙은 이름이 ★통과했다(0810 실측).
# 만들어지는 이름이 달라 실제 가로채기는 안 되지만, 규칙은 규칙대로 못 박는다.
_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*\Z")

# ★프로세스 동작 자체를 바꾸는 이름은 시크릿에 들어 있어도 주입하지 않는다.
# Secrets Manager를 쓸 수 있는 사람이 앱의 실행 방식까지 바꿀 수 있으면 안 된다
# (예: LD_PRELOAD로 임의 코드 주입, PATH로 다른 바이너리 실행).
_FORBIDDEN = frozenset({
    "PATH", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONWARNINGS",
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "DYLD_INSERT_LIBRARIES",
    "HOME", "SHELL", "IFS", "BASH_ENV", "ENV",
    # 로더 자신의 설정 — 시크릿이 자기 출처를 바꾸지 못하게 한다
    "SECRETS_BACKEND", "SECRETS_NAMES", "SECRETS_ACCESS_KEY", "SECRETS_SECRET_KEY",
    "SECRETS_ENDPOINT", "SECRETS_IAM_ENDPOINT", "SECRETS_REQUIRED_VARS",
})


class SecretsLoadError(RuntimeError):
    """Secrets Manager를 쓰기로 했는데 못 읽었다. 기동을 막는다."""


@dataclass
class LoadResult:
    """무엇이 주입됐는지. ★값은 담지 않는다 — 이름과 개수만."""

    backend: str
    loaded: list[str] = field(default_factory=list)      # 주입한 환경변수 이름
    secrets_read: list[str] = field(default_factory=list)  # 읽은 시크릿 이름
    skipped: list[str] = field(default_factory=list)     # 이름 규칙·금지 목록에 걸린 것

    def summary(self) -> str:
        if self.backend == "none":
            return "Secrets Manager 미사용 (SECRETS_BACKEND=none) — 환경변수·.env를 그대로 씁니다"
        parts = [
            f"Secrets Manager에서 시크릿 {len(self.secrets_read)}건을 읽어 "
            f"환경변수 {len(self.loaded)}개를 주입했습니다",
            f"  시크릿: {', '.join(self.secrets_read) or '(없음)'}",
            f"  변수  : {', '.join(self.loaded) or '(없음)'}",
        ]
        if self.skipped:
            parts.append(f"  ★주입하지 않음: {', '.join(self.skipped)}")
        return "\n".join(parts)


# ── ★429 재시도 (2026-08-11 신설)
#
# 왜 — 0811 에 행동AI·캡차·백엔드를 잇달아 재시작했더니 금고 API 가 ★429(요청이
#   너무 잦음)를 줬다. 그때는 uvicorn 이 워커를 다시 띄워 ★우연히 회복했다.
#   하지만 ★노드 한 대가 죽어 파드가 한꺼번에 뜨면 기동이 통째로 실패한다.
#   기다렸다 다시 묻는 것 말고 방법이 없다.
#
# ⚠️★기동 예산 안에서만 기다린다. 로더는 앱이 뜨기 ★전에 도는 코드라
#   여기서 오래 끌면 startupProbe 가 파드를 죽인다(백엔드는 ★최대 90초다).
#   그래서 ★요청마다가 아니라 ★기동 한 번당 총 재시도 시간을 묶는다.
#   예산을 다 쓰면 ★기다리지 않고 바로 실패한다 — 더 끌면 원인이 안 보인다.
#   실측: 로더 전체가 정상일 때 ★1.2~1.7초. 예산 20초를 더해도 여유가 크다.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRY_BUDGET_SEC = 20.0     # 기동 한 번이 재시도에 쓸 수 있는 총합
_RETRY_BASE_SEC = 0.4        # 0.4 → 0.8 → 1.6 → 3.2 …
_RETRY_MAX_SLEEP = 5.0
_retry_left = 0.0            # load_secrets_into_env() 가 예산을 채운다


def _retry_wait(attempt: int, retry_after: str | None) -> float:
    """다음 시도까지 기다릴 시간. ★서버가 Retry-After 를 주면 그것을 따른다."""
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), _RETRY_MAX_SLEEP)
        except ValueError:
            pass
    # ★지터를 넣는다 — 파드가 한꺼번에 뜨면 재시도도 같은 순간에 몰려
    #   429 가 그대로 되풀이된다.
    base = min(_RETRY_BASE_SEC * (2 ** attempt), _RETRY_MAX_SLEEP)
    return base * (1.0 + random.random() * 0.5)


def _http(url: str, *, method: str = "GET", body: dict | None = None,
          headers: dict | None = None):
    global _retry_left
    safe = url.split("?")[0]
    attempt = 0
    while True:
        req = urllib.request.Request(
            url, method=method,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers=headers or {},
        )
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            return urllib.request.urlopen(req, timeout=_TIMEOUT)
        except urllib.error.HTTPError as e:
            # ★본문을 그대로 넣지 않는다 — 오류 응답에 값이 섞여 로그로 샐 수 있다.
            retryable = e.code in _RETRY_STATUS
            after = e.headers.get("Retry-After") if e.headers else None
            failure = SecretsLoadError(f"{method} {safe} → HTTP {e.code}")
        except urllib.error.URLError as e:
            # 연결 자체가 안 된 경우 — 네트워크가 늦는 중일 수 있어 다시 시도한다.
            retryable = True
            after = None
            failure = SecretsLoadError(f"{method} {safe} → 연결 실패 ({e.reason})")

        wait = _retry_wait(attempt, after)
        if not retryable or wait > _retry_left:
            raise failure from None
        _retry_left -= wait
        attempt += 1
        logger.warning(
            "[SECRETS] %s %s 실패 — %.1f초 뒤 다시 시도 (남은 재시도 예산 %.1f초)",
            method, safe, wait, _retry_left,
        )
        time.sleep(wait)


def _token(iam_endpoint: str, access_key: str, secret_key: str) -> str:
    """서비스 계정 키(application_credential)로 IAM 토큰을 받는다."""
    resp = _http(
        f"{iam_endpoint.rstrip('/')}/auth/tokens",
        method="POST",
        body={"auth": {"identity": {
            "methods": ["application_credential"],
            "application_credential": {"id": access_key, "secret": secret_key},
        }}},
    )
    token = resp.headers.get("X-Subject-Token")
    if not token:
        raise SecretsLoadError("IAM 응답에 X-Subject-Token이 없습니다")
    return token


def _unwrap(payload: dict) -> dict:
    """시크릿 값 응답에서 {변수이름: 값} 을 꺼낸다.

    실제 응답 모양 (0731 확인):

        {"version": {"secret": {"JWT_SECRET_KEY": "..."}}}

    ★키가 이미 환경변수 이름이라 별도 매핑표가 필요 없다. 시크릿을 만들 때
      그 규칙으로 넣기로 했다(`00-아키텍처-설계.md` 2-E).
    """
    node = payload
    for key in ("version", "secret"):
        if not isinstance(node, dict) or key not in node:
            raise SecretsLoadError(
                f"시크릿 응답 모양이 예상과 다릅니다 (기대: version.secret, 실제 키: "
                f"{sorted(node.keys()) if isinstance(node, dict) else type(node).__name__})"
            )
        node = node[key]
    if isinstance(node, str):          # 문자열로 한 번 더 감싸 오는 경우 대비
        node = json.loads(node)
    if not isinstance(node, dict):
        raise SecretsLoadError(f"시크릿 값이 사전이 아닙니다 ({type(node).__name__})")
    return node


def _remember(result: LoadResult) -> LoadResult:
    global _LAST
    _LAST = result
    return result


def _list_secrets(sm: str, headers: dict) -> list[dict]:
    """시크릿 목록을 ★끝까지 가져온다.

    ⚠️★★이 API 는 한 번에 ★10건만 준다 (2026-08-07 실측).

        {"pagination": {"offset": 0, "limit": 10, "total": 11}, "secrets": [...10건...]}

    첫 장만 읽으면 11번째 시크릿이 "없는 것"이 되어 ★기동이 막힌다.
    실제로 그렇게 됐다 — 시크릿을 8개에서 11개로 늘린 날 바로 이 선을 넘었고,
    `catchap-portone-keys` 가 목록에서 빠져 "찾을 수 없습니다" 로 죽을 상태였다.
    (마침 파드가 안 뜨는 시점이라 서비스는 안 끊겼지만, ★재시작 한 번이면 전부 죽었다.)

    ★`limit` 을 크게 줘도 되지만 그것만 믿지 않는다 — 서버가 조용히 잘라도 알 수 없다.
      `pagination.total` 에 닿을 때까지 `offset` 을 넘겨서 ★센 개수로 확인한다.
    """
    out: list[dict] = []
    seen: set = set()
    offset = 0
    for _ in range(50):                      # 무한 루프 방지 (100건×50 = 5000건까지)
        body = json.load(_http(f"{sm}/api/v1/secrets?limit=100&offset={offset}", headers=headers))
        if isinstance(body, list):           # 페이지 정보 없이 배열만 주는 경우 대비
            page, total = body, len(body)
        else:
            page = body.get("secrets") or []
            total = (body.get("pagination") or {}).get("total")
        if not isinstance(page, list) or not page:
            break
        for item in page:
            name = item.get("name") if isinstance(item, dict) else None
            if name and name not in seen:
                seen.add(name)
                out.append(item)
        offset += len(page)
        if not isinstance(total, int) or offset >= total:
            break
    return out


def load_secrets_into_env(environ: dict | None = None) -> LoadResult:
    """Secrets Manager를 읽어 환경변수로 주입한다. 기본값(none)이면 아무것도 안 한다."""
    env = os.environ if environ is None else environ
    backend = (env.get("SECRETS_BACKEND") or "none").strip().lower()
    if backend in ("", "none", "off", "false", "0"):
        return _remember(LoadResult(backend="none"))
    if backend != "kakaocloud":
        raise SecretsLoadError(
            f"SECRETS_BACKEND 값이 올바르지 않습니다: {backend!r} (none | kakaocloud)"
        )

    # ★재시도 예산을 기동마다 다시 채운다. ★전역이라 여기서 초기화하지 않으면
    #   한 번 쓴 예산이 다음 호출에 그대로 남는다.
    global _retry_left
    _retry_left = _RETRY_BUDGET_SEC

    access_key = (env.get("SECRETS_ACCESS_KEY") or "").strip()
    secret_key = (env.get("SECRETS_SECRET_KEY") or "").strip()
    # ★이름 목록을 필수로 둔다. 프로젝트의 시크릿을 전부 긁어 오면 다른 사람이 만든 것까지
    #   읽으려 들고(접근 제어가 걸린 것은 실패), 나중에 시크릿이 하나 늘었을 때 앱 동작이
    #   말없이 바뀐다. 무엇을 읽을지는 배포가 명시한다.
    names = [n.strip() for n in (env.get("SECRETS_NAMES") or "").split(",") if n.strip()]
    missing = [k for k, v in (("SECRETS_ACCESS_KEY", access_key),
                              ("SECRETS_SECRET_KEY", secret_key),
                              ("SECRETS_NAMES", names)) if not v]
    if missing:
        raise SecretsLoadError(
            "SECRETS_BACKEND=kakaocloud 인데 다음이 비어 있습니다: " + ", ".join(missing)
        )

    iam = (env.get("SECRETS_IAM_ENDPOINT") or _IAM_DEFAULT).rstrip("/")
    sm = (env.get("SECRETS_ENDPOINT") or _SM_DEFAULT).rstrip("/")
    headers = {"X-Auth-Token": _token(iam, access_key, secret_key)}

    catalog = _list_secrets(sm, headers)
    by_name = {s.get("name"): s for s in catalog if isinstance(s, dict)}

    unknown = [n for n in names if n not in by_name]
    if unknown:
        # ★목록을 몇 건 봤는지 같이 말한다 — 페이지가 잘렸는지 바로 알 수 있게.
        raise SecretsLoadError(
            "SECRETS_NAMES에 있는 시크릿을 찾을 수 없습니다: " + ", ".join(unknown)
            + f" (목록 {len(by_name)}건을 확인했습니다)"
        )

    result = LoadResult(backend=backend)
    for name in names:
        meta = by_name[name]
        sid = meta.get("id")
        version = meta.get("default_version") or meta.get("version")
        if not sid or "*" in str(sid):
            # 접근 제어가 걸린 시크릿은 ID가 마스킹돼 요청이 성립하지 않는다.
            raise SecretsLoadError(
                f"시크릿 {name!r}의 ID가 마스킹돼 있습니다 — 이 계정이 접근 명단에 없습니다"
            )
        payload = json.load(
            _http(f"{sm}/api/v1/secrets/{sid}/versions/{version}/value", headers=headers)
        )
        result.secrets_read.append(name)
        for var, value in _unwrap(payload).items():
            if not _NAME_RE.match(var) or var in _FORBIDDEN:
                result.skipped.append(var)
                continue
            env[var] = str(value)
            result.loaded.append(var)

    if not result.loaded:
        raise SecretsLoadError(
            f"시크릿 {len(names)}건을 읽었지만 주입할 변수가 하나도 없습니다"
        )

    # ★★★부분 실패를 잡는다 — 위의 검사는 ★전체 합계가 0일 때만 걸린다.
    #
    # 시크릿 6건 중 하나에서 키 이름이 바뀌거나 빠지면, 나머지 5건이 성공했으므로
    # 위 검사는 통과한다. 그러면 그 변수만 없는 채로 앱이 ★조용히 뜬다.
    # 0810 에 실제로 재현했다 — `DB_USER` 가 `DB_USERNAME` 으로 잘못 들어간 시크릿을
    # 흉내 냈더니 예외 없이 통과했고, 캡차가 dataclass 기본값(당시 `catchap_dba`)으로
    # DB 에 붙으러 갔다.
    #
    # ★그래서 「반드시 있어야 하는 변수」를 배포가 명시하게 한다. `SECRETS_NAMES` 와
    #   같은 철학이다 — 무엇을 기대하는지는 코드가 아니라 배포가 안다.
    # ★비워 두면(기본값) 아무 검사도 안 한다 = 이 코드가 없는 것과 같다.
    required = [v.strip() for v in (env.get("SECRETS_REQUIRED_VARS") or "").split(",") if v.strip()]
    if required:
        got = set(result.loaded)
        missing = [v for v in required if v not in got]
        if missing:
            raise SecretsLoadError(
                "SECRETS_REQUIRED_VARS 에 적힌 변수가 시크릿에 없습니다: "
                + ", ".join(missing)
                + f" (시크릿 {len(result.secrets_read)}건에서 변수 {len(result.loaded)}개를 받았습니다)"
            )

    logger.info("%s", result.summary())
    return _remember(result)
