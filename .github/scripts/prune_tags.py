#!/usr/bin/env python3
"""이미지 태그가 너무 쌓이면 ★알려 준다. (지우지는 못한다)

★왜 「지우기」가 아니라 「알리기」인가 — 2026-08-12 에 확인했다.

  처음에는 자동으로 지우려고 만들었다. 그런데 실제로 돌려 보니 전부 401 이었다.

      catchap-ops (프로젝트 리더)  → 토큰이 주는 권한: pull
      catchap-ci  (프로젝트 멤버)  → 토큰이 주는 권한: pull, push
      ★delete 를 요청해도 안 준다.

  공식 문서도 같은 말을 한다.
      · 이미지·태그 삭제는 ★콘솔로만 (API·CLI 방법이 문서에 없다)
        https://docs.kakaocloud.com/en/service/container-pack/cr/how-to-guides/cr-manage-image
      · Container Registry 는 ★OpenAPI 목록에 아예 없다
        https://docs.kakaocloud.com/en/openapi
      · 자동 정리·보존정책 기능도 ★없다
      · 「만료기한」은 정리가 아니라 ★그날부터 Push·Pull 을 막는 스위치다

  ★IAM 역할 25개에 Container Registry 역할이 아예 없어서, 권한을 더 줄 방법도 없다.

  → 그래서 이 스크립트는 ★세어서 알리기만 한다. 지우는 것은 사람이 콘솔에서 한다.

★왜 그래도 만드나 — 안 세면 아무도 모른다.
  0812 에 손으로 세어 보니 프론트엔드만 57개, 다섯 이미지 합쳐 130개 남짓이었다.
  쌓이는 것을 아무도 몰랐던 것이 문제였다.

★나중에 카카오가 삭제 API 를 열면 TRY_DELETE=1 로 켜면 된다.
  그 경로도 남겨 두었다 — 다만 ★기본은 꺼져 있다.
"""
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

ACCEPT = ",".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])
# ★해시 태그만 센다. buildcache·latest 등은 이름 모양으로 걸러진다.
HASH_TAG = re.compile(r"^[0-9a-f]{7,40}$")


def http(url, method="GET", headers=None, timeout=30):
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(url, method=method, headers=dict(headers or {})), timeout=timeout)
        return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except Exception as e:  # noqa: BLE001
        return 0, {}, str(e).encode()


def token_for(host, repo, user, password, actions="pull,delete"):
    _, hd, _ = http("https://%s/v2/" % host)
    chal = hd.get("Www-Authenticate") or hd.get("WWW-Authenticate") or ""
    m = re.search(r'realm="([^"]+)"', chal)
    s = re.search(r'service="([^"]+)"', chal)
    if not m:
        return None
    url = "%s?service=%s&scope=repository:%s:%s" % (
        m.group(1), urllib.parse.quote(s.group(1) if s else ""), repo, actions)
    basic = base64.b64encode(("%s:%s" % (user, password)).encode()).decode()
    st, _, body = http(url, headers={"Authorization": "Basic " + basic})
    if st != 200:
        return None
    try:
        d = json.loads(body)
        return d.get("token") or d.get("access_token")
    except Exception:  # noqa: BLE001
        return None


def notice(kind, title, line):
    print("::%s title=%s::%s" % (kind, title, line))
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n### %s\n\n%s\n" % (title, line))
        except Exception:  # noqa: BLE001
            pass


def main():
    host = os.environ.get("REGISTRY", "")
    repo = os.environ.get("IMAGE_PATH", "")
    user = os.environ.get("REGISTRY_USERNAME", "")
    pw = os.environ.get("REGISTRY_PASSWORD", "")
    limit = int(os.environ.get("TAG_LIMIT", "20"))
    try_delete = os.environ.get("TRY_DELETE", "").lower() in ("1", "true", "yes")

    if not (host and repo and user and pw):
        print("  설정이 모자라 점검을 건너뜁니다.")
        return 0

    tok = token_for(host, repo, user, pw)
    if not tok:
        print("  토큰을 못 받아 점검을 건너뜁니다.")
        return 0
    H = {"Authorization": "Bearer " + tok}

    st, _, body = http("https://%s/v2/%s/tags/list" % (host, repo), headers=H)
    if st != 200:
        print("  태그 목록을 못 읽었습니다 (HTTP %s)." % st)
        return 0
    tags = json.loads(body).get("tags") or []
    hashy = [t for t in tags if HASH_TAG.match(t)]
    other = sorted(set(tags) - set(hashy))
    print("  태그 %d개 (해시 %d · 그 외 %d)" % (len(tags), len(hashy), len(other)))
    if other:
        print("  세지 않는 것: %s" % ", ".join(other))

    if len(hashy) <= limit:
        print("  %d개 ≤ 기준 %d — 아직 괜찮습니다." % (len(hashy), limit))
        return 0

    line = ("**%s** 의 태그가 **%d개** 입니다 (기준 %d).\n\n"
            "카카오클라우드 Container Registry 는 **API 로 태그를 지울 수 없습니다** "
            "— 콘솔에서만 됩니다.\n\n"
            "```\n콘솔 → Container Registry → %s → 이미지 → 태그 탭\n"
            "     → 오래된 것 선택 → [태그 삭제] → \"영구 삭제\" 입력\n```\n\n"
            "⚠️**`buildcache` 는 지우지 마십시오** — 없으면 빌드가 매번 처음부터 돕니다.\n"
            "⚠️**지우기 직전에 클러스터가 쓰는 태그를 다시 확인**하십시오. "
            "0812 에 30분 전 목록으로 지웠다가 프론트엔드가 깨졌습니다."
            % (repo, len(hashy), limit, repo.split("/")[0]))
    notice("warning", "이미지 태그 정리가 필요합니다", line)

    if not try_delete:
        return 0

    # ── 아래는 ★카카오가 삭제 API 를 열면 쓸 경로. 기본은 꺼져 있다.
    print("  TRY_DELETE 가 켜져 있어 삭제를 시도합니다.")
    st, _, b = http("https://%s/v2/%s/manifests/%s" % (host, repo, hashy[-1]), "HEAD",
                    dict(H, Accept=ACCEPT))
    print("    (시험) HEAD %s → HTTP %s" % (hashy[-1], st))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        # ★점검이 실패해도 배포는 성공이어야 한다
        print("  점검 중 오류 (배포에는 영향 없음): %s" % e.__class__.__name__)
        sys.exit(0)
