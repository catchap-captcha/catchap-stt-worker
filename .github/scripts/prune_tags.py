#!/usr/bin/env python3
"""오래된 이미지 태그를 정리한다 — 최근 N개만 남긴다.

★왜 만들었나
  배포할 때마다 커밋 해시로 태그가 쌓이는데 지우는 곳이 없었다.
  2026-08-12 에 손으로 정리해 보니 프론트엔드만 57개였다.

★2026-08-12 사고에서 배운 것 (이 스크립트가 그걸 막는다)
  손으로 정리할 때 "지금 쓰는 태그"를 한 번 재고 30분 뒤에 그 목록으로 지웠다.
  그 사이 배포가 두 번 돌아 ★쓰는 태그가 바뀌었고, 그걸 지워서 프론트가 깨졌다.
  → 이 스크립트는 ★고정 목록을 쓰지 않는다. 매번 지금 목록을 읽어
    ★날짜순 최근 N개를 남긴다. 배포가 동시에 돌아도 최근 것은 안 지운다.

★안전장치
  · KEEP 개수만큼 최근 것을 남긴다 (기본 10)
  · 방금 올린 태그는 무조건 남긴다
  · buildcache · latest 처럼 ★해시가 아닌 태그는 건드리지 않는다
  · 날짜를 못 읽은 태그는 ★후보에서 뺀다 (모르면 안 지운다)
  · 실패해도 0 으로 끝난다 — ★정리 실패가 배포를 깨뜨리면 안 된다

⚠️★남은 위험 — KEEP 을 너무 작게 두지 말 것
  이 스크립트는 "최근 N개"를 남길 뿐, 클러스터가 지금 무엇을 쓰는지는 모른다.
  배포가 실패해 클러스터가 옛 태그에 머물러 있는데 그 뒤로 N개가 더 올라가면,
  ★쓰는 태그가 밀려나 지워질 수 있다.
  · 배포 직후에 도는 한 방금 올린 것이 최신이라 실제로는 안전하다
  · 그래도 KEEP 은 ★10 이상을 권한다. 태그 하나는 몇 KB 다 — 아낄 이유가 없다
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
# ★해시 태그만 정리 대상. buildcache·latest 등은 이름 모양으로 걸러진다.
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
    st, hd, _ = http("https://%s/v2/" % host)
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


def created_at(host, repo, tag, H):
    """이미지가 언제 만들어졌는지. 못 읽으면 None (그러면 안 지운다)."""
    st, hd, body = http("https://%s/v2/%s/manifests/%s" % (host, repo, tag),
                        headers=dict(H, Accept=ACCEPT))
    if st != 200:
        return None, None
    digest = hd.get("Docker-Content-Digest")
    try:
        man = json.loads(body)
    except Exception:  # noqa: BLE001
        return None, digest
    # 목록(index)이면 첫 항목을 따라간다
    if "manifests" in man and man["manifests"]:
        sub = man["manifests"][0].get("digest")
        st, hd2, body = http("https://%s/v2/%s/manifests/%s" % (host, repo, sub),
                             headers=dict(H, Accept=ACCEPT))
        if st != 200:
            return None, digest
        try:
            man = json.loads(body)
        except Exception:  # noqa: BLE001
            return None, digest
    cfg = (man.get("config") or {}).get("digest")
    if not cfg:
        return None, digest
    st, _, blob = http("https://%s/v2/%s/blobs/%s" % (host, repo, cfg), headers=H)
    if st != 200:
        return None, digest
    try:
        return json.loads(blob).get("created"), digest
    except Exception:  # noqa: BLE001
        return None, digest


def main():
    host = os.environ.get("REGISTRY", "")
    repo = os.environ.get("IMAGE_PATH", "")
    user = os.environ.get("REGISTRY_USERNAME", "")
    pw = os.environ.get("REGISTRY_PASSWORD", "")
    keep_n = int(os.environ.get("KEEP_TAGS", "10"))
    just_pushed = os.environ.get("PUSHED_TAG", "").strip()

    if not (host and repo and user and pw):
        print("  설정이 모자라 정리를 건너뜁니다.")
        return 0

    H0 = token_for(host, repo, user, pw)
    if not H0:
        print("  토큰을 못 받아 정리를 건너뜁니다.")
        return 0
    H = {"Authorization": "Bearer " + H0}

    st, _, body = http("https://%s/v2/%s/tags/list" % (host, repo), headers=H)
    if st != 200:
        print("  태그 목록을 못 읽어 정리를 건너뜁니다 (HTTP %s)." % st)
        return 0
    tags = (json.loads(body).get("tags") or [])
    print("  태그 %d개" % len(tags))

    # ★해시 모양이 아닌 것(buildcache·latest 등)은 아예 후보에서 뺀다
    cand = [t for t in tags if HASH_TAG.match(t) and t != just_pushed]
    skipped = [t for t in tags if t not in cand]
    if skipped:
        print("  건드리지 않음: %s" % ", ".join(sorted(skipped)))
    if len(cand) <= keep_n:
        print("  정리 대상 %d개 ≤ 남길 개수 %d — 할 일 없음." % (len(cand), keep_n))
        return 0

    dated = []
    for t in cand:
        c, dg = created_at(host, repo, t, H)
        if c and dg:
            dated.append((c, t, dg))
    if len(dated) <= keep_n:
        print("  날짜를 읽은 것이 %d개뿐이라 정리하지 않습니다." % len(dated))
        return 0

    dated.sort(reverse=True)                      # 최근 것이 앞
    keep = dated[:keep_n]
    drop = dated[keep_n:]
    print("  남김 %d개 (최근순) · 지움 %d개" % (len(keep), len(drop)))

    dry = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")
    if dry:
        print("  ★시험 모드 — 실제로 지우지 않습니다.")
        for c, t, _ in drop:
            print("    (지울 것) %s  %s" % (t, c[:19]))
        return 0

    ok = 0
    for c, t, dg in drop:
        st, _, b = http("https://%s/v2/%s/manifests/%s" % (host, repo, dg), "DELETE", H)
        if st in (200, 202):
            ok += 1
            print("    지움 %s (%s)" % (t, c[:10]))
        else:
            print("    ★못 지움 %s → HTTP %s %s" % (t, st, b.decode("utf-8", "replace")[:80]))
    print("  정리 끝 — %d/%d" % (ok, len(drop)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        # ★정리가 실패해도 배포는 성공이어야 한다
        print("  정리 중 오류 (배포에는 영향 없음): %s" % e.__class__.__name__)
        sys.exit(0)
