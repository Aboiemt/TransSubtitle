"""从 OpenAI 兼容上游拉取模型列表。"""

from __future__ import annotations

from openai import OpenAI


def normalize_base_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return u
    # 允许用户填 http://host:port 或 .../v1
    if not u.endswith("/v1"):
        if u.endswith("/v1/"):
            return u.rstrip("/")
        u = f"{u}/v1"
    return u


def fetch_remote_models(api_key: str, base_url: str, timeout: float = 20.0) -> list[str]:
    """
    GET {base_url}/models
    返回模型 id 列表（已排序）。
    """
    key = (api_key or "").strip()
    base = normalize_base_url(base_url)
    if not base:
        raise ValueError("请先填写 API 地址")
    if not key:
        raise ValueError("请先填写 API Key")

    client = OpenAI(api_key=key, base_url=base, timeout=timeout)
    resp = client.models.list()
    ids: list[str] = []
    for m in resp.data:
        mid = getattr(m, "id", None)
        if mid:
            ids.append(str(mid))
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    out.sort(key=str.lower)
    return out
