from __future__ import annotations

import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import io
import json
import os
import time
import subprocess
import threading
from typing import Any, Literal
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from urllib.parse import urlsplit

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
PREFERRED_MODELS = ["qwen3-vl:8b-instruct", "qwen3-vl:8b"]
_runtime_lock = threading.Lock()
_runtime_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_status_cache: tuple[float, dict[str, Any]] | None = None
_image_cache: OrderedDict[tuple[str, ...], str] = OrderedDict()
_http_lock = threading.Lock()
_http_connections: dict[str, http.client.HTTPConnection] = {}
_MAX_IMAGE_SIDE = int(os.getenv("QWEN_IMAGE_MAX_SIDE", "1536"))
_IMAGE_CACHE_LIMIT = int(os.getenv("QWEN_IMAGE_CACHE_LIMIT", "256"))
_PREP_WORKERS = max(1, int(os.getenv("QWEN_PREP_WORKERS", "2")))
_NUM_CTX = int(os.getenv("QWEN_NUM_CTX", "8192"))
_NUM_PREDICT = int(os.getenv("QWEN_NUM_PREDICT", "384"))
_FINAL_NUM_PREDICT = int(os.getenv("QWEN_FINAL_NUM_PREDICT", "180"))
_KEEP_ALIVE = os.getenv("QWEN_KEEP_ALIVE", "-1").strip()
PROMPT_VERSION = "qwen_visual_v4"
SCHEMA_VERSION = "qwen_structured_v3"


def _request(path: str, payload: dict[str, Any] | None = None, timeout: int = 8) -> Any:
    """Call Ollama over a small keep-alive connection pool.

    The production worker is intentionally single-flight, so one persistent
    connection is enough and avoids rebuilding a TCP connection for every
    image pair.
    """
    parsed = urlsplit(OLLAMA_URL)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(f"无效的 Ollama 地址：{OLLAMA_URL}")
    host_key = f"{parsed.scheme}://{parsed.netloc}"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    target = f"{parsed.path.rstrip('/')}{path}" or path
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    with _http_lock:
        conn = _http_connections.get(host_key)
        if conn is None:
            cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
            conn = cls(parsed.hostname, port, timeout=timeout)
            _http_connections[host_key] = conn
        try:
            conn.timeout = timeout
            # Changing HTTPConnection.timeout after connect does not update the
            # timeout on its already-open socket. Propagate it so long-running
            # vision requests can exceed the short status-probe timeout.
            if conn.sock is not None:
                conn.sock.settimeout(timeout)
            conn.request("POST" if payload is not None else "GET", target, body=data, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            if response.status >= 400:
                raise RuntimeError(f"Ollama HTTP {response.status}: {raw.decode('utf-8', errors='replace')[-1000:]}")
            return json.loads(raw.decode("utf-8"))
        except Exception:
            try:
                conn.close()
            finally:
                _http_connections.pop(host_key, None)
            raise


def list_models() -> list[dict[str, Any]]:
    return _request("/api/tags").get("models", [])


def is_vision_model(name: str) -> bool:
    lowered = name.lower()
    return "qwen3-vl" in lowered or "qwen2.5-vl" in lowered or "qwen2-vl" in lowered


def status(force: bool = False) -> dict[str, Any]:
    global _status_cache
    now = time.monotonic()
    if not force and _status_cache and now - _status_cache[0] < 10:
        return dict(_status_cache[1])
    try:
        # Status probes must not queue behind a long vision inference request.
        parsed = urlsplit(OLLAMA_URL)
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        host = "127.0.0.1" if parsed.hostname == "localhost" else parsed.hostname
        conn = cls(host, parsed.port, timeout=1)
        try:
            conn.request("GET", f"{parsed.path.rstrip('/')}/api/tags")
            response = conn.getresponse()
            if response.status >= 400:
                raise RuntimeError(f"Ollama HTTP {response.status}")
            models = json.loads(response.read().decode("utf-8")).get("models", [])
        finally:
            conn.close()
    except Exception as exc:
        value = {"reachable": False, "ready": False, "error": str(exc), "models": []}
        _status_cache = (now, value)
        return dict(value)
    names = [m.get("name", "") for m in models]
    available = [n for n in names if is_vision_model(n)]
    selected = next((n for n in PREFERRED_MODELS if n in names), None) or (available[0] if available else None)
    value = {"reachable": True, "ready": bool(selected), "selected_model": selected, "models": models, "vision_models": available, "preferred_model": PREFERRED_MODELS[0]}
    _status_cache = (now, value)
    return dict(value)


def _runtime(model: str) -> dict[str, Any]:
    """Cache tags/show for a batch; status probing must not run per image pair."""
    now = time.monotonic()
    with _runtime_lock:
        cached = _runtime_cache.get(model)
        if cached and now - cached[0] < 300:
            return cached[1]
    tags = list_models()
    names = [m.get("name", "") for m in tags]
    if model not in names:
        raise RuntimeError(f"模型未加载：{model}")
    details = _request("/api/show", {"model": model}).get("capabilities", [])
    if "vision" not in details:
        raise ValueError(f"模型不支持 Vision：{model}")
    value = {"model": model, "capabilities": details}
    with _runtime_lock:
        _runtime_cache[model] = (now, value)
    return value


def pull(model: str = PREFERRED_MODELS[0]) -> dict[str, Any]:
    if not is_vision_model(model):
        raise ValueError("只能拉取支持图片输入的 Qwen-VL 模型")
    proc = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, timeout=3600)
    output = (proc.stdout or "") + (proc.stderr or "")
    return {"ok": proc.returncode == 0, "model": model, "output": output[-12000:], "returncode": proc.returncode, "status": status(force=True) if proc.returncode == 0 else None}


def _image_b64(data: str) -> str:
    if data.startswith("data:"):
        raw = base64.b64decode(data.split(",", 1)[1])
        key = ("data", hashlib.sha256(raw).hexdigest(), str(_MAX_IMAGE_SIDE))
        with _runtime_lock:
            cached = _image_cache.get(key)
            if cached:
                _image_cache.move_to_end(key)
                return cached
        value = _optimized_image_b64(raw, "data")
        _remember_image(key, value)
        return value
    if os.path.exists(data):
        path = Path(data)
        stat = path.stat()
        key = (str(path.resolve()), str(stat.st_size), str(stat.st_mtime_ns), str(_MAX_IMAGE_SIDE))
        with _runtime_lock:
            cached = _image_cache.get(key)
        if cached:
            with _runtime_lock:
                _image_cache.move_to_end(key)
            return cached
        value = _optimized_image_b64(path.read_bytes(), str(path))
        _remember_image(key, value)
        return value
    return data


def _remember_image(key: tuple[str, ...], value: str) -> None:
    with _runtime_lock:
        _image_cache[key] = value
        _image_cache.move_to_end(key)
        while len(_image_cache) > _IMAGE_CACHE_LIMIT:
            _image_cache.popitem(last=False)


def preload_images(paths: list[str], workers: int | None = None) -> dict[str, Any]:
    """Prepare reusable main images without blocking the single GPU worker."""
    unique = list(dict.fromkeys(path for path in paths if path))
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers or _PREP_WORKERS, thread_name_prefix="qwen-image-prep") as pool:
        futures = {pool.submit(_image_b64, path): path for path in unique}
        for future, path in ((future, path) for future, path in futures.items()):
            try:
                future.result()
            except Exception as exc:
                failures.append({"path": path, "error": str(exc)})
    return {"prepared": len(unique) - len(failures), "failed": failures}


def pair_cache_key(query_sha: str, candidate_sha: str, category: str, model: str, target: str) -> str:
    value = ["pair", PROMPT_VERSION, SCHEMA_VERSION, _MAX_IMAGE_SIDE, model, category, target, query_sha, candidate_sha]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def final_cache_key(query_sha: str, candidate_shas: list[str], category: str, model: str, target: str,
                    scoring_context: dict[str, Any] | None = None) -> str:
    value = ["final", PROMPT_VERSION, SCHEMA_VERSION, _MAX_IMAGE_SIDE, model, category, target,
             query_sha, candidate_shas, scoring_context]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _optimized_image_b64(raw: bytes, source: str) -> str:
    """Downsample large inputs once; retain enough detail for logos and patterns."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("RGB")
            if max(image.size) > _MAX_IMAGE_SIDE:
                scale = _MAX_IMAGE_SIDE / max(image.size)
                image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=88, optimize=True)
            return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return base64.b64encode(raw).decode("ascii")


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Recognition(StrictOutput):
    description: str
    visible_details: list[str]
    uncertain_details: list[str]


class PairVerdict(StrictOutput):
    match: bool
    confidence: float = Field(ge=0, le=1, description="模型自报分数，未经校准，不是真实概率")
    evidence: list[str] = Field(max_length=2)
    conflicts: list[str] = Field(max_length=2)


class FinalVerdict(StrictOutput):
    selected: Literal["TOP1", "TOP2"]
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(max_length=3)
    conflicts: list[str] = Field(max_length=3)


OPTIONS = {"temperature": 0, "num_ctx": _NUM_CTX, "num_predict": _NUM_PREDICT}
RULES = "只比较服装本身，忽略模特、背景、姿势、光照和角度。检查颜色、印花、领口、袖口、号码字体描边、Logo和特殊设计；不可见细节不要编造，不能从肤色印花推断透明材质。confidence仅为模型自报分数，不是真实概率。"


class StructuredOutputError(ValueError):
    def __init__(self, raw, reason):
        super().__init__(f"Qwen结构化输出校验失败: {reason}")
        self.raw = raw


def generate(prompt: str, images: list[str], model: str | None = None,
             output_model: type[BaseModel] = Recognition,
             options: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = model or status().get("selected_model")
    if not selected:
        raise RuntimeError("未检测到可用的 Qwen-VL")
    _runtime(selected)
    request_options = dict(OPTIONS)
    request_options.update(options or {})
    keep_alive: int | str = int(_KEEP_ALIVE) if _KEEP_ALIVE.lstrip("-").isdigit() else _KEEP_ALIVE
    payload = {"model": selected, "prompt": prompt, "images": [_image_b64(i) for i in images],
               "stream": False, "format": output_model.model_json_schema(), "options": request_options,
               "keep_alive": keep_alive}
    started = time.perf_counter()
    response = _request("/api/generate", payload, timeout=900)
    elapsed = time.perf_counter() - started
    raw = response.get("response", "")
    if not response.get("done"):
        raise StructuredOutputError(raw, "推理未完整结束")
    try:
        parsed = output_model.model_validate_json(raw)
    except ValueError as exc:
        raise StructuredOutputError(raw, str(exc)) from exc
    return {"model": selected, "result": parsed.model_dump(), "raw": raw,
            "elapsed_seconds": elapsed, "response": {k:v for k,v in response.items() if k != "context"},
            "confidence_note": "模型自报值，未经校准，不是真实概率"}


def compare(query: str, candidate: str, target: str, candidate_index: int = 1,
            model: str | None = None) -> dict[str, Any]:
    if not target.strip():
        raise ValueError("必须明确目标商品或提供已裁剪的目标图")
    prompt = (f"IMAGE_1=QUERY；IMAGE_2=CANDIDATE_{candidate_index}。直接比较这两张原图，禁止交换角色。"
              f"目标：{target}。只返回简短JSON：match、confidence、最多2条evidence、最多2条conflicts。" + RULES)
    result = generate(prompt, [query, candidate], model, PairVerdict)
    # The application owns candidate identity; the model cannot renumber it.
    result["candidate_index"] = candidate_index
    return result


def rank_pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the fused score when present, otherwise preserve the Qwen-only ranking."""
    def verdict(row: dict[str, Any]) -> dict[str, Any]:
        value = row.get("result", row)
        return value if isinstance(value, dict) else {}

    def score(row: dict[str, Any]) -> float | None:
        value = row.get("fusion_score")
        if value is None:
            value = verdict(row).get("fusion_score")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    if any(score(row) is not None for row in rows):
        return sorted(rows, key=lambda row: (
            score(row) is None,
            -(score(row) if score(row) is not None else 0.0),
            int(row.get("candidate_index", verdict(row).get("candidate_index", 0))),
        ))

    def key(row: dict[str, Any]) -> tuple[bool, float, int]:
        result = verdict(row)
        matched = bool(result.get("match", False))
        confidence = float(result.get("confidence", 0.0))
        # For match=true, higher self-reported confidence ranks first. If all
        # candidates were rejected, prefer the rejection the model was least
        # certain about as the forced-choice fallback.
        confidence_key = -confidence if matched else confidence
        return (not matched, confidence_key, int(row.get("candidate_index", 0)))

    return sorted(rows, key=key)


def judge_top2(query: str, candidates: list[str], pairs: list[dict], target: str,
               model: str | None = None,
               scoring_context: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    top = rank_pairwise(pairs)[:2]
    if len(top) != 2:
        raise ValueError("Top2裁决需要至少两个成功完成的候选判断")
    ids = [r["candidate_index"] for r in top]
    score_by_id = {
        int(row.get("candidate_index", 0)): row
        for row in (scoring_context or [])
        if isinstance(row, dict) and row.get("candidate_index") is not None
    }
    score_lines = []
    for label, candidate_id in (("TOP1", ids[0]), ("TOP2", ids[1])):
        row = score_by_id.get(candidate_id)
        if row:
            score_lines.append(
                f"{label}候选{candidate_id}辅助分数：Qwen两图支持分={row.get('qwen_support_score')}，"
                f"Color分={row.get('color_score')}，LightGlue局部分={row.get('local_score')}，"
                f"LightGlue+Color分={row.get('lightglue_color_score')}，融合排序分={row.get('fusion_score')}。"
            )
    auxiliary = ("辅助数值只作为排序权重参考，不是概率，也不能代替直接看图；"
                 + "".join(score_lines)) if score_lines else ""
    prompt = (f"直接查看三张原图，不使用任何文字描述。IMAGE_1=QUERY；IMAGE_2=TOP1候选{ids[0]}；IMAGE_3=TOP2候选{ids[1]}。"
              f"目标是：{target}。{auxiliary}必须重新直接比较三张原图，并在TOP1和TOP2中选择更接近同款的一张，不可弃权；即使两张都存在差异，也选差异较少的一张并列出冲突。只返回JSON字段 selected, confidence, evidence(最多2条), conflicts(最多2条)。"
              + RULES)
    images = [query, candidates[ids[0]-1], candidates[ids[1]-1]]
    try:
        verdict = generate(prompt, images, model, FinalVerdict,
                           {"temperature": 0, "num_ctx": _NUM_CTX, "num_predict": _FINAL_NUM_PREDICT})
    except StructuredOutputError:
        retry_prompt = (f"直接看原图：IMAGE_1是QUERY，IMAGE_2是TOP1，IMAGE_3是TOP2。目标：{target}。{auxiliary}"
                        "必须二选一，不能输出NONE或无法判断；选更接近同款的一张。严格只输出：{\"selected\":\"TOP1\",\"confidence\":0.0,\"evidence\":[],\"conflicts\":[]}。"
                        "selected只能是TOP1或TOP2。")
        verdict = generate(retry_prompt, images, model, FinalVerdict,
                           {"temperature": 0, "num_ctx": _NUM_CTX, "num_predict": min(100, _FINAL_NUM_PREDICT)})
    selected = verdict["result"]["selected"]
    verdict["selected_candidate_index"] = ids[0 if selected == "TOP1" else 1]
    verdict["top2_candidate_indices"] = ids
    return verdict


def judge(query: str, candidates: list[str], target: str, model: str | None = None) -> dict[str, Any]:
    if len(candidates) != 5:
        raise ValueError("实验Judge需要恰好5张候选主图")
    start = time.perf_counter()
    pairs = [compare(query, c, target, i+1, model) for i,c in enumerate(candidates)]
    final = judge_top2(query, candidates, pairs, target, model)
    return {"result": {"pairs": pairs, "final": final, "total_seconds": time.perf_counter()-start,
                       "confidence_note": "模型自报分数仅用于实验排序，不是真实概率；需要人工确认"}}
