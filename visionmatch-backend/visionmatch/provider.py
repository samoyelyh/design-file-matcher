from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .schemas import PAIR_SCHEMA, REVIEW_SCHEMA, validate_output, write_schema


class ProviderError(RuntimeError):
    def __init__(self, kind: str, detail: str, *, stop: bool = False):
        super().__init__(detail)
        self.kind = kind
        self.stop = stop


def executable() -> str:
    override = os.environ.get("VISIONMATCH_CODEX")
    if override:
        path = Path(override).expanduser()
        if path.is_absolute() and path.is_file():
            return str(path.resolve())
        raise ProviderError("CODEX_MISSING", "VISIONMATCH_CODEX 必须指向有效的 Codex CLI 绝对文件路径。", stop=True)

    binary = shutil.which("codex")
    if binary:
        return binary

    # Logon tasks do not inherit the desktop app's temporary CLI PATH.
    # Discover the installed runtime on each call, so app updates can replace
    # versioned directories without leaving a hard-coded executable behind.
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        root = Path(local_app_data)
        standalone = root / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe"
        if standalone.is_file():
            return str(standalone.resolve())
        desktop_bin = root / "OpenAI" / "Codex" / "bin"
        versioned = sorted(
            (path for path in desktop_bin.glob("*/codex.exe") if path.is_file()),
            key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True,
        )
        for path in [*versioned, desktop_bin / "codex.exe"]:
            if path.is_file():
                return str(path.resolve())
    raise ProviderError("CODEX_MISSING", "找不到 Codex CLI；请检查安装，或用 VISIONMATCH_CODEX 指定 codex.exe 的绝对路径。", stop=True)


def safe_environment() -> dict[str, str]:
    env = os.environ.copy()
    # Do not pass API keys to Codex or silently fall back to API-key billing.
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
                 "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        env.pop(name, None)
    env["VISIONMATCH_CHILD"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def login_status() -> str:
    result = subprocess.run([executable(), "login", "status"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=20, env=safe_environment())
    status = (result.stdout + " " + result.stderr).strip()
    if result.returncode != 0 or "logged in using chatgpt" not in status.lower():
        raise ProviderError("AUTH", "Codex 当前不是 ChatGPT 登录状态；任务已停止。", stop=True)
    return "ChatGPT"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scrub(text: str) -> str:
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{10,}\b", "[REDACTED]", text)
    return text[-12000:]


def classify_error(text: str, returncode: int | None = None) -> tuple[str, bool]:
    lower = text.lower()
    if any(s in lower for s in ("usage limit", "usage_limit", "rate limit", "rate_limit", "quota", "额度", "insufficient_quota")):
        return "QUOTA", True
    if any(s in lower for s in ("unauthorized", "authentication", "not logged in", "access token", "401")):
        return "AUTH", True
    if any(s in lower for s in ("model_not_found", "model is not supported", "model is not available", "model_not_supported")):
        return "MODEL_UNAVAILABLE", True
    return "CLI_ERROR", False


def codex_args(*, workdir: Path, images: list[Path], schema_path: Path, output_path: Path,
               model: str, effort: str) -> list[str]:
    args = [executable(), "--no-daemon", "-a", "never", "exec", "--ignore-user-config", "--ignore-rules",
            "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only", "--json",
            "--disable", "shell_tool", "--disable", "unified_exec", "--disable", "multi_agent",
            "--disable", "apps", "--disable", "plugins", "--disable", "hooks", "--disable", "memories",
            "--disable", "skill_search", "--disable", "computer_use", "--disable", "browser_use",
            "--disable", "image_generation", "--disable", "unbounded_connection_retries",
            "-c", 'model_provider="openai"', "-c", 'web_search="disabled"',
            "-c", "agents.enabled=false", "-c", "project_doc_max_bytes=0",
            "-c", f'model_reasoning_effort="{effort}"',
            "--model", model, "--cd", str(workdir), "--output-schema", str(schema_path),
            "--output-last-message", str(output_path)]
    for image in images:
        args.extend(["--image", str(image)])
    args.append("-")
    return args


def terminate_process(process: subprocess.Popen) -> None:
    if os.name == "nt":
        # This targets only the process tree started by this particular request.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        import signal
        os.killpg(process.pid, signal.SIGKILL)


def invoke(request_root: Path, source_images: list[dict], prompt: str, kind: str,
           model: str, effort: str, timeout: int) -> dict:
    if os.environ.get("VISIONMATCH_CHILD") == "1":
        raise ProviderError("RECURSION_BLOCKED", "禁止在 Codex 子进程中递归启动匹配。", stop=True)
    schema = PAIR_SCHEMA if kind == "PAIR" else REVIEW_SCHEMA
    request_dir = request_root / uuid.uuid4().hex
    request_dir.mkdir(parents=True, exist_ok=False)
    staged = []
    for index, source_image in enumerate(source_images, 1):
        source = Path(source_image["path"])
        if not source.is_file() or file_hash(source) != source_image["sha256"]:
            raise ProviderError("SOURCE_CHANGED", "图片不存在或内容发生变化，必须重新扫描。", stop=True)
        dest = request_dir / f"image_{index}{source.suffix.lower()}"
        shutil.copyfile(source, dest)
        if file_hash(dest) != source_image["sha256"]:
            raise ProviderError("COPY_ERROR", "图片副本校验失败。", stop=True)
        staged.append(dest)
    schema_path = request_dir / "schema.json"
    output_path = request_dir / "result.json"
    write_schema(schema_path, schema)
    args = codex_args(workdir=request_dir, images=staged, schema_path=schema_path,
                      output_path=output_path, model=model, effort=effort)
    start = time.monotonic()
    try:
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   encoding="utf-8", errors="replace", env=safe_environment(),
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                   start_new_session=os.name != "nt")
    except OSError as exc:
        raise ProviderError("START_ERROR", scrub(str(exc)), stop=True) from exc
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        terminate_process(process)
        process.communicate()
        raise ProviderError("TIMEOUT_UNKNOWN", "请求超时；远端是否已执行和额度消耗未知。", stop=True) from exc
    except KeyboardInterrupt:
        terminate_process(process)
        process.communicate()
        raise
    elapsed = round(time.monotonic() - start, 3)
    events, usage, thread_id, tool_used = [], None, None, False
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        typ = event.get("type")
        if typ == "thread.started":
            thread_id = event.get("thread_id")
        if typ == "turn.completed":
            usage = event.get("usage")
        item = event.get("item") or {}
        if typ in ("item.started", "item.completed") and item.get("type") not in ("agent_message", "reasoning", None):
            tool_used = True
        # Persist only metadata, usage and error category; never store tool results or environment.
        events.append({"type": typ, "item_type": item.get("type"), "usage": event.get("usage")})
    (request_dir / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    if tool_used:
        raise ProviderError("UNEXPECTED_TOOL", "模型尝试使用工具；本次结果被丢弃并暂停。", stop=True)
    if process.returncode != 0:
        kind_name, stop = classify_error(stdout + "\n" + stderr, process.returncode)
        raise ProviderError(kind_name, scrub(stderr or stdout or f"退出码 {process.returncode}"), stop=stop)
    if not output_path.is_file():
        raise ProviderError("NO_OUTPUT", "Codex 未生成最终 JSON。")
    try:
        value = json.loads(output_path.read_text(encoding="utf-8-sig"))
        validate_output(value, schema)
    except (ValueError, OSError) as exc:
        raise ProviderError("INVALID_OUTPUT", scrub(str(exc))) from exc
    return {"result": value, "elapsed_seconds": elapsed, "usage": usage,
            "thread_id": thread_id, "request_dir": str(request_dir), "model": model}
