"""Async process lifecycle around the existing Codex CLI command and schemas."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable

from . import provider
from .schemas import PAIR_SCHEMA, REVIEW_SCHEMA, validate_output, write_schema


class RequestError(provider.ProviderError):
    def __init__(self, kind: str, detail: str, *, retryable: bool = False,
                 stop: bool = False, retry_after: float | None = None,
                 request_dir: str | None = None, started: bool = False):
        super().__init__(kind, detail, stop=stop)
        self.retryable = retryable
        self.retry_after = retry_after
        self.request_dir = request_dir
        self.started = started


def retry_after(text: str) -> float | None:
    numeric = re.search(r'(?i)retry[-_ ]after["\s:=]+([0-9]+(?:\.[0-9]+)?)', text)
    if numeric:
        return float(numeric.group(1))
    human = re.search(r'(?i)(?:try again|retry) in\s+([0-9.]+)\s*(seconds?|minutes?|s\b|m\b)', text)
    if human:
        return float(human.group(1)) * (60 if human.group(2).lower().startswith('m') else 1)
    date = re.search(r'(?im)^retry-after:\s*(.+)$', text)
    if date:
        try:
            return max(0, parsedate_to_datetime(date.group(1)).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            pass
    return None


def classify_error(text: str) -> RequestError:
    lower = text.lower()
    quota = ("insufficient_quota", "usage_limit_reached", "usage limit reached", "you've hit your usage limit",
             "you have reached your usage limit", "usage limit exceeded", "exceeded your current quota", "quota exhausted", "quota exceeded", "subscription limit",
             "额度耗尽", "额度不足", "用量已达上限", "使用额度已用尽")
    if any(value in lower for value in quota):
        return RequestError("QUOTA", provider.scrub(text), stop=True)
    if any(value in lower for value in ("unauthorized", "not logged in", "authentication", "401", "invalid token")):
        return RequestError("AUTH", provider.scrub(text), stop=True)
    if any(value in lower for value in ("model_not_found", "model_not_supported", "model is not available", "model is not supported")):
        return RequestError("MODEL_UNAVAILABLE", provider.scrub(text), stop=True)
    if any(value in lower for value in ("429", "rate limit", "rate_limit", "too many requests")):
        return RequestError("RATE_LIMIT", provider.scrub(text), retryable=True, retry_after=retry_after(text))
    if any(value in lower for value in ("503", "502", "500", "overloaded", "service busy", "temporar", "network",
                                       "connection", "stream disconnected", "error sending request", "timeout")):
        return RequestError("TRANSIENT", provider.scrub(text), retryable=True, retry_after=retry_after(text))
    return RequestError("CLI_ERROR", provider.scrub(text), retryable=True)


def prepare_request(request_root: Path, source_images: list[dict], prompt: str, kind: str,
                    model: str, effort: str) -> tuple[Path, Path, dict, list[str]]:
    if os.environ.get("VISIONMATCH_CHILD") == "1":
        raise RequestError("RECURSION_BLOCKED", "禁止递归启动匹配。", stop=True)
    if kind not in ("PAIR", "REVIEW"):
        raise ValueError("未知请求种类。")
    schema = PAIR_SCHEMA if kind == "PAIR" else REVIEW_SCHEMA
    folder = request_root / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    images = []
    for index, item in enumerate(source_images, 1):
        source = Path(item["path"])
        if not source.is_file() or provider.file_hash(source) != item["sha256"]:
            raise RequestError("SOURCE_CHANGED", "图片内容发生变化或已缺失。", stop=True, request_dir=str(folder))
        destination = folder / f"image_{index}{source.suffix.lower()}"
        shutil.copyfile(source, destination)
        if provider.file_hash(destination) != item["sha256"]:
            raise RequestError("COPY_ERROR", "图片副本校验失败。", stop=True, request_dir=str(folder))
        images.append(destination)
    schema_path, output_path = folder / "schema.json", folder / "result.json"
    write_schema(schema_path, schema)
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    args = provider.codex_args(workdir=folder, images=images, schema_path=schema_path,
                               output_path=output_path, model=model, effort=effort)
    return folder, output_path, schema, args


async def terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec("taskkill", "/PID", str(process.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            await asyncio.wait_for(killer.wait(), 10)
        except asyncio.TimeoutError:
            killer.kill()
            await killer.wait()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.wait()


def parse_response(folder: Path, output_path: Path, schema: dict, stdout: str, stderr: str,
                   returncode: int) -> tuple[dict, dict | None, str | None]:
    events, usage, thread_id, tool_used, turns = [], None, None, False, 0
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        kind, item = event.get("type"), event.get("item") or {}
        if kind == "thread.started":
            thread_id = event.get("thread_id")
        if kind == "turn.completed":
            usage = event.get("usage")
            turns += 1
        if kind in ("item.started", "item.completed") and item.get("type") not in ("agent_message", "reasoning", None):
            tool_used = True
        events.append({"type": kind, "item_type": item.get("type"), "usage": event.get("usage")})
    (folder / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    if tool_used or turns > 1:
        raise RequestError("UNEXPECTED_TOOL", "出现工具调用或多轮执行，已暂停。", stop=True)
    if returncode:
        raise classify_error(stdout + "\n" + stderr)
    if not output_path.is_file():
        error = classify_error(stdout + "\n" + stderr)
        if error.kind == "CLI_ERROR":
            error = RequestError("NO_OUTPUT", "未得到最终 JSON。", retryable=True)
        raise error
    try:
        value = json.loads(output_path.read_text(encoding="utf-8-sig"))
        validate_output(value, schema)
    except (ValueError, OSError) as exc:
        raise RequestError("INVALID_OUTPUT", provider.scrub(str(exc)), retryable=True) from exc
    if turns != 1:
        raise RequestError("INCOMPLETE_TURN", "未得到完整完成事件，结果未采用。", retryable=True)
    return value, usage, thread_id


async def invoke_async(request_root: Path, source_images: list[dict], prompt: str, kind: str,
                       model: str, effort: str, timeout: float, *,
                       on_started: Callable[[int], None] | None = None) -> dict:
    folder, output, schema, args = await asyncio.to_thread(prepare_request, request_root, source_images, prompt, kind, model, effort)
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    started = time.monotonic()
    launch = asyncio.create_task(asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=provider.safe_environment(),
        creationflags=flags, start_new_session=os.name != "nt"))
    try:
        process = await asyncio.shield(launch)
    except asyncio.CancelledError:
        try:
            process = await launch
            await terminate_process(process)
        finally:
            raise
    except OSError as exc:
        raise RequestError("START_ERROR", provider.scrub(str(exc)), stop=True, request_dir=str(folder)) from exc
    if on_started:
        try:
            on_started(process.pid)
        except BaseException:
            await terminate_process(process)
            await process.communicate()
            raise
    communication = asyncio.create_task(process.communicate(prompt.encode("utf-8")))
    try:
        remaining=max(0,timeout-(time.monotonic()-started))
        raw_stdout, raw_stderr = await asyncio.wait_for(asyncio.shield(communication), remaining)
    except asyncio.TimeoutError as exc:
        await terminate_process(process)
        await communication
        raise RequestError("TIMEOUT_UNKNOWN", "超时；远端执行及额度消耗可能已发生。", retryable=True,
                           request_dir=str(folder), started=True) from exc
    except asyncio.CancelledError:
        await terminate_process(process)
        await communication
        raise
    stdout, stderr = raw_stdout.decode("utf-8", errors="replace"), raw_stderr.decode("utf-8", errors="replace")
    try:
        result, usage, thread_id = parse_response(folder, output, schema, stdout, stderr, process.returncode)
    except RequestError as exc:
        exc.request_dir, exc.started = str(folder), True
        raise
    return {"result": result, "elapsed_seconds": round(time.monotonic() - started, 3), "usage": usage,
            "thread_id": thread_id, "request_dir": str(folder), "model": model, "cli_started": True}
