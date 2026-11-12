from __future__ import annotations

from dataclasses import dataclass, fields

HARD_MAX_CONCURRENCY = 4


def execution_fingerprint() -> str:
    """Audit scheduling revisions without invalidating unchanged vision cache keys."""
    import hashlib
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parent
    values={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
        ('execution.py','queue_store.py','concurrent_runner.py','provider_async.py')}
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ExecutionOptions:
    concurrency: int = 2
    max_concurrency: int = 4
    max_model_calls: int = 4
    max_retries: int = 2
    request_timeout: float = 120
    retry_backoff: float = 2.0
    retry_backoff_max: float = 60.0
    retry_jitter: float = 0.2
    poll_interval: float = 0.1
    progress_interval: float = 1.0

    @classmethod
    def from_settings(cls, settings: dict, **overrides) -> "ExecutionOptions":
        values = {field.name: settings[field.name] for field in fields(cls) if field.name in settings}
        if "request_timeout" not in values and settings.get("timeout_seconds") is not None:
            values["request_timeout"] = settings["timeout_seconds"]
        if "max_retries" not in values and settings.get("max_attempts_per_item") is not None:
            values["max_retries"] = settings["max_attempts_per_item"] - 1
        values.update({key: value for key, value in overrides.items() if value is not None})
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        for name in ("concurrency", "max_concurrency", "max_model_calls", "max_retries"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} 必须是整数。")
        if not 1 <= self.concurrency <= self.max_concurrency <= HARD_MAX_CONCURRENCY:
            raise ValueError("并发数必须为 1–4，且不能超过 max_concurrency。")
        if self.max_model_calls < 1 or not 0 <= self.max_retries <= 2:
            raise ValueError("调用预算必须大于 0，max_retries 必须为 0–2。")
        if not 0 < self.request_timeout <= 1800:
            raise ValueError("request_timeout 必须大于 0 且不超过 1800 秒。")
        if not 0 <= self.retry_backoff <= self.retry_backoff_max <= 300:
            raise ValueError("退避范围必须为 0–300 秒。")
        if not 0 <= self.retry_jitter <= 1 or self.poll_interval <= 0 or self.progress_interval <= 0:
            raise ValueError("抖动比例或轮询周期无效。")
