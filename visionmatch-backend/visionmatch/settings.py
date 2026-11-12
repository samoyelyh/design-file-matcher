from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    folder_a: str
    folder_b: str
    output_folder: str
    recursive: bool = False
    model: str = "gpt-6.1-terra"
    reasoning_effort: str = "medium"
    # Legacy names remain loadable so existing plans keep their exact cache keys.
    timeout_seconds: int | None = None
    max_attempts_per_item: int | None = None
    concurrency: int = 2
    max_concurrency: int = 4
    max_model_calls: int = 4
    max_retries: int = 2
    request_timeout: int = 120
    retry_backoff: float = 2.0
    retry_backoff_max: float = 60.0
    retry_jitter: float = 0.2
    review_margin: int = 10
    review_mode: str = "ambiguous"
    prompt_version: str = "pair-v1"
    rule_version: str = "apparel-v1"

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None:
            object.__setattr__(self, "request_timeout", self.timeout_seconds)
        else:
            object.__setattr__(self, "timeout_seconds", self.request_timeout)
        if self.max_attempts_per_item is not None:
            object.__setattr__(self, "max_retries", self.max_attempts_per_item - 1)
        else:
            object.__setattr__(self, "max_attempts_per_item", self.max_retries + 1)

    @classmethod
    def load(cls, path: Path) -> "Settings":
        with path.open(encoding="utf-8-sig") as source:
            settings = cls(**json.load(source))
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.model != "gpt-6.1-terra":
            raise ValueError("本项目只允许 gpt-6.1-terra。")
        if self.reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("无效的 reasoning_effort。")
        if not isinstance(self.recursive, bool):
            raise ValueError("recursive 必须为布尔值。")
        if not 1 <= self.timeout_seconds <= 1800:
            raise ValueError("timeout_seconds 必须在 1 到 1800 之间。")
        if not 1 <= self.max_attempts_per_item <= 3:
            raise ValueError("单个任务最多尝试 3 次。")
        from .execution import ExecutionOptions
        ExecutionOptions.from_settings(self.snapshot()).validate()
        if not 0 <= self.review_margin <= 100:
            raise ValueError("复核分差必须在 0 到 100 之间。")
        if self.review_mode not in {"all", "ambiguous"}:
            raise ValueError("review_mode 只能是 all 或 ambiguous。")
        a, b, output = (Path(p).resolve() for p in (self.folder_a, self.folder_b, self.output_folder))
        if not a.is_dir() or not b.is_dir():
            raise ValueError("输入目录 A 和 B 必须存在。")
        if a == b or a in b.parents or b in a.parents:
            raise ValueError("输入目录 A 和 B 不能相同或互相包含。")
        if output == a or output == b or a in output.parents or b in output.parents:
            raise ValueError("输出目录不能位于输入目录内。")
        if output in a.parents or output in b.parents:
            raise ValueError("输出目录不能包含输入目录。")

    def snapshot(self) -> dict:
        return asdict(self)
