"""Local, disposable cache of expensive LightGlue pair diagnostics."""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path


class PairResultCache:
    def __init__(self, path: Path, config: dict):
        self.path = path
        self.config = json.dumps(config, sort_keys=True)
        self.fingerprints: dict[str, str] = {}
        self.rows: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.rows = data
        except (OSError, ValueError):
            pass

    def _fingerprint(self, path: str) -> str:
        if path not in self.fingerprints:
            source = Path(path)
            stat = source.stat()
            self.fingerprints[path] = f"{source.absolute()}|{stat.st_size}|{stat.st_mtime_ns}"
        return self.fingerprints[path]

    def key(self, query: str, main: str) -> str:
        value = json.dumps([self.config, self._fingerprint(query), self._fingerprint(main)])
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        row = self.rows.get(key)
        fields = {"raw_match_count", "ransac_inlier_count", "inlier_ratio",
                  "mean_match_confidence", "coverage_score", "local_score"}
        if (isinstance(row, dict) and set(row) == fields
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in row.values())):
            self.hits += 1
            return row
        self.misses += 1
        return None

    def save(self) -> None:
        temporary = self.path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(self.rows), encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError:
            # Cache availability must never decide whether matching succeeds.
            pass
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
