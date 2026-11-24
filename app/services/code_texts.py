from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path

from app.config import settings
from app.services.package_identity import package_root


def _read_file(path):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise OSError(f'号段文件位置不可写：{path}')
    return path.read_bytes() if path.is_file() else None


def _replace_file(path, content):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('xb') as f:
            f.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _merged_content(previous, new_lines):
    encoding = 'utf-8-sig' if previous and previous.startswith(b'\xef\xbb\xbf') else 'utf-8'
    try:
        current = (previous or b'').decode(encoding)
    except UnicodeDecodeError:
        encoding = 'gbk'
        current = previous.decode(encoding)
    old_lines = [line.strip() for line in current.splitlines() if line.strip()]
    lines = list(dict.fromkeys([*old_lines, *new_lines]))
    if previous is not None and lines == old_lines:
        return previous, len(lines)
    newline = '\r\n' if '\r\n' in current else '\n'
    ending = newline if current.endswith('\n') else ''
    return (newline.join(lines) + ending).encode(encoding), len(lines)


@contextmanager
def publish_code_texts(batch_root, codes):
    if not codes:
        yield []
        return
    batch_dir = Path(batch_root).resolve()
    root = package_root(batch_dir).resolve()
    upload_root = Path(settings.upload_root).resolve()
    if batch_dir.is_relative_to(upload_root) and (root == upload_root or not root.is_relative_to(upload_root)):
        root = batch_dir
    dirs = sorted((p for p in root.rglob('*') if '.design_matcher' not in p.relative_to(root).parts
                   and p.is_dir() and p.resolve().is_relative_to(root)),
                  key=lambda p: (len(p.relative_to(root).as_posix()), p.as_posix().casefold()))
    psd_dir = next((p for p in dirs if p.name.casefold() == 'psd'), root)
    effects_dir = next((p for p in dirs if '效果图' in p.name), root)
    outputs = []
    changes = []
    for directory, filename, column in (
        (psd_dir, '美工使用号段.txt', 'target_stem'),
        (effects_dir, '运营使用号段.txt', 'operation_code'),
    ):
        target = directory / filename
        if not target.exists() and not target.is_symlink():
            target = next((p / filename for p in [root, *dirs] if (p / filename).is_file()), target)
        if not target.resolve().is_relative_to(root):
            raise OSError(f'号段文件超出设计包目录：{target}')
        previous = _read_file(target)
        content, count = _merged_content(previous, [getattr(code, column) for code in codes])
        outputs.append({'path': str(target), 'relative_path': target.relative_to(root).as_posix(), 'line_count': count})
        if content != previous:
            changes.append((target, previous, content))

    published = []
    try:
        for target, previous, content in changes:
            if _read_file(target) != previous:
                raise OSError(f'号段文件在改名期间发生变化：{target}')
            _replace_file(target, content)
            published.append((target, previous))
        yield outputs
    except BaseException:
        for target, previous in reversed(published):
            if previous is None:
                target.unlink()
            else:
                _replace_file(target, previous)
        raise
