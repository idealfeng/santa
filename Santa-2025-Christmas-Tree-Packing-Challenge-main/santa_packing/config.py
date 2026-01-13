"""Shared helpers for loading configuration files.

Config files are expected to live under `configs/` at the repository root and
be committed to version control to improve reproducibility.

Supported formats:
- JSON (always)
- YAML (optional; requires `pyyaml` to be installed)

Supported shapes:
- {"args": ["--flag", "value", "--bool-flag", ...]}  # explicit argv
- {"<section>": {"some_flag": 123, "bool_flag": true, ...}}  # section mapping
- {"some_flag": 123, "nested": {"flag": 1}}  # mapping (nested keys flattened)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def repo_root_from_cwd() -> Path:
    """Return the repository root (directory containing `pyproject.toml`), if found."""
    cwd = Path.cwd().resolve()
    for cand in (cwd, *cwd.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    return cwd


def default_config_path(filename: str) -> Path | None:
    """Return `configs/<filename>` under the repo root if it exists."""
    root = repo_root_from_cwd()
    path = (root / "configs" / filename).resolve()
    return path if path.is_file() else None


def _load_config_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"YAML config requires pyyaml: {exc}") from exc
        return yaml.safe_load(raw)
    return json.loads(raw)


def _flatten_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested dict keys by joining with '_' (e.g. sa.nmax -> sa_nmax)."""

    out: dict[str, Any] = {}

    def _walk(prefix: str, obj: dict[str, Any]) -> None:
        for key, value in obj.items():
            key_str = str(key).strip()
            if not key_str:
                continue
            name = f"{prefix}_{key_str}" if prefix else key_str
            if isinstance(value, dict):
                _walk(name, value)
            else:
                out[name] = value

    _walk("", mapping)
    return out


def config_to_argv(config_path: Path, *, section_keys: Iterable[str] = ()) -> list[str]:
    """Convert a JSON/YAML config file into argv-like tokens.

    The returned list is intended to be prepended to `sys.argv[1:]` so that
    explicit CLI flags override config defaults.
    """
    data = _load_config_file(config_path)

    if isinstance(data, dict) and "args" in data:
        args = data["args"]
        if not isinstance(args, list):
            raise TypeError(f"{config_path}: expected 'args' to be a list, got {type(args).__name__}")
        argv = [str(x) for x in args]
        if any(tok == "--config" or tok.startswith("--config=") for tok in argv):
            raise ValueError(f"{config_path}: 'args' must not include --config (avoid recursion)")
        return argv

    if not isinstance(data, dict):
        raise TypeError(f"{config_path}: expected a mapping at top-level, got {type(data).__name__}")

    section: dict[str, Any] | None = None
    for key in section_keys:
        cand = data.get(key)
        if isinstance(cand, dict):
            section = cand
            break
    mapping = _flatten_mapping(section if section is not None else data)

    argv: list[str] = []
    for key, value in mapping.items():
        if key in {"config", "args"}:
            continue

        flag = str(key).strip()
        if not flag:
            continue
        if not flag.startswith("--"):
            flag = "--" + flag.replace("_", "-")

        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.append(",".join(str(x) for x in value))
            continue
        if isinstance(value, dict):
            raise TypeError(f"{config_path}: nested dict value not supported after flattening: {key!r}")

        argv.append(flag)
        argv.append(str(value))

    return argv
