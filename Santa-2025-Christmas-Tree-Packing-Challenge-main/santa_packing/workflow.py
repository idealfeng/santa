"""High-level workflow API (generate -> improve -> validate/score -> archive).

This module is the recommended public interface for this repo.

Programmatic use:
  from santa_packing.workflow import solve
  res = solve()

CLI use (single entrypoint):
  python -m santa_packing
  santa-solve
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from santa_packing.config import config_to_argv, default_config_path, repo_root_from_cwd
from santa_packing.scoring import OverlapMode, score_submission


@dataclass(frozen=True)
class SolveResult:
    """Artifacts and score produced by `solve()`."""

    run_dir: Path
    generated_submission: Path
    base_submission: Path
    improved_submission: Path | None
    final_submission: Path
    exported_submission: Path | None
    score: float
    score_data: dict


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


def _safe_int(value: object, default: int = 1) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return int(default)


def _filter_generator_args(extra: Sequence[str]) -> list[str]:
    blocked = {"--config", "--out", "--nmax", "--seed", "--overlap-mode"}
    extra = [tok for tok in extra if tok != "--"]
    for tok in extra:
        if (
            tok in blocked
            or tok.startswith("--config=")
            or tok.startswith("--out=")
            or tok.startswith("--nmax=")
            or tok.startswith("--seed=")
            or tok.startswith("--overlap-mode=")
        ):
            raise ValueError(f"Pass {tok} via `solve(...)` kwargs (avoid inconsistencies).")
    return list(extra)


def solve(
    *,
    nmax: int = 200,
    seed: int | None = None,
    overlap_mode: OverlapMode = "kaggle",
    name: str | None = None,
    submissions_dir: Path = Path("submissions"),
    config: Path | None = None,
    no_config: bool = False,
    improve: bool = True,
    smooth_window: int = 60,
    improve_n200: bool = True,
    autofix: bool = True,
    export: bool = True,
    export_path: Path = Path("submission.csv"),
    generator_args: Sequence[str] = (),
) -> SolveResult:
    """Run the end-to-end pipeline and return artifacts + score.

    This is the preferred API entrypoint (no CLI/argparse required).

    Note:
        `overlap_mode` controls how "overlap" is defined during finalization and
        scoring:
        - `strict`: touching is allowed (boundary contact is not considered overlap).
        - `conservative`: touching counts as overlap (more robust, less dense).
        - `kaggle`: same as `strict` (touching allowed); kept as a convenience alias.
    """
    root = repo_root_from_cwd()

    nmax = int(nmax)
    if nmax <= 0 or nmax > 200:
        raise ValueError("nmax must be in [1, 200]")

    if no_config and config is not None:
        raise ValueError("Use either config=... or no_config=True, not both.")

    if not no_config and config is None:
        config = default_config_path("submit_strong.json") or default_config_path("submit.json")
    if config is not None:
        config = Path(config)
        if not config.is_absolute():
            config = (root / config).resolve()
        if not config.is_file():
            raise FileNotFoundError(f"Config not found: {config}")

    submissions_dir = Path(submissions_dir)
    if not submissions_dir.is_absolute():
        submissions_dir = (root / submissions_dir).resolve()
    export_path = Path(export_path)
    if not export_path.is_absolute():
        export_path = (root / export_path).resolve()

    extra = _filter_generator_args(generator_args)

    sha = _git("rev-parse", "HEAD") or "unknown"
    short = sha[:7] if sha != "unknown" else "nogit"
    dirty = bool(_git("status", "--porcelain"))
    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = _safe_slug(name) if name else None
    run_name = f"{ts}_{short}" + (f"_{label}" if label else "")

    run_dir = submissions_dir / run_name
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = submissions_dir / f"{run_name}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    generated_csv = run_dir / "submission.csv"
    improved_csv = run_dir / "submission_improved.csv"
    final_csv = run_dir / "submission_final.csv"

    gen_log = run_dir / "generate.log"
    improve_log = run_dir / "improve.log"
    autofix_log = run_dir / "autofix.log"
    score_log = run_dir / "score.log"

    config_copy: str | None = None
    if config is not None:
        dst = run_dir / f"config{config.suffix}"
        dst.write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
        config_copy = str(dst)

    from santa_packing.cli.generate_submission import main as generate_main  # noqa: E402

    gen_argv: list[str] = []
    if config is not None:
        gen_argv += ["--config", str(config)]
    gen_argv += ["--nmax", str(int(nmax))]
    if seed is not None:
        gen_argv += ["--seed", str(int(seed))]
    gen_argv += ["--overlap-mode", str(overlap_mode)]
    gen_argv += list(extra)
    gen_argv += ["--out", str(generated_csv)]

    gen_rc = 1
    with gen_log.open("w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                gen_rc = int(generate_main(gen_argv))
            except SystemExit as e:
                gen_rc = _safe_int(e.code or 1)
            except Exception:
                traceback.print_exc()
                gen_rc = 1
    if gen_rc != 0:
        raise RuntimeError(f"Generation failed (see {gen_log}).")

    def _score_or_raise(path: Path) -> dict:
        result = score_submission(
            path,
            nmax=int(nmax),
            check_overlap=True,
            overlap_mode=str(overlap_mode),
            require_complete=True,
        )
        return result.to_json()

    def _autofix_and_rescore(path: Path) -> tuple[Path, dict] | None:
        if not bool(autofix):
            return None
        try:
            from santa_packing.cli.autofix_submission import main as autofix_main  # noqa: E402
        except Exception:
            return None

        fixed_path = path.with_name(path.stem + "_fixed" + path.suffix)
        fix_seed = int(seed) if seed is not None else 123
        fix_argv = [
            str(path),
            "--out",
            str(fixed_path),
            "--nmax",
            str(int(nmax)),
            "--seed",
            str(fix_seed),
            "--overlap-mode",
            str(overlap_mode),
        ]

        fix_rc = 1
        with autofix_log.open("a", encoding="utf-8") as f:
            f.write(f"+ autofix_submission {' '.join(fix_argv)}\n")
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                try:
                    fix_rc = int(autofix_main(fix_argv))
                except SystemExit as e:
                    fix_rc = _safe_int(e.code or 1)
                except Exception:
                    traceback.print_exc()
                    fix_rc = 1
        if fix_rc != 0 or not fixed_path.is_file():
            return None
        try:
            return fixed_path, _score_or_raise(fixed_path)
        except Exception:
            return None

    base_csv = generated_csv
    score_base_path = run_dir / "score_base.json"
    try:
        base_score = _score_or_raise(base_csv)
    except Exception:
        fixed = _autofix_and_rescore(base_csv)
        if fixed is None:
            score_log.write_text(traceback.format_exc() + "\n", encoding="utf-8")
            raise RuntimeError(f"Scoring failed (see {score_log}).")
        base_csv, base_score = fixed
    score_base_path.write_text(json.dumps(base_score, indent=2) + "\n", encoding="utf-8")

    best_csv = base_csv
    best_score = base_score
    improved_out: Path | None = None

    if bool(improve):
        from santa_packing.cli.improve_submission import main as improve_main  # noqa: E402

        improve_argv = [
            str(best_csv),
            "--out",
            str(improved_csv),
            "--nmax",
            str(int(nmax)),
            "--smooth-window",
            str(int(smooth_window)),
            "--overlap-mode",
            str(overlap_mode),
        ]
        if bool(improve_n200) and int(nmax) >= 200:
            improve_argv.append("--improve-n200")
            seed_val = int(seed) if seed is not None else 123
            improve_argv += ["--n200-insert-seed", str(seed_val), "--n200-sa-seed", str(seed_val)]

        improve_rc = 1
        with improve_log.open("w", encoding="utf-8") as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                try:
                    improve_rc = int(improve_main(improve_argv))
                except SystemExit as e:
                    improve_rc = _safe_int(e.code or 1)
                except Exception:
                    traceback.print_exc()
                    improve_rc = 1

        if improve_rc != 0 and bool(improve_n200):
            retry_argv = [tok for tok in improve_argv if tok != "--improve-n200"]
            if retry_argv != improve_argv:
                with improve_log.open("a", encoding="utf-8") as f:
                    f.write("\n[retry] without --improve-n200\n")
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        try:
                            improve_rc = int(improve_main(retry_argv))
                        except SystemExit as e:
                            improve_rc = _safe_int(e.code or 1)
                        except Exception:
                            traceback.print_exc()
                            improve_rc = 1

        improve_data: dict | None = None
        if improve_rc == 0 and improved_csv.is_file():
            improved_out = improved_csv
            try:
                improve_data = _score_or_raise(improved_csv)
            except Exception:
                fixed = _autofix_and_rescore(improved_csv)
                if fixed is not None:
                    improved_out = fixed[0]
                    improve_data = fixed[1]

        if improve_data is not None and float(improve_data["score"]) < float(best_score["score"]):
            best_csv = improved_out or improved_csv
            best_score = improve_data

    shutil.copyfile(best_csv, final_csv)

    score_data = best_score
    (run_dir / "score.json").write_text(json.dumps(score_data, indent=2) + "\n", encoding="utf-8")
    score_log.write_text("", encoding="utf-8")

    exported: Path | None = None
    if bool(export):
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final_csv, export_path)
        exported = export_path

    meta = {
        "timestamp_utc": ts,
        "git": {"sha": sha, "dirty": dirty},
        "paths": {
            "run_dir": str(run_dir),
            "generated_submission": str(generated_csv),
            "base_submission": str(base_csv),
            "improved_submission": str(improved_out) if improved_out is not None else None,
            "final_submission": str(final_csv),
        },
        "config": {"path": str(config) if config is not None else None, "copy": config_copy},
        "generator": {"argv": gen_argv, "log": str(gen_log)},
        "improve": {
            "enabled": bool(improve),
            "smooth_window": int(smooth_window),
            "improve_n200": bool(improve_n200),
            "log": str(improve_log),
        },
        "autofix": {"enabled": bool(autofix), "log": str(autofix_log)},
        "scorer": {
            "nmax": int(nmax),
            "check_overlap": True,
            "overlap_mode": str(overlap_mode),
            "require_complete": True,
            "log": str(score_log),
        },
        "export": {"enabled": bool(export), "path": str(exported) if exported is not None else None},
        "score": score_data,
        "python": sys.version,
    }
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return SolveResult(
        run_dir=run_dir,
        generated_submission=generated_csv,
        base_submission=base_csv,
        improved_submission=improved_out,
        final_submission=final_csv,
        exported_submission=exported,
        score=float(score_data.get("score", float("nan"))),
        score_data=score_data,
    )


def archive_submission(
    submission: Path,
    *,
    nmax: int = 200,
    overlap_mode: OverlapMode = "kaggle",
    name: str | None = None,
    submissions_dir: Path = Path("submissions"),
    autofix: bool = True,
    autofix_seed: int = 123,
    export: bool = True,
    export_path: Path = Path("submission.csv"),
    extra_meta: dict | None = None,
) -> SolveResult:
    """Archive and score an existing `submission.csv`.

    This is useful when a notebook/tool produced a CSV and you want the same
    `submissions/<timestamp>.../` bundle with `score.json` + `meta.json`.
    """
    root = repo_root_from_cwd()

    submission = Path(submission)
    if not submission.is_absolute():
        submission = (root / submission).resolve()
    if not submission.is_file():
        raise FileNotFoundError(f"submission not found: {submission}")

    nmax = int(nmax)
    if nmax <= 0 or nmax > 200:
        raise ValueError("nmax must be in [1, 200]")

    submissions_dir = Path(submissions_dir)
    if not submissions_dir.is_absolute():
        submissions_dir = (root / submissions_dir).resolve()
    export_path = Path(export_path)
    if not export_path.is_absolute():
        export_path = (root / export_path).resolve()

    sha = _git("rev-parse", "HEAD") or "unknown"
    short = sha[:7] if sha != "unknown" else "nogit"
    dirty = bool(_git("status", "--porcelain"))
    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = _safe_slug(name) if name else None
    run_name = f"{ts}_{short}" + (f"_{label}" if label else "")

    run_dir = submissions_dir / run_name
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = submissions_dir / f"{run_name}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    input_csv = run_dir / "submission_input.csv"
    base_csv = run_dir / "submission_base.csv"
    final_csv = run_dir / "submission_final.csv"
    score_log = run_dir / "score.log"
    autofix_log = run_dir / "autofix.log"

    shutil.copyfile(submission, input_csv)
    shutil.copyfile(submission, base_csv)

    def _score_or_raise(path: Path) -> dict:
        result = score_submission(
            path,
            nmax=int(nmax),
            check_overlap=True,
            overlap_mode=str(overlap_mode),
            require_complete=True,
        )
        return result.to_json()

    score_data: dict
    try:
        score_data = _score_or_raise(base_csv)
        score_log.write_text("", encoding="utf-8")
    except Exception:
        if not bool(autofix):
            score_log.write_text(traceback.format_exc() + "\n", encoding="utf-8")
            raise

        try:
            from santa_packing.cli.autofix_submission import main as autofix_main  # noqa: E402
        except Exception:
            score_log.write_text(traceback.format_exc() + "\n", encoding="utf-8")
            raise

        fixed_csv = run_dir / "submission_fixed.csv"
        fix_argv = [
            str(base_csv),
            "--out",
            str(fixed_csv),
            "--nmax",
            str(int(nmax)),
            "--seed",
            str(int(autofix_seed)),
            "--overlap-mode",
            str(overlap_mode),
        ]

        fix_rc = 1
        with autofix_log.open("w", encoding="utf-8") as f:
            f.write(f"+ autofix_submission {' '.join(fix_argv)}\n")
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                try:
                    fix_rc = int(autofix_main(fix_argv))
                except SystemExit as e:
                    fix_rc = _safe_int(e.code or 1)
                except Exception:
                    traceback.print_exc()
                    fix_rc = 1
        if fix_rc != 0 or not fixed_csv.is_file():
            score_log.write_text(traceback.format_exc() + "\n", encoding="utf-8")
            raise RuntimeError(f"autofix failed (see {autofix_log})")

        shutil.copyfile(fixed_csv, base_csv)
        score_data = _score_or_raise(base_csv)
        score_log.write_text("", encoding="utf-8")

    shutil.copyfile(base_csv, final_csv)
    (run_dir / "score.json").write_text(json.dumps(score_data, indent=2) + "\n", encoding="utf-8")

    exported: Path | None = None
    if bool(export):
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final_csv, export_path)
        exported = export_path

    meta = {
        "timestamp_utc": ts,
        "git": {"sha": sha, "dirty": dirty},
        "paths": {
            "run_dir": str(run_dir),
            "input_submission": str(input_csv),
            "base_submission": str(base_csv),
            "final_submission": str(final_csv),
        },
        "scorer": {
            "nmax": int(nmax),
            "check_overlap": True,
            "overlap_mode": str(overlap_mode),
            "require_complete": True,
            "log": str(score_log),
        },
        "autofix": {"enabled": bool(autofix), "seed": int(autofix_seed), "log": str(autofix_log)},
        "export": {"enabled": bool(export), "path": str(exported) if exported is not None else None},
        "score": score_data,
        "python": sys.version,
    }
    if extra_meta:
        meta["extra"] = extra_meta

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return SolveResult(
        run_dir=run_dir,
        generated_submission=input_csv,
        base_submission=base_csv,
        improved_submission=None,
        final_submission=final_csv,
        exported_submission=exported,
        score=float(score_data.get("score", float("nan"))),
        score_data=score_data,
    )


def cli_main(argv: list[str] | None = None) -> int:
    """Thin CLI wrapper around `solve()` (kept for convenience)."""
    argv = list(sys.argv[1:] if argv is None else argv)

    cfg_default = default_config_path("submit_strong.json") or default_config_path("submit.json")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre.add_argument("--no-config", action="store_true")
    pre_args, _ = pre.parse_known_args(argv)
    if pre_args.no_config and pre_args.config is not None:
        raise SystemExit("Use either --config or --no-config, not both.")

    config_path = None if pre_args.no_config else (pre_args.config or cfg_default)
    config_args = config_to_argv(config_path, section_keys=("make_submit", "submit")) if config_path is not None else []

    ap = argparse.ArgumentParser(description="Generate a strong, Kaggle-safe submission and archive artifacts.")
    ap.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="JSON/YAML config passed to the generator (defaults to configs/submit_strong.json when present).",
    )
    ap.add_argument("--no-config", action="store_true", help="Disable loading the default config (if any).")
    ap.add_argument("--nmax", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--overlap-mode",
        type=str,
        default="kaggle",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap predicate used for validation/scoring (strict/kaggle allow touching; conservative counts touching).",
    )
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--submissions-dir", type=Path, default=Path("submissions"))
    ap.add_argument("--improve", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--smooth-window", type=int, default=60)
    ap.add_argument("--improve-n200", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--autofix", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--export", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--export-path", type=Path, default=Path("submission.csv"))
    args, extra = ap.parse_known_args(config_args + argv)

    if args.no_config and args.config is not None:
        raise SystemExit("Use either --config or --no-config, not both.")

    extra = _filter_generator_args(extra)
    try:
        res = solve(
            nmax=int(args.nmax),
            seed=args.seed,
            overlap_mode=str(args.overlap_mode),  # type: ignore[arg-type]
            name=args.name,
            submissions_dir=args.submissions_dir,
            config=args.config,
            no_config=bool(args.no_config),
            improve=bool(args.improve),
            smooth_window=int(args.smooth_window),
            improve_n200=bool(args.improve_n200),
            autofix=bool(args.autofix),
            export=bool(args.export),
            export_path=args.export_path,
            generator_args=extra,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Run: {res.run_dir}")
    print(f"Score: {res.score_data.get('score')}")
    print(f"Submission: {res.final_submission}")
    if res.exported_submission is not None:
        print(f"Exported: {res.exported_submission}")
    return 0
