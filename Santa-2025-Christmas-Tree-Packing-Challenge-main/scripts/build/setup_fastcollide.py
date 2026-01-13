"""Setuptools entrypoint to build the `santa_packing.fastcollide` extension."""

from __future__ import annotations

from pathlib import Path

from setuptools import Extension, setup


def main() -> None:
    """Define and run setuptools `setup()` for the extension."""
    import numpy as np

    root = Path(__file__).resolve().parents[2]
    source = root / "santa_packing" / "fastcollide.cpp"
    if not source.is_file():
        raise FileNotFoundError(f"Missing source file: {source}")
    ext = Extension(
        name="santa_packing.fastcollide",
        sources=[str(source)],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )

    setup(
        name="santa_packing_fastcollide",
        version="0.0.0",
        ext_modules=[ext],
    )


if __name__ == "__main__":
    main()
