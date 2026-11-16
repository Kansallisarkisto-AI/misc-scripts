#!/usr/bin/env python3
import subprocess, tomllib
from pathlib import Path

proj = tomllib.loads(Path("pyproject.toml").read_text())
extras = proj.get("project", {}).get("optional-dependencies", {}).keys()

# base first, then each extra constrained to the base's pins
subprocess.run(["uv", "pip", "compile", "pyproject.toml", "-o", "requirements/requirements.txt"], check=True)

for extra in extras:
    subprocess.run([
        "uv", "pip", "compile", "pyproject.toml",
        "--extra", extra,
        "-c", "requirements/requirements.txt",   # keeps shared packages at identical versions
        "-o", f"requirements/requirements-{extra}.txt",
    ], check=True)