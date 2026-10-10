"""Assemble the Hugging Face Space (free CPU demo) and optionally upload it.

    python scripts/build_hf_space.py                     # assemble into build/hf-space (inspect, or docker build it)
    python scripts/build_hf_space.py --push USER/SPACE   # also create/update the Space (needs `hf auth login`)

The Space holds only what the demo needs: the backend package, the sender and adapter checkpoints, the bundled
records, the recorded Gemma answers and the frontend. No tests, logs, results or local environments.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "build" / "hf-space"
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info", ".venv*", "tests")


def assemble() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.iterdir():  # empty it rather than delete it: Windows refuses to delete a folder in use
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for f in ("Dockerfile", "start.sh", "README.md"):
        shutil.copy2(REPO / "deploy/hf-space" / f, OUT / f)
    b = OUT / "backend"
    b.mkdir()
    shutil.copy2(REPO / "backend/pyproject.toml", b)
    shutil.copytree(REPO / "backend/src", b / "src", ignore=SKIP)
    shutil.copytree(REPO / "backend/artifacts", b / "artifacts", ignore=SKIP)
    shutil.copytree(REPO / "backend/data/records", b / "data/records", ignore=SKIP)
    (b / "data/replay").mkdir(parents=True)
    shutil.copy2(REPO / "backend/data/replay/answers.jsonl", b / "data/replay/answers.jsonl")
    shutil.copytree(REPO / "frontend", OUT / "frontend", ignore=SKIP)
    shutil.copy2(REPO / "LICENSE", OUT / "LICENSE")
    for p in OUT.rglob("*.sh"):  # the container is Linux: no Windows line endings in scripts
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"assembled {OUT} ({size / 2**20:.1f} MB, {sum(1 for p in OUT.rglob('*') if p.is_file())} files)")
    return OUT


def push(space_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=space_id, repo_type="space",
                      commit_message="Deploy from github.com/a1if/ecg-multi-agent-triage", delete_patterns=["*"])
    print(f"pushed: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", metavar="USER/SPACE")
    a = ap.parse_args()
    assemble()
    if a.push:
        push(a.push)
