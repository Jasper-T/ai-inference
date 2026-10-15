#!/usr/bin/env python3
"""Run repeatable SDK checks with the invoking Python environment."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "suite",
        nargs="?",
        default="all",
        choices=("all", "quick", "smoke", "ort"),
        help="all: pytest + smoke; quick: exclude real ORT; "
        "smoke: synthetic pipeline; ort: CPU ORT synthetic graphs",
    )
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="extra pytest arguments, e.g. all -- -x"
    )
    args = parser.parse_args()
    extra = args.pytest_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if args.suite == "smoke" and extra:
        parser.error("smoke does not accept pytest arguments")

    env = os.environ.copy()
    paths = [str(ROOT / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    commands = []
    if args.suite != "smoke":
        command = [sys.executable, "-m", "pytest", "-q"]
        if args.suite == "quick":
            command += ["-k", "not real_ort"]
        elif args.suite == "ort":
            command += ["-k", "real_ort"]
        commands.append(command + extra)
    if args.suite in ("all", "smoke"):
        commands.append([sys.executable, str(ROOT / "scripts" / "smoke.py")])

    print(f"Python: {sys.executable}\nSuite: {args.suite}", flush=True)
    try:
        for command in commands:
            print("Running: " + " ".join(command[1:]), flush=True)
            result = subprocess.run(command, cwd=ROOT, env=env, check=False)
            if result.returncode:
                return result.returncode if result.returncode > 0 else 1
    except KeyboardInterrupt:
        return 130
    print(f"{args.suite}: passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
