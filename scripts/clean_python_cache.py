"""Remove repository Python bytecode, including local environments; skip Git internals."""

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list caches without deleting")
    args = parser.parse_args()
    removed = 0
    for directory, dirs, files in os.walk(ROOT, topdown=True, followlinks=False):
        dirs[:] = [name for name in dirs if name != ".git"]
        for name in list(dirs):
            path = Path(directory) / name
            if name == "__pycache__" and not path.is_symlink():
                if not args.dry_run:
                    shutil.rmtree(path)
                print(path.relative_to(ROOT))
                dirs.remove(name)
                removed += 1
        for name in files:
            path = Path(directory) / name
            if path.suffix in (".pyc", ".pyo") and not path.is_symlink():
                if not args.dry_run:
                    path.unlink()
                print(path.relative_to(ROOT))
                removed += 1
    print(f"{'Found' if args.dry_run else 'Removed'} {removed} Python cache entries")


if __name__ == "__main__":
    main()
