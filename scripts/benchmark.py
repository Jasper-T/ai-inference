"""Measure an existing Model through public APIs; no SDK timing state or subclass."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2
import numpy as np
from loguru import logger

from ai_inference import Config, Model

ROOT = Path(__file__).resolve().parents[1]


def benchmark(
    model, images, *, runs: int = 20, warmup: int = 3, synchronize: Callable[[], None] | None = None
) -> dict:
    """Exclude warmup; measure public stage calls with optional device completion."""
    if isinstance(runs, bool) or not isinstance(runs, int) or runs <= 0:
        raise ValueError("runs must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    count = len(images) if isinstance(images, list) else 1
    if count == 0:
        raise ValueError("benchmark requires at least one sample")
    sync = synchronize if synchronize is not None else lambda: None
    for _ in range(warmup):
        model.predict(images)
        sync()
    measurements = {name: [] for name in ("preprocess", "predict", "postprocess", "total")}
    for _ in range(runs):
        sync()
        start = perf_counter()
        prepared = model.preprocess(images)
        sync()
        after_pre = perf_counter()
        raw = model.infer(prepared)
        sync()
        after_predict = perf_counter()
        model.postprocess(raw)
        sync()
        end = perf_counter()
        for name, value in zip(
            measurements,
            (after_pre - start, after_predict - after_pre, end - after_predict, end - start),
        ):
            measurements[name].append(value)

    def summary(values):
        values = np.asarray(values) * 1000
        return {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    elapsed = sum(measurements["total"])
    return {
        "runs": runs,
        "warmup_runs": warmup,
        "samples_per_run": count,
        "latency_unit": "ms",
        "latency": {name: summary(values) for name, values in measurements.items()},
        "throughput_samples_per_second": runs * count / elapsed if elapsed > 0 else None,
    }


def format_report(report: dict) -> str:
    """Readable terminal summary; JSON files retain the full measurement precision."""
    lines = [
        f"Model: {report.get('model', '-')} | Backend: {report.get('backend', '-')}",
        f"Runs: {report['runs']} | Warmup: {report['warmup_runs']} | Samples/run: {report['samples_per_run']}",
    ]
    if report.get("providers"):
        lines.append(f"Providers: {', '.join(report['providers'])}")
    columns = ("Mean (ms)", "P50 (ms)", "P95 (ms)", "Min (ms)", "Max (ms)")
    lines += ["", f"{'Stage':<14}" + "".join(f"{name:>12}" for name in columns)]
    for name, values in report["latency"].items():
        lines.append(
            f"{name:<14}"
            + "".join(f"{values[key]:>12.3f}" for key in ("mean", "p50", "p95", "min", "max"))
        )
    throughput = report["throughput_samples_per_second"]
    lines += [
        "",
        f"Throughput: {throughput:.2f} samples/s" if throughput is not None else "Throughput: n/a",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/yolo11.yaml")
    parser.add_argument("--image", type=Path, default=ROOT / "tests/bus.jpg")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--save",
        type=Path,
        metavar="PATH",
        help="optionally save a JSON report; default: terminal only",
    )
    args = parser.parse_args()
    if args.runs <= 0 or args.warmup < 0 or args.samples <= 0:
        parser.error("runs/samples must be positive; warmup must be nonnegative")
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"cannot read image: {args.image}")
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    cfg = Config.from_yaml(args.config)
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"warmup": False})})
    model = Model(cfg)
    report = benchmark(model, [image] * args.samples, runs=args.runs, warmup=args.warmup)
    report.update(
        model=cfg.model.name,
        backend=cfg.runtime.backend,
        batch=cfg.runtime.batch.model_dump(),
        providers=list(getattr(model.engine, "providers", ())),
    )
    print(format_report(report))
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nSaved report: {args.save}")


if __name__ == "__main__":
    main()
