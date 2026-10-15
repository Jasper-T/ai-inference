"""Call a running HTTP service; optionally attach SFace five-point landmarks."""

import argparse
import base64
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--source", choices=("file", "base64", "url", "bytes"), default="file")
    parser.add_argument("--image", type=Path, default=ROOT / "tests/bus.jpg")
    parser.add_argument("--image-url", help="Public image URL; required for --source url")
    parser.add_argument(
        "--keypoints", type=Path, help="JSON file containing five original xy points"
    )
    args = parser.parse_args()
    if args.source == "url" and not args.image_url:
        parser.error("--source url requires --image-url")

    metadata = {}
    if args.keypoints is not None:
        metadata["keypoints"] = json.loads(args.keypoints.read_text(encoding="utf-8"))

    # Only the server loads model weights. This client sends encoded images and metadata.
    request = {}
    if args.source == "url":
        request["json"] = {"images": [{"url": args.image_url, **metadata}]}
    else:
        data = args.image.read_bytes()
        if args.source == "file":
            request["files"] = {"files": (args.image.name, data, "application/octet-stream")}
            if metadata:
                request["data"] = {"metadata": json.dumps([metadata])}
        elif args.source == "base64":
            request["json"] = {
                "images": [{"data": base64.b64encode(data).decode("ascii"), **metadata}]
            }
        else:
            request["content"] = data
            request["headers"] = {"Content-Type": "application/octet-stream"}
            if metadata:
                request["params"] = {"metadata": json.dumps(metadata)}

    try:
        with httpx.Client(
            base_url=args.base_url.rstrip("/"), timeout=60, trust_env=False
        ) as client:
            response = client.post(f"/v1/predict/{args.source}", **request)
    except httpx.RequestError as exc:
        raise SystemExit(f"Service request failed: {exc}") from None
    if response.is_error:
        raise SystemExit(f"HTTP {response.status_code}: {response.text}")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
