"""Bounded image decoding and public-network-only URL fetching."""

import base64
import binascii
import http.client
import ipaddress
import socket
import ssl
import time
import warnings
from io import BufferedReader, BytesIO, RawIOBase
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, UnidentifiedImageError

from .settings import Settings


class InputError(Exception):
    def __init__(self, message: str, status: int = 422, index: int | None = None):
        super().__init__(message)
        self.status = status
        self.index = index


def decode_base64(value: str, settings: Settings) -> bytes:
    if value.startswith("data:"):
        header, separator, value = value.partition(",")
        if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
            raise InputError("Expected an image base64 data URL")
    if len(value) > 4 * ((settings.max_image_bytes + 2) // 3):
        raise InputError("Image exceeds byte limit", 413)
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise InputError("Invalid base64 image") from exc
    if len(data) > settings.max_image_bytes:
        raise InputError("Image exceeds byte limit", 413)
    return data


def decode_image(data: bytes, settings: Settings, remaining_pixels: int) -> np.ndarray:
    if len(data) > settings.max_image_bytes:
        raise InputError("Image exceeds byte limit", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                    raise InputError("Supported images: JPEG, PNG, WEBP, BMP")
                if getattr(image, "n_frames", 1) != 1:
                    raise InputError("Animated images are unsupported")
                width, height = image.size
                if width * height > min(settings.max_image_pixels, remaining_pixels):
                    raise InputError("Image or batch exceeds pixel limit", 413)
                pixels = np.array(image.convert("RGB"))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InputError("Image exceeds pixel limit", 413) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InputError("Invalid image data") from exc
    if settings.input_color == "BGR":
        pixels = pixels[:, :, ::-1]
    return np.ascontiguousarray(pixels)


def public_address(host: str, port: int) -> str:
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise InputError("URL host has no addresses")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global or (
            ip.version == 6
            and (ip.ipv4_mapped is not None or ip.sixtofour is not None or ip.teredo is not None)
        ):
            raise InputError("URL must resolve exclusively to public addresses")
    return addresses[0][4][0]


def remaining_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Download deadline exceeded")
    return remaining


class DeadlineReader(RawIOBase):
    """Keep a socket file reference and enforce the deadline on every network read."""

    def __init__(self, sock, deadline):
        super().__init__()
        self.sock, self.deadline = sock, deadline
        self.raw = sock.makefile("rb", buffering=0)

    def readable(self):
        return True

    def readinto(self, buffer):
        self.sock.settimeout(remaining_timeout(self.deadline))
        return self.raw.readinto(buffer)

    def close(self):
        try:
            self.raw.close()
        finally:
            super().close()


class DeadlineSocket:
    """HTTP's buffered parser must not hide repeated reads from the deadline."""

    def __init__(self, sock, deadline):
        self.sock, self.deadline = sock, deadline

    def sendall(self, data):
        self.sock.settimeout(remaining_timeout(self.deadline))
        self.sock.sendall(data)

    def makefile(self, mode):
        if mode != "rb":
            raise ValueError("Only HTTP response reading is supported")
        return BufferedReader(DeadlineReader(self.sock, self.deadline))

    def close(self):
        self.sock.close()


def download_image(value: str, settings: Settings) -> bytes:
    connection = response = None
    try:
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.fragment
            or any(ord(char) <= 32 or ord(char) == 127 for char in value)
        ):
            raise InputError("Expected a public HTTP(S) URL without credentials or fragment")
        port = url.port or (443 if url.scheme == "https" else 80)
        if port != (443 if url.scheme == "https" else 80):
            raise InputError("Only standard HTTP(S) ports are allowed")
        host = url.hostname.encode("idna").decode("ascii")
        address = public_address(host, port)
        deadline = time.monotonic() + settings.url_timeout_seconds
        if url.scheme == "https":
            connection = http.client.HTTPSConnection(
                host,
                port,
                timeout=settings.url_timeout_seconds,
                context=ssl.create_default_context(),
            )
        else:
            connection = http.client.HTTPConnection(
                host, port, timeout=settings.url_timeout_seconds
            )

        # Pin the validated address; HTTPS still verifies TLS against the original hostname.
        def connect_pinned(target, timeout, source_address=None):
            sock = socket.create_connection(
                (address, port), remaining_timeout(deadline), source_address
            )
            try:
                # HTTPS performs its handshake next, using the remaining total budget.
                sock.settimeout(remaining_timeout(deadline))
                return sock
            except BaseException:
                sock.close()
                raise

        connection._create_connection = connect_pinned
        connection.connect()
        connection.sock = DeadlineSocket(connection.sock, deadline)
        path = url.path or "/"
        if url.query:
            path += "?" + url.query
        connection.request(
            "GET", path, headers={"Accept": "image/*", "Accept-Encoding": "identity"}
        )
        response = connection.getresponse()
        if response.status != 200:
            raise InputError("Image URL must return HTTP 200; redirects are unsupported", 422)
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise InputError("Encoded HTTP responses are unsupported")
        length = response.getheader("Content-Length")
        if length and int(length) > settings.max_image_bytes:
            raise InputError("Downloaded image exceeds byte limit", 413)
        chunks, size = [], 0
        while True:
            remaining_timeout(deadline)
            chunk = response.read1(min(65536, settings.max_image_bytes + 1 - size))
            remaining_timeout(deadline)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > settings.max_image_bytes:
                raise InputError("Downloaded image exceeds byte limit", 413)
        return b"".join(chunks)
    except InputError:
        raise
    except (TimeoutError, socket.timeout) as exc:
        raise InputError("Image download timed out", 504) from exc
    except (OSError, ValueError, http.client.HTTPException, UnicodeError) as exc:
        raise InputError("Unable to download image URL", 422) from exc
    finally:
        if response is not None:
            response.close()
        if connection is not None:
            connection.close()
