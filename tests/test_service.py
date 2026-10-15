"""HTTP boundary regression tests; no real model weights or network required."""

import base64
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, RawIOBase
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient
from PIL import Image

from ai_inference import FeaturePrediction, Prediction
from ai_inference.service.app import create_app
from ai_inference.service.inputs import InputError, download_image, public_address
from ai_inference.service.settings import Settings


class FakeModel:
    instances = 0

    def __init__(self, path):
        type(self).instances += 1
        self.calls = []
        self.config = SimpleNamespace(
            model=SimpleNamespace(name="test"),
            runtime=SimpleNamespace(backend="fake", batch=SimpleNamespace(model_dump=lambda: {})),
            preprocess={"source_color": "BGR", "steps": []},
            inputs=(),
            outputs=(),
        )

    def predict(self, images):
        self.calls.append(images)
        return [
            Prediction(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64), image.shape[:2])
            for image in images
        ]


@pytest.fixture
def image():
    output = BytesIO()
    Image.new("RGB", (3, 2), (255, 0, 0)).save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def application():
    return create_app(Settings(model_config_path="unused.yaml"), model_factory=FakeModel)


def test_all_sources_share_color_results_and_one_model(application, image, monkeypatch):
    import ai_inference.service.app as module

    monkeypatch.setattr(module, "download_image", lambda url, settings: image)
    before = FakeModel.instances
    with TestClient(application) as client:
        responses = [
            client.post("/v1/predict/bytes", content=image, headers={"Content-Type": "image/png"}),
            client.post("/v1/predict/file", files=[("files", ("a.png", image, "image/png"))]),
            client.post(
                "/v1/predict/base64", json={"images": [{"data": base64.b64encode(image).decode()}]}
            ),
            client.post("/v1/predict/url", json={"images": [{"url": "https://example.com/a.png"}]}),
        ]
        assert all(response.status_code == 200 for response in responses)
        assert all(response.json() == responses[0].json() for response in responses)
        assert responses[0].json()["results"][0]["original_shape"] == [2, 3]
        assert FakeModel.instances == before + 1
        assert all(
            np.array_equal(call[0][0, 0], [0, 0, 255])
            for call in application.state.runtime.model.calls
        )
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        assert client.get("/v1/model").json()["name"] == "test"
        assert "unused.yaml" not in client.get("/v1/model").text
    assert application.state.runtime is None


def test_batch_is_ordered_and_atomic(application, image):
    encoded = base64.b64encode(image).decode()
    with TestClient(application) as client:
        response = client.post(
            "/v1/predict/base64", json={"images": [{"data": encoded}, {"data": "bad"}]}
        )
        assert response.status_code == 422
        assert response.json()["error"]["index"] == 1
        assert not application.state.runtime.model.calls
        response = client.post(
            "/v1/predict/file", files=[("files", ("a.png", image)), ("files", ("b.png", image))]
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) == 2
        assert (
            client.post("/v1/predict/base64", json={"images": [{"data": encoded}] * 9}).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/predict/file", files={"files": ("a.png", image)}, data={"metadata": "[]"}
            ).status_code
            == 422
        )


def test_limits_and_validation(image):
    app = create_app(
        Settings(
            model_config_path="unused",
            max_request_bytes=200,
            max_image_bytes=100,
            max_image_pixels=5,
        ),
        model_factory=FakeModel,
    )
    with TestClient(app) as client:
        assert client.post("/v1/predict/bytes", content=b"x" * 201).status_code == 413
        assert (
            client.post(
                "/v1/predict/bytes", content=image, headers={"Content-Type": "image/png"}
            ).status_code
            == 413
        )
        assert (
            client.post(
                "/v1/predict/bytes", content=b"x" * 101, headers={"Content-Type": "image/png"}
            ).status_code
            == 413
        )
        assert client.post("/v1/predict/bytes", content=image).status_code == 415
        response = client.post(
            "/v1/predict/base64", json={"images": [{"data": "secret", "oops": True}]}
        )
        assert response.status_code == 422
        assert "secret" not in response.text


def test_chunked_request_limit(application):
    with TestClient(application) as client:
        application.state.settings = application.state.settings.model_copy(
            update={"max_request_bytes": 4}
        )
        response = client.post(
            "/v1/predict/bytes",
            content=iter([b"123", b"456"]),
            headers={"Content-Type": "image/png"},
        )
        assert response.status_code == 413


def test_busy_rejects_without_blocking_health(application, image):
    entered, release = threading.Event(), threading.Event()
    with TestClient(application) as client:
        original = application.state.runtime.model.predict

        def slow(images):
            entered.set()
            assert release.wait(5)
            return original(images)

        application.state.runtime.model.predict = slow
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                client.post,
                "/v1/predict/bytes",
                content=image,
                headers={"Content-Type": "image/png"},
            )
            try:
                assert entered.wait(5)
                assert client.post("/v1/predict/bytes", content=image).status_code == 503
                assert client.get("/health/ready").status_code == 200
            finally:
                release.set()
            assert future.result().status_code == 200


def test_model_failure_is_hidden_and_admission_is_released(application, image):
    with TestClient(application) as client:

        def fail(images):
            raise RuntimeError("secret internal model path")

        application.state.runtime.model.predict = fail
        for _ in range(2):
            response = client.post(
                "/v1/predict/bytes", content=image, headers={"Content-Type": "image/png"}
            )
            assert response.status_code == 500
            assert "secret" not in response.text


def test_startup_failure_propagates():
    def fail(path):
        raise RuntimeError("load failed")

    with pytest.raises(RuntimeError, match="load failed"):
        with TestClient(create_app(Settings(model_config_path="unused"), model_factory=fail)):
            pass


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "::ffff:8.8.8.8"]
)
def test_url_rejects_nonpublic_or_mapped_addresses(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", (address, 443))])
    with pytest.raises(InputError, match="public"):
        public_address("example.com", 443)


def test_url_rejects_mixed_dns_answers(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [(2, 1, 6, "", (ip, 443)) for ip in ["8.8.8.8", "127.0.0.1"]],
    )
    with pytest.raises(InputError):
        public_address("example.com", 443)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://user:pass@example.com/a",
        "http://example.com:8080/a",
        "http://example.com/\r\nX:1",
    ],
)
def test_bad_url(url):
    with pytest.raises(InputError):
        download_image(url, Settings(model_config_path="unused"))


@pytest.fixture
def download_wire(monkeypatch):
    import ai_inference.service.inputs as module

    clock = [0.0]
    destinations, sockets, hostnames = [], [], []
    monkeypatch.setattr(module, "public_address", lambda *args: "8.8.8.8")
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def install(data, *, slow_after=None, connect_cost=0.0, tls_cost=0.0):
        class WireFile(RawIOBase):
            def readable(self):
                return True

            def readinto(self, buffer):
                if wire.offset >= len(data):
                    return 0
                if slow_after is not None and wire.offset >= slow_after:
                    wire.delay(0.04)
                    size = 1
                else:
                    size = len(buffer)
                    if slow_after is not None:
                        size = min(size, slow_after - wire.offset)
                chunk = data[wire.offset : wire.offset + size]
                buffer[: len(chunk)] = chunk
                wire.offset += len(chunk)
                return len(chunk)

        class Wire:
            offset = 0
            closed = False
            timeout = 0

            def delay(self, cost):
                if cost >= self.timeout:
                    clock[0] += self.timeout
                    raise TimeoutError()
                clock[0] += cost

            def settimeout(self, value):
                self.timeout = value

            def setsockopt(self, *args):
                pass

            def sendall(self, data):
                pass

            def makefile(self, *args, **kwargs):
                self.file = WireFile()
                return self.file

            def close(self):
                self.closed = True

        wire = Wire()
        sockets.append(wire)

        def connect(target, timeout, source):
            destinations.append(target)
            wire.settimeout(timeout)
            try:
                wire.delay(connect_cost)
            except BaseException:
                wire.close()
                raise
            return wire

        def wrap(sock, *, server_hostname):
            hostnames.append(server_hostname)
            wire.delay(tls_cost)
            return sock

        monkeypatch.setattr(socket, "create_connection", connect)
        context = module.ssl.create_default_context()
        monkeypatch.setattr(context, "wrap_socket", wrap)
        monkeypatch.setattr(module.ssl, "create_default_context", lambda: context)
        return wire

    return SimpleNamespace(
        install=install,
        clock=clock,
        destinations=destinations,
        sockets=sockets,
        hostnames=hostnames,
    )


def test_url_pins_address_and_rejects_redirects(download_wire, image):
    settings = Settings(model_config_path="unused")
    wire = download_wire.install(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n" + image)
    assert download_image("https://example.com/a.png", settings) == image
    assert download_wire.destinations == [("8.8.8.8", 443)]
    assert download_wire.hostnames == ["example.com"]
    assert wire.closed and wire.file.closed
    wire = download_wire.install(b"HTTP/1.1 302 Found\r\nContent-Length: 0\r\n\r\n")
    with pytest.raises(InputError, match="redirects"):
        download_image("https://example.com/a.png", settings)
    assert wire.closed and wire.file.closed
    wire = download_wire.install(b"HTTP/1.1 200 OK\r\n\r\n" + image)
    with pytest.raises(InputError) as exc:
        download_image(
            "https://example.com/a.png", settings.model_copy(update={"max_image_bytes": 4})
        )
    assert exc.value.status == 413
    assert wire.closed and wire.file.closed


@pytest.mark.parametrize("phase", ["headers", "chunked", "closed_body", "connect", "tls"])
def test_url_deadline_applies_to_each_network_read_and_handshake(download_wire, phase):
    headers = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n"
    if phase == "headers":
        wire = download_wire.install(headers + b"image", slow_after=0)
    elif phase == "chunked":
        headers = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
        wire = download_wire.install(
            headers + b"0" * 100 + b"1\r\nx\r\n0\r\n\r\n", slow_after=len(headers)
        )
    elif phase == "closed_body":
        wire = download_wire.install(headers + b"image", slow_after=len(headers))
    else:
        wire = download_wire.install(
            headers + b"image", connect_cost=0.11 if phase == "connect" else 0.06, tls_cost=0.06
        )
    with pytest.raises(InputError) as exc:
        download_image(
            "https://example.com/a.png",
            Settings(model_config_path="unused", url_timeout_seconds=0.1),
        )
    assert exc.value.status == 504
    assert download_wire.clock[0] == pytest.approx(0.1)
    assert wire.closed
    if hasattr(wire, "file"):
        assert wire.file.closed


@pytest.mark.parametrize("chunked", [False, True])
def test_url_closed_connection_keeps_real_socket_file_readable(monkeypatch, image, chunked):
    import ai_inference.service.inputs as module

    client_sock, server_sock = socket.socketpair()
    errors = []

    class LocalSocket:
        def __getattr__(self, name):
            return getattr(client_sock, name)

        def setsockopt(self, *args):
            # The local stream exercises real SocketIO without TCP-specific options.
            pass

    def serve():
        try:
            with server_sock:
                server_sock.settimeout(2)
                request = b""
                while b"\r\n\r\n" not in request:
                    piece = server_sock.recv(4096)
                    if not piece:
                        raise RuntimeError("Client closed before sending headers")
                    request += piece
                headers = b"HTTP/1.1 200 OK\r\nConnection: close\r\n"
                if chunked:
                    headers += b"Transfer-Encoding: chunked\r\n\r\n"
                    body = f"{len(image):x}\r\n".encode() + image + b"\r\n0\r\n\r\n"
                else:
                    headers += f"Content-Length: {len(image)}\r\n\r\n".encode()
                    body = image
                server_sock.sendall(headers + body)
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(module, "public_address", lambda *args: "8.8.8.8")
    monkeypatch.setattr(socket, "create_connection", lambda *args: LocalSocket())
    worker = threading.Thread(target=serve)
    worker.start()
    try:
        assert (
            download_image("http://example.com/a.png", Settings(model_config_path="unused"))
            == image
        )
    finally:
        client_sock.close()
        worker.join(3)
    assert not worker.is_alive()
    assert not errors


def post_image(client, source, image, keypoints=None):
    metadata = {} if keypoints is None else {"keypoints": keypoints}
    if source == "file":
        return client.post(
            "/v1/predict/file",
            files={"files": ("a.png", image)},
            data={"metadata": json.dumps([metadata])},
        )
    if source == "bytes":
        return client.post(
            "/v1/predict/bytes",
            content=image,
            headers={"Content-Type": "image/png"},
            params={"metadata": json.dumps(metadata)},
        )
    value = (
        {"data": base64.b64encode(image).decode()}
        if source == "base64"
        else {"url": "https://example.com/image.png"}
    )
    return client.post(f"/v1/predict/{source}", json={"images": [{**value, **metadata}]})


@pytest.mark.parametrize("source", ["file", "base64", "url", "bytes"])
@pytest.mark.parametrize("face", [False, True])
def test_invalid_keypoints_rejected_before_model(source, face, image, monkeypatch):
    monkeypatch.setattr("ai_inference.service.app.download_image", lambda *args: image)

    def factory(path):
        model = FakeModel(path)
        if face:
            model.config.preprocess["steps"] = [{"face_align": {"aligned_input": False}}]
        return model

    app = create_app(Settings(model_config_path="unused"), model_factory=factory)
    points = [[1, 1]] * 5 if face else [[1, 1], [2, 1], [1.5, 1], [1, 2], [2, 2]]
    with TestClient(app) as client:
        response = post_image(client, source, image, points)
        assert response.status_code == 422
        assert response.json()["error"]["index"] == 0
        assert not app.state.runtime.model.calls
        assert client.get("/health/ready").status_code == 200
        if not face:
            assert post_image(client, source, image).status_code == 200


def test_face_batch_validation_is_atomic_and_aligned_crops_are_checked(image):
    def factory(path):
        model = FakeModel(path)
        model.config.preprocess["steps"] = [{"face_align": {"aligned_input": True}}]
        return model

    app = create_app(Settings(model_config_path="unused"), model_factory=factory)
    encoded = base64.b64encode(image).decode()
    points = [[1, 1], [2, 1], [1.5, 1], [1, 2], [2, 2]]
    with TestClient(app) as client:
        response = client.post(
            "/v1/predict/base64",
            json={"images": [{"data": encoded, "keypoints": points}, {"data": encoded}]},
        )
        assert response.status_code == 422
        assert response.json()["error"]["index"] == 1
        assert not app.state.runtime.model.calls
        aligned = BytesIO()
        Image.new("RGB", (112, 112)).save(aligned, format="PNG")
        assert post_image(client, "bytes", aligned.getvalue()).status_code == 200


def test_url_timeout_releases_admission(application, image, monkeypatch):
    def timeout(*args):
        raise InputError("Image download timed out", 504)

    monkeypatch.setattr("ai_inference.service.app.download_image", timeout)
    with TestClient(application) as client:
        assert post_image(client, "url", image).status_code == 504
        assert post_image(client, "bytes", image).status_code == 200


def test_feature_response(application, image):
    with TestClient(application) as client:
        application.state.runtime.model.predict = lambda images: [
            FeaturePrediction(np.array([0.5, 0.5]), (2, 3))
        ]
        response = client.post(
            "/v1/predict/bytes", content=image, headers={"Content-Type": "image/png"}
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["embedding"] == [0.5, 0.5]


def test_face_metadata_and_required_points(image, monkeypatch):
    from ai_inference import FaceInput

    monkeypatch.setattr("ai_inference.service.app.download_image", lambda url, settings: image)

    seen = []

    def factory(path):
        model = FakeModel(path)
        model.config.preprocess["steps"] = [{"face_align": {"aligned_input": False}}]

        def predict(images):
            seen.extend(images)
            return [FeaturePrediction(np.array([1.0]), sample.image.shape[:2]) for sample in images]

        model.predict = predict
        return model

    app = create_app(Settings(model_config_path="unused"), model_factory=factory)
    points = [[1, 1], [2, 1], [1.5, 1], [1, 2], [2, 2]]
    import json

    with TestClient(app) as client:
        response = client.post(
            "/v1/predict/bytes", content=image, headers={"Content-Type": "image/png"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["index"] == 0
        for response in [
            client.post(
                "/v1/predict/bytes",
                content=image,
                headers={"Content-Type": "image/png"},
                params={"metadata": json.dumps({"keypoints": points})},
            ),
            client.post(
                "/v1/predict/file",
                files={"files": ("a.png", image)},
                data={"metadata": json.dumps([{"keypoints": points}])},
            ),
            client.post(
                "/v1/predict/base64",
                json={"images": [{"data": base64.b64encode(image).decode(), "keypoints": points}]},
            ),
            client.post(
                "/v1/predict/url",
                json={"images": [{"url": "https://example.com/face.png", "keypoints": points}]},
            ),
        ]:
            assert response.status_code == 200
        assert len(seen) == 4
        assert all(isinstance(sample, FaceInput) for sample in seen)
        assert all(np.array_equal(sample.keypoints, points) for sample in seen)


def test_color_mismatch_rejects_startup():
    with pytest.raises(ValueError, match="input_color"):
        with TestClient(
            create_app(
                Settings(model_config_path="unused", input_color="RGB"), model_factory=FakeModel
            )
        ):
            pass


def test_env_settings(monkeypatch):
    monkeypatch.setenv("AI_INFERENCE_MODEL_CONFIG", "configs/yolo11.yaml")
    monkeypatch.setenv("AI_INFERENCE_MAX_BATCH_SIZE", "2")
    settings = Settings.from_env()
    assert str(settings.model_config_path) == "configs/yolo11.yaml"
    assert settings.max_batch_size == 2
