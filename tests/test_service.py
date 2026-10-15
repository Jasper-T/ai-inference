"""HTTP boundary regression tests; no real model weights or network required."""

import base64
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
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


def test_url_pins_address_and_rejects_redirects(monkeypatch, image):
    import ai_inference.service.inputs as module

    settings = Settings(model_config_path="unused")
    monkeypatch.setattr(module, "public_address", lambda *a: "8.8.8.8")
    connections = []
    destinations = []
    monkeypatch.setattr(
        socket, "create_connection", lambda target, timeout, source: destinations.append(target)
    )

    class Connection:
        status = 200
        sock = None

        def __init__(self, host, port, **kwargs):
            self.host = host
            self.remaining = image
            self.closed = False
            connections.append(self)

        def request(self, *args, **kwargs):
            self._create_connection((self.host, 443), 10)

        def getresponse(self):
            return self

        def getheader(self, name, default=None):
            return default

        def read1(self, size):
            data, self.remaining = self.remaining[:size], self.remaining[size:]
            return data

        def close(self):
            self.closed = True

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    assert download_image("https://example.com/a.png", settings) == image
    assert destinations == [("8.8.8.8", 443)]
    assert connections[0].host == "example.com"
    assert connections[0].closed
    Connection.status = 302
    with pytest.raises(InputError, match="redirects"):
        download_image("https://example.com/a.png", settings)
    assert connections[-1].closed
    Connection.status = 200
    with pytest.raises(InputError) as exc:
        download_image(
            "https://example.com/a.png", settings.model_copy(update={"max_image_bytes": 4})
        )
    assert exc.value.status == 413


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
