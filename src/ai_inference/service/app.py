"""FastAPI entry point: one process, one model, one admitted inference request."""

from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from ai_inference import Model

from .inputs import InputError, decode_base64, download_image
from .runtime import Runtime
from .schemas import Base64Request, ImageMetadata, PredictResponse, URLRequest
from .settings import Settings


def error_response(status, message, index=None):
    return JSONResponse(status_code=status, content={"error": {"message": message, "index": index}})


class RequestLimits:
    """Apply admission and body limits before JSON/multipart parsing, including chunked bodies."""

    def __init__(self, app, owner):
        self.app, self.owner = app, owner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        runtime = getattr(self.owner.state, "runtime", None)
        prediction = scope["path"].startswith("/v1/predict/")
        if not prediction:
            return await self.app(scope, receive, send)
        admitted = False
        if prediction:
            if runtime is None:
                return await error_response(503, "Model is not ready")(scope, receive, send)
            admitted = runtime.lock.acquire(blocking=False)
            if not admitted:
                return await error_response(503, "Model is busy; retry later")(scope, receive, send)
        try:
            chunks, size = [], 0
            settings = self.owner.state.settings
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                size += len(message.get("body", b""))
                if size > settings.max_request_bytes:
                    return await error_response(413, "Request exceeds byte limit")(
                        scope, receive, send
                    )
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            delivered = False

            async def replay():
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}

            await self.app(scope, replay, send)
        finally:
            if admitted:
                runtime.lock.release()


def create_app(settings: Settings | None = None, *, model_factory=Model) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        resolved = settings if settings is not None else Settings.from_env()
        app.state.settings = resolved
        # Model already honors runtime.warmup; do not warm up twice.
        app.state.runtime = await run_in_threadpool(Runtime, resolved, model_factory)
        try:
            yield
        finally:
            app.state.runtime = None
            # The SDK currently has no public close() contract. Drop our model reference.

    app = FastAPI(title="Single-model inference", version="1.0", lifespan=lifespan)
    app.state.runtime = None
    app.add_middleware(RequestLimits, owner=app)

    @app.exception_handler(InputError)
    async def input_error(request, exc):
        return error_response(exc.status, str(exc), exc.index)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Avoid echoing submitted image bytes, base64 payloads, or URLs.
        return error_response(422, "Invalid request fields; see /docs for the input schema")

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error_response(exc.status_code, str(exc.detail))

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready():
        if app.state.runtime is None:
            raise HTTPException(503, "Model is not ready")
        return {"status": "ready"}

    @app.get("/v1/model")
    async def model_info():
        runtime = app.state.runtime
        if runtime is None:
            raise HTTPException(503, "Model is not ready")
        config = runtime.model.config
        return {
            "name": config.model.name,
            "backend": config.runtime.backend,
            "input_color": app.state.settings.input_color,
            "inputs": [spec.model_dump() for spec in config.inputs or ()],
            "outputs": [spec.model_dump() for spec in config.outputs or ()],
            "batch": config.runtime.batch.model_dump(),
            "limits": app.state.settings.model_dump(exclude={"model_config_path"}),
        }

    async def execute(sources, metadata):
        try:
            return await run_in_threadpool(app.state.runtime.predict, sources, metadata)
        except InputError:
            raise
        except Exception:
            logger.exception("HTTP inference failed")
            raise HTTPException(500, "Model inference failed") from None

    @app.post(
        "/v1/predict/base64", response_model=PredictResponse, response_model_exclude_none=True
    )
    async def predict_base64(payload: Base64Request):
        sources = [
            partial(decode_base64, image.data, app.state.settings) for image in payload.images
        ]
        return await execute(sources, payload.images)

    @app.post("/v1/predict/url", response_model=PredictResponse, response_model_exclude_none=True)
    async def predict_url(payload: URLRequest):
        sources = [
            partial(download_image, image.url, app.state.settings) for image in payload.images
        ]
        return await execute(sources, payload.images)

    @app.post("/v1/predict/file", response_model=PredictResponse, response_model_exclude_none=True)
    async def predict_file(files: list[UploadFile] = File(...), metadata: str | None = Form(None)):
        try:
            infos = (
                [ImageMetadata() for _ in files]
                if metadata is None
                else TypeAdapter(list[ImageMetadata]).validate_json(metadata)
            )
            sources = [
                partial(file.file.read, app.state.settings.max_image_bytes + 1) for file in files
            ]
            return await execute(sources, infos)
        except ValidationError:
            raise InputError("Invalid image metadata") from None
        finally:
            for file in files:
                await file.close()

    @app.post("/v1/predict/bytes", response_model=PredictResponse, response_model_exclude_none=True)
    async def predict_bytes(request: Request, metadata: str | None = None):
        media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if media_type not in {
            "application/octet-stream",
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp",
        }:
            raise HTTPException(415, "Expected encoded image bytes")
        try:
            info = ImageMetadata.model_validate_json("{}" if metadata is None else metadata)
        except ValidationError:
            raise InputError("keypoints must be a JSON object containing five points") from None
        data = await request.body()
        return await execute([lambda: data], [info])

    return app


app = create_app()
