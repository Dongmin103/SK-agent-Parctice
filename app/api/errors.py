from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas import ErrorResponse
from app.domain.compound import InvalidSmilesError

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AppServiceError(Exception):
    code: str
    message: str
    status_code: int = 400
    details: object | None = None

    def __str__(self) -> str:
        return self.message


def _sanitize_request_path(request: Request) -> str:
    return request.url.path.split("?", 1)[0]


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(InvalidSmilesError)
    async def handle_invalid_smiles_error(
        request: Request,
        exc: InvalidSmilesError,
    ) -> JSONResponse:
        del request
        payload = ErrorResponse(
            error={
                "code": "invalid_smiles",
                "message": str(exc),
            }
        )
        return JSONResponse(status_code=400, content=payload.model_dump(exclude_none=True))

    @app.exception_handler(AppServiceError)
    async def handle_app_service_error(
        request: Request,
        exc: AppServiceError,
    ) -> JSONResponse:
        del request
        payload = ErrorResponse(
            error={
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump(exclude_none=True))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request
        payload = ErrorResponse(
            error={
                "code": "request_validation_error",
                "message": "Request validation failed.",
                "details": exc.errors(),
            }
        )
        return JSONResponse(status_code=422, content=payload.model_dump(exclude_none=True))

    @app.exception_handler(ValueError)
    async def handle_bad_request_error(
        request: Request,
        exc: ValueError,
    ) -> JSONResponse:
        del request
        payload = ErrorResponse(
            error={
                "code": "bad_request",
                "message": str(exc),
            }
        )
        return JSONResponse(status_code=400, content=payload.model_dump(exclude_none=True))

    @app.exception_handler(Exception)
    async def handle_internal_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        LOGGER.exception(
            "Unhandled API error for %s %s",
            request.method,
            _sanitize_request_path(request),
            exc_info=exc,
        )
        payload = ErrorResponse(
            error={
                "code": "internal_error",
                "message": "An unexpected error occurred.",
            }
        )
        return JSONResponse(status_code=500, content=payload.model_dump(exclude_none=True))
