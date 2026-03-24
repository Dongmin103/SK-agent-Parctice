from __future__ import annotations

import json
import os
from typing import Any

import boto3

from app.domain.prediction_registry import (
    CORE_PREDICTION_PROPERTY_SPECS,
    PredictionPropertySpec,
    TxGemmaPredictionPropertyRegistry,
    get_core_prediction_specs,
)
from app.domain.models import PredictionBundle, PredictionSignal

TXGEMMA_ENDPOINT_ENV_VARS = (
    "TXGEMMA_SAGEMAKER_ENDPOINT_NAME",
    "TXGEMMA_ENDPOINT_NAME",
)
TXGEMMA_REGION_ENV_VARS = ("TXGEMMA_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")


class TxGemmaClient:
    """Deterministic SageMaker adapter for TxGemma predictions."""

    def __init__(
        self,
        *,
        endpoint_name: str,
        region_name: str | None = None,
        content_type: str = "application/json",
        accept: str = "application/json",
        runtime_client: Any | None = None,
        property_specs: tuple[PredictionPropertySpec, ...] | None = None,
        expected_signal_names: tuple[str, ...] = (),
    ) -> None:
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        self.content_type = content_type
        self.accept = accept
        self._runtime_client = runtime_client
        self.property_specs = property_specs or get_core_prediction_specs(
            expected_signal_names or None
        )
        self.registry = TxGemmaPredictionPropertyRegistry(self.property_specs)
        self.expected_signal_names = tuple(spec.name for spec in self.property_specs)

    def predict(
        self,
        *,
        smiles: str,
        target: str | None = None,
        compound_name: str | None = None,
    ) -> PredictionBundle:
        signals: list[PredictionSignal] = []
        missing_signals: list[str] = []
        canonical_smiles = smiles

        for spec in self.property_specs:
            prompt = spec.render_prompt(
                smiles=smiles,
                target=target,
                compound_name=compound_name,
            )
            try:
                response_text = self._invoke_endpoint(prompt, spec)
                signal = self.registry.normalize_response(
                    spec_name=spec.name,
                    raw_text=response_text,
                )
            except Exception:
                missing_signals.append(spec.name)
                continue

            signals.append(signal)

        return PredictionBundle(
            source="txgemma",
            target=target,
            compound_name=compound_name,
            canonical_smiles=canonical_smiles,
            signals=signals,
            missing_signals=missing_signals,
        )

    def _build_request_payload(
        self,
        prompt: str,
        spec: PredictionPropertySpec,
    ) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "max_new_tokens": 64,
            "do_sample": False,
            "return_full_text": False,
        }
        grammar = self._build_grammar(spec)
        if grammar is not None:
            parameters["grammar"] = grammar

        return {
            "inputs": prompt,
            "parameters": parameters,
        }

    def _build_grammar(self, spec: PredictionPropertySpec) -> dict[str, Any] | None:
        if not spec.constrain_with_grammar or not spec.answer_options:
            return None

        return {
            "type": "json",
            "value": {
                "type": "object",
                "properties": {
                    "property": {
                        "type": "string",
                        "enum": [spec.name],
                    },
                    "answer": {
                        "type": "string",
                        "enum": [option.token for option in spec.answer_options],
                    },
                },
                "required": ["property", "answer"],
            },
        }

    def _invoke_endpoint(self, prompt: str, spec: PredictionPropertySpec) -> str:
        payload = self._build_request_payload(prompt, spec)
        response = self._runtime_client_or_create().invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType=self.content_type,
            Accept=self.accept,
            Body=json.dumps(payload).encode("utf-8"),
        )
        body = response.get("Body")
        if body is None or not hasattr(body, "read"):
            raise ValueError("prediction response body is missing")

        raw = body.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        parsed = json.loads(text)
        return self._extract_generated_text(parsed)

    def _runtime_client_or_create(self) -> Any:
        if self._runtime_client is None:
            self._runtime_client = boto3.client(
                "sagemaker-runtime",
                region_name=self.region_name,
            )
        return self._runtime_client

    def _extract_generated_text(self, payload: Any) -> str:
        text = self._unwrap_prediction_payload(payload)
        if text is not None:
            return text
        raise ValueError("prediction payload must include generated_text")

    def _unwrap_prediction_payload(self, payload: Any) -> str | None:
        if isinstance(payload, str):
            return payload

        if isinstance(payload, list):
            for item in payload:
                text = self._unwrap_prediction_payload(item)
                if text is not None:
                    return text
            return None

        if not isinstance(payload, dict):
            return None

        for key in ("generated_text", "text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
            text = self._unwrap_prediction_payload(value)
            if text is not None:
                return text

        if "answer" in payload or "value" in payload:
            return json.dumps(payload)

        for key in ("predictions", "outputs", "results", "data", "response", "body", "choices", "message", "content"):
            if key not in payload:
                continue
            text = self._unwrap_prediction_payload(payload[key])
            if text is not None:
                return text

        return None


def build_txgemma_client(
    *,
    endpoint_name: str | None = None,
    region_name: str | None = None,
    expected_signal_names: tuple[str, ...] = (),
) -> TxGemmaClient:
    resolved_endpoint_name = endpoint_name or next(
        (os.environ.get(name) for name in TXGEMMA_ENDPOINT_ENV_VARS if os.environ.get(name)),
        None,
    )
    if not resolved_endpoint_name:
        raise ValueError(
            "A TxGemma endpoint name is required. Set "
            "TXGEMMA_SAGEMAKER_ENDPOINT_NAME or TXGEMMA_ENDPOINT_NAME, "
            "or pass endpoint_name."
        )

    resolved_region_name = region_name or next(
        (os.environ.get(name) for name in TXGEMMA_REGION_ENV_VARS if os.environ.get(name)),
        None,
    )

    return TxGemmaClient(
        endpoint_name=resolved_endpoint_name,
        region_name=resolved_region_name,
        property_specs=get_core_prediction_specs(expected_signal_names or None),
        expected_signal_names=expected_signal_names,
    )
