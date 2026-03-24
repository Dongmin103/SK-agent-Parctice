from __future__ import annotations

import io
import json
import os

import pytest

from app.domain.models import PredictionBundle


class FakeSageMakerRuntimeClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self._remaining = list(response) if isinstance(response, list) else None
        self._last_payload: object | None = None
        self.calls: list[dict[str, object]] = []

    def invoke_endpoint(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if self._remaining is not None:
            if self._remaining:
                payload = self._remaining.pop(0)
                self._last_payload = payload
            elif self._last_payload is not None:
                payload = self._last_payload
            else:
                payload = self.response
        else:
            payload = self.response
        body = json.dumps(payload).encode("utf-8")
        return {"Body": io.BytesIO(body)}


def make_client() -> object:
    from app.clients.txgemma import TxGemmaClient
    from app.domain.prediction_registry import get_core_prediction_specs

    return TxGemmaClient(
        endpoint_name="txgemma-test-endpoint",
        region_name="ap-northeast-2",
        property_specs=get_core_prediction_specs(("Solubility", "hERG")),
    )


def test_predict_txgemma_aggregates_property_level_calls(monkeypatch) -> None:
    from app.clients.txgemma import TxGemmaClient
    from app.domain.prediction_registry import get_core_prediction_specs

    specs = tuple(
        spec for spec in get_core_prediction_specs() if spec.name in ("Solubility", "hERG", "DILI")
    )
    client = TxGemmaClient(
        endpoint_name="txgemma-test-endpoint",
        region_name="ap-northeast-2",
        property_specs=specs,
    )
    runtime = FakeSageMakerRuntimeClient(
        [
            [{"generated_text": "-1.25"}],
            [{"generated_text": "(A)"}],
            [{"generated_text": "(B)"}],
        ]
    )

    monkeypatch.setattr(
        "boto3.client",
        lambda service_name, region_name=None: runtime if service_name == "sagemaker-runtime" else None,
    )

    bundle = client.predict(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target="KRAS G12C",
        compound_name="Aspirin",
    )

    assert [signal.name for signal in bundle.signals] == ["Solubility", "hERG", "DILI"]
    assert bundle.missing_signals == []
    assert len(runtime.calls) == 3
    prompt_texts = [json.loads(call["Body"].decode("utf-8"))["inputs"] for call in runtime.calls]
    assert "solubility" in prompt_texts[0].lower()
    assert "hERG" in prompt_texts[1]
    assert "liver" in prompt_texts[2].lower() or "dili" in prompt_texts[2].lower()


def test_predict_txgemma_normalizes_prediction_bundle(monkeypatch) -> None:
    client = make_client()
    runtime = FakeSageMakerRuntimeClient(
        [
            [{"generated_text": "-3.25"}],
            [{"generated_text": "(B)"}],
        ]
    )

    monkeypatch.setattr(
        "boto3.client",
        lambda service_name, region_name=None: runtime if service_name == "sagemaker-runtime" else None,
    )

    bundle = client.predict(
        smiles="N#CC1=CC=CC=C1",
        target="KRAS G12C",
        compound_name="ABC-101",
    )

    assert isinstance(bundle, PredictionBundle)
    assert bundle.source == "txgemma"
    assert bundle.target == "KRAS G12C"
    assert bundle.compound_name == "ABC-101"
    assert bundle.canonical_smiles == "N#CC1=CC=CC=C1"
    assert bundle.missing_signals == []
    assert [signal.name for signal in bundle.signals] == ["Solubility", "hERG"]
    assert bundle.signals[0].value == "Solubility -3.25"
    assert bundle.signals[0].unit == "log mol/L"
    assert bundle.signals[0].confidence is None
    assert bundle.signals[0].risk_level == "medium"
    assert runtime.calls[0]["EndpointName"] == "txgemma-test-endpoint"
    request_payload = json.loads(runtime.calls[0]["Body"].decode("utf-8"))
    assert request_payload["inputs"]
    assert "N#CC1=CC=CC=C1" in request_payload["inputs"]
    assert "KRAS G12C" in request_payload["inputs"]
    assert "ABC-101" in request_payload["inputs"]
    assert isinstance(request_payload["parameters"], dict)


def test_predict_txgemma_records_missing_signals(monkeypatch) -> None:
    from app.clients.txgemma import TxGemmaClient
    from app.domain.prediction_registry import get_core_prediction_specs

    client = TxGemmaClient(
        endpoint_name="txgemma-test-endpoint",
        region_name="ap-northeast-2",
        property_specs=get_core_prediction_specs(("Solubility", "Lipophilicity", "HIA", "hERG")),
    )
    runtime = FakeSageMakerRuntimeClient(
        [
            [{"generated_text": "(B)"}],
            [{"generated_text": "not-json"}],
            [{"generated_text": "-1.25"}],
            [{"generated_text": "(A)"}],
        ]
    )

    monkeypatch.setattr(
        "boto3.client",
        lambda service_name, region_name=None: runtime if service_name == "sagemaker-runtime" else None,
    )

    bundle = client.predict(smiles="N#CC1=CC=CC=C1", target="KRAS G12C")

    assert bundle.missing_signals == ["Lipophilicity"]
    assert [signal.name for signal in bundle.signals] == ["HIA", "Solubility", "hERG"]


def test_predict_txgemma_rejects_malformed_payload(monkeypatch) -> None:
    client = make_client()

    class MalformedRuntimeClient:
        def invoke_endpoint(self, **kwargs: object) -> dict[str, object]:
            return {"Body": io.BytesIO(b'[{"generated_text":"not-json"}]')}

    monkeypatch.setattr(
        "boto3.client",
        lambda service_name, region_name=None: MalformedRuntimeClient() if service_name == "sagemaker-runtime" else None,
    )

    bundle = client.predict(smiles="N#CC1=CC=CC=C1", target="KRAS G12C")

    assert bundle.signals == []
    assert bundle.missing_signals == ["Solubility", "hERG"]


def test_predict_txgemma_extracts_json_from_markdown_code_fence(monkeypatch) -> None:
    client = make_client()
    runtime = FakeSageMakerRuntimeClient(
        [
            {
                "generated_text": """```json
{"property":"Solubility","answer":"-1.25"}
```"""
            },
            {"generated_text": """```json
{"property":"hERG","answer":"A"}
```"""},
        ]
    )

    monkeypatch.setattr(
        "boto3.client",
        lambda service_name, region_name=None: runtime if service_name == "sagemaker-runtime" else None,
    )

    bundle = client.predict(smiles="N#CC1=CC=CC=C1", target="KRAS G12C")

    assert bundle.canonical_smiles == "N#CC1=CC=CC=C1"
    assert bundle.signals[0].name == "Solubility"
    assert bundle.signals[1].name == "hERG"


def test_predict_txgemma_accepts_sagemaker_predictions_wrapper(monkeypatch) -> None:
    client = make_client()
    runtime = FakeSageMakerRuntimeClient(
        [
            {"predictions": [{"generated_text": '{"property":"Solubility","answer":"-2.50"}'}]},
            {"predictions": [{"generated_text": '{"property":"hERG","answer":"A"}'}]},
        ]
    )

    monkeypatch.setattr(
        "boto3.client",
        lambda service_name, region_name=None: runtime if service_name == "sagemaker-runtime" else None,
    )

    bundle = client.predict(smiles="N#CC1=CC=CC=C1", target="KRAS G12C")

    assert bundle.missing_signals == []
    assert [signal.name for signal in bundle.signals] == ["Solubility", "hERG"]
    assert bundle.signals[0].value == "Solubility -2.50"
    assert bundle.signals[1].value == "low hERG inhibition risk"


def test_predict_txgemma_accepts_direct_structured_answer_payload(monkeypatch) -> None:
    client = make_client()
    runtime = FakeSageMakerRuntimeClient(
        [
            {"property": "Solubility", "answer": "-0.75"},
            {"property": "hERG", "answer": "B"},
        ]
    )

    monkeypatch.setattr(
        "boto3.client",
        lambda service_name, region_name=None: runtime if service_name == "sagemaker-runtime" else None,
    )

    bundle = client.predict(smiles="N#CC1=CC=CC=C1", target="KRAS G12C")

    assert bundle.missing_signals == []
    assert [signal.name for signal in bundle.signals] == ["Solubility", "hERG"]
    assert bundle.signals[0].value == "Solubility -0.75"
    assert bundle.signals[1].value == "high hERG inhibition risk"


def test_predict_txgemma_limits_grammar_to_recovery_tasks_and_keeps_herg_unconstrained(monkeypatch) -> None:
    client = make_client()
    runtime = FakeSageMakerRuntimeClient(
        [
            [{"generated_text": '{"property":"Solubility","answer":"-2.25"}'}],
            [{"generated_text": '{"property":"hERG","answer":"B"}'}],
        ]
    )

    monkeypatch.setattr(
        "boto3.client",
        lambda service_name, region_name=None: runtime if service_name == "sagemaker-runtime" else None,
    )

    bundle = client.predict(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target="KRAS G12C",
        compound_name="Aspirin",
    )

    first_payload = json.loads(runtime.calls[0]["Body"].decode("utf-8"))
    second_payload = json.loads(runtime.calls[1]["Body"].decode("utf-8"))

    assert bundle.missing_signals == []
    assert bundle.signals[0].name == "Solubility"
    assert first_payload["parameters"]["grammar"]["value"]["properties"]["property"]["enum"] == ["Solubility"]
    assert "grammar" not in second_payload["parameters"]


def test_build_txgemma_client_defaults_to_all_tdc_admet_twenty_two_tasks(monkeypatch) -> None:
    from app.clients.txgemma import build_txgemma_client

    monkeypatch.setenv(
        "TXGEMMA_SAGEMAKER_ENDPOINT_NAME",
        "huggingface-pytorch-tgi-inference-2026-03-23-07-40-06-070",
    )
    monkeypatch.setenv("TXGEMMA_AWS_REGION", "ap-southeast-2")

    client = build_txgemma_client()

    assert len(client.property_specs) == 22
    assert client.expected_signal_names == tuple(spec.name for spec in client.property_specs)


def test_predict_txgemma_attaches_grammar_to_tuned_live_recovery_tasks(monkeypatch) -> None:
    from app.clients.txgemma import TxGemmaClient
    from app.domain.prediction_registry import get_core_prediction_specs

    runtime = FakeSageMakerRuntimeClient(
        [
            [{"generated_text": '{"property":"Solubility","answer":"B"}'}],
            [{"generated_text": '{"property":"PPBR","answer":"B"}'}],
            [{"generated_text": '{"property":"hERG","answer":"A"}'}],
        ]
    )
    client = TxGemmaClient(
        endpoint_name="txgemma-test-endpoint",
        region_name="ap-northeast-2",
        runtime_client=runtime,
        property_specs=get_core_prediction_specs(("Solubility", "PPBR", "hERG")),
    )

    bundle = client.predict(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target="KRAS G12C",
        compound_name="Aspirin",
    )

    assert bundle.missing_signals == []
    first_payload = json.loads(runtime.calls[0]["Body"].decode("utf-8"))
    second_payload = json.loads(runtime.calls[1]["Body"].decode("utf-8"))
    third_payload = json.loads(runtime.calls[2]["Body"].decode("utf-8"))

    assert first_payload["parameters"]["grammar"]["value"]["properties"]["property"]["enum"] == ["Solubility"]
    assert second_payload["parameters"]["grammar"]["value"]["properties"]["property"]["enum"] == ["PPBR"]
    assert "grammar" not in third_payload["parameters"]


def test_build_txgemma_client_uses_endpoint_and_region_from_environment(monkeypatch) -> None:
    from app.clients.txgemma import build_txgemma_client

    monkeypatch.setenv(
        "TXGEMMA_SAGEMAKER_ENDPOINT_NAME",
        "huggingface-pytorch-tgi-inference-2026-03-23-07-40-06-070",
    )
    monkeypatch.setenv("TXGEMMA_AWS_REGION", "ap-southeast-2")

    client = build_txgemma_client()

    assert client.endpoint_name == "huggingface-pytorch-tgi-inference-2026-03-23-07-40-06-070"
    assert client.region_name == "ap-southeast-2"


def test_build_txgemma_client_falls_back_to_aws_default_region(monkeypatch) -> None:
    from app.clients.txgemma import build_txgemma_client

    monkeypatch.setenv(
        "TXGEMMA_SAGEMAKER_ENDPOINT_NAME",
        "huggingface-pytorch-tgi-inference-2026-03-23-07-40-06-070",
    )
    monkeypatch.delenv("TXGEMMA_AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")

    client = build_txgemma_client()

    assert client.region_name == "ap-southeast-2"


def test_build_txgemma_client_requires_endpoint_name(monkeypatch) -> None:
    from app.clients.txgemma import build_txgemma_client

    monkeypatch.delenv("TXGEMMA_SAGEMAKER_ENDPOINT_NAME", raising=False)
    monkeypatch.delenv("TXGEMMA_ENDPOINT_NAME", raising=False)

    with pytest.raises(ValueError, match="endpoint"):
        build_txgemma_client()


def test_build_txgemma_client_supports_legacy_endpoint_env_name(monkeypatch) -> None:
    from app.clients.txgemma import build_txgemma_client

    monkeypatch.setenv(
        "TXGEMMA_ENDPOINT_NAME",
        "huggingface-pytorch-tgi-inference-2026-03-23-07-40-06-070",
    )
    monkeypatch.setenv("TXGEMMA_AWS_REGION", "ap-southeast-2")

    client = build_txgemma_client()

    assert client.endpoint_name == "huggingface-pytorch-tgi-inference-2026-03-23-07-40-06-070"
    assert client.region_name == "ap-southeast-2"


def test_build_txgemma_client_missing_endpoint_error_mentions_supported_env_names(monkeypatch) -> None:
    from app.clients.txgemma import build_txgemma_client

    monkeypatch.delenv("TXGEMMA_SAGEMAKER_ENDPOINT_NAME", raising=False)
    monkeypatch.delenv("TXGEMMA_ENDPOINT_NAME", raising=False)

    with pytest.raises(
        ValueError,
        match="TXGEMMA_SAGEMAKER_ENDPOINT_NAME|TXGEMMA_ENDPOINT_NAME",
    ):
        build_txgemma_client(region_name="ap-southeast-2")


@pytest.mark.sagemaker_live
def test_live_txgemma_endpoint_returns_prediction_bundle() -> None:
    from app.clients.txgemma import build_txgemma_client

    endpoint_name = os.environ.get("TXGEMMA_SAGEMAKER_ENDPOINT_NAME") or os.environ.get(
        "TXGEMMA_ENDPOINT_NAME"
    )
    region_name = (
        os.environ.get("TXGEMMA_AWS_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    if not endpoint_name or not region_name:
        pytest.skip(
            "Set TXGEMMA_SAGEMAKER_ENDPOINT_NAME or TXGEMMA_ENDPOINT_NAME and "
            "TXGEMMA_AWS_REGION/AWS_REGION/AWS_DEFAULT_REGION to run this test."
        )

    client = build_txgemma_client()
    client.expected_signal_names = (
        "Caco2",
        "Bioavailability",
        "HIA",
        "Pgp",
        "Lipophilicity",
        "Solubility",
        "BBB",
        "PPBR",
        "VDss",
        "CYP2C9 inhibition",
        "CYP2D6 inhibition",
        "CYP3A4 inhibition",
        "CYP2C9 substrate",
        "CYP2D6 substrate",
        "CYP3A4 substrate",
        "Half-life",
        "Clearance hepatocyte",
        "Clearance microsome",
        "LD50",
        "hERG",
        "AMES",
        "DILI",
    )

    bundle = client.predict(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target="KRAS G12C",
        compound_name="Aspirin",
    )

    assert isinstance(bundle, PredictionBundle)
    assert bundle.source == "txgemma"
    assert len(bundle.signals) + len(bundle.missing_signals) == 22
