from __future__ import annotations

import pytest


def _load_registry_symbols():
    from app.clients.txgemma import (
        CORE_PREDICTION_PROPERTY_SPECS,
        PredictionPropertySpec,
        TxGemmaPredictionPropertyRegistry,
    )

    return CORE_PREDICTION_PROPERTY_SPECS, PredictionPropertySpec, TxGemmaPredictionPropertyRegistry


def test_core_prediction_property_specs_cover_the_tdc_admet_twenty_two_signals_in_order() -> None:
    core_specs, _, _ = _load_registry_symbols()

    canonical_names = [spec.name for spec in core_specs]

    assert canonical_names == [
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
    ]


@pytest.mark.parametrize(
    "spec_name, smiles, target, compound_name",
    [
        ("BBB", "CC(=O)OC1=CC=CC=C1C(=O)O", "CNS target", "Aspirin"),
        ("CYP2D6 inhibition", "N#CC1=CC=CC=C1", "KRAS G12C", "ABC-101"),
        ("Solubility", "CC(=O)OC1=CC=CC=C1C(=O)O", "KRAS G12C", "ABC-101"),
        ("DILI", "CCN(CC)CCOC(=O)C1=CC=CC=C1", "Liver safety", "XYZ-900"),
    ],
)
def test_each_spec_renders_a_prompt_that_mentions_the_key_inputs(
    spec_name: str,
    smiles: str,
    target: str,
    compound_name: str,
) -> None:
    core_specs, _, TxGemmaPredictionPropertyRegistry = _load_registry_symbols()
    registry = TxGemmaPredictionPropertyRegistry(core_specs)

    spec = registry.get(spec_name)
    prompt = spec.render_prompt(
        smiles=smiles,
        target=target,
        compound_name=compound_name,
    )

    assert smiles in prompt
    assert target in prompt
    assert compound_name in prompt
    assert spec.name in prompt
    assert "Return only valid JSON" in prompt


@pytest.mark.parametrize(
    "spec_name, raw_text, expected_value_fragment, expected_risk_level",
    [
        ("BBB", "(B)", "crosses the BBB", "high"),
        ("hERG", "(A)", "low hERG inhibition risk", "low"),
        ("DILI", "(B)", "high DILI risk", "high"),
        ("CYP2D6 inhibition", "(A)", "low CYP2D6 inhibition risk", "low"),
    ],
)
def test_option_letter_responses_normalize_to_signal_text_and_risk_level(
    spec_name: str,
    raw_text: str,
    expected_value_fragment: str,
    expected_risk_level: str,
) -> None:
    core_specs, _, TxGemmaPredictionPropertyRegistry = _load_registry_symbols()
    registry = TxGemmaPredictionPropertyRegistry(core_specs)

    signal = registry.normalize_response(spec_name=spec_name, raw_text=raw_text)

    assert signal.name == spec_name
    assert expected_value_fragment in str(signal.value)
    assert signal.risk_level == expected_risk_level


def test_regression_numeric_response_outside_allowed_range_is_rejected() -> None:
    core_specs, PredictionPropertySpec, TxGemmaPredictionPropertyRegistry = _load_registry_symbols()
    registry = TxGemmaPredictionPropertyRegistry(core_specs)

    # The regression spec is intentionally numeric and bounded.
    spec = registry.get("Solubility")
    assert isinstance(spec, PredictionPropertySpec)

    with pytest.raises(ValueError, match="range"):
        registry.normalize_response(spec_name="Solubility", raw_text="99.0")


def test_solubility_prompt_uses_few_shot_json_examples() -> None:
    core_specs, _, TxGemmaPredictionPropertyRegistry = _load_registry_symbols()
    registry = TxGemmaPredictionPropertyRegistry(core_specs)

    prompt = registry.get("Solubility").render_prompt(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target="KRAS G12C",
        compound_name="Aspirin",
    )

    assert "Example 1:" in prompt
    assert "Drug SMILES: CO" in prompt
    assert 'Answer: {"property":"Solubility","answer":"C"}' in prompt
    assert "Example 2:" in prompt
    assert "Drug SMILES: CCCCCCCC" in prompt
    assert 'Answer: {"property":"Solubility","answer":"A"}' in prompt
    assert "Return only valid JSON" in prompt
    assert "JSON schema:" in prompt


def test_numeric_regression_tasks_preserve_units_and_name_prefix() -> None:
    core_specs, _, TxGemmaPredictionPropertyRegistry = _load_registry_symbols()
    registry = TxGemmaPredictionPropertyRegistry(core_specs)

    signal = registry.normalize_response(spec_name="Caco2", raw_text="1.75")

    assert signal.name == "Caco2"
    assert signal.value == "Caco2 1.75"
    assert signal.unit == "cm/s"
    assert signal.risk_level is None


@pytest.mark.parametrize(
    "spec_name, expected_tokens",
    [
        ("Caco2", ("A", "B", "C")),
        ("Lipophilicity", ("A", "B", "C")),
        ("Solubility", ("A", "B", "C")),
        ("PPBR", ("A", "B", "C")),
        ("LD50", ("A", "B", "C")),
    ],
)
def test_tuned_live_recovery_specs_use_classification_answer_options(
    spec_name: str,
    expected_tokens: tuple[str, ...],
) -> None:
    core_specs, _, TxGemmaPredictionPropertyRegistry = _load_registry_symbols()
    registry = TxGemmaPredictionPropertyRegistry(core_specs)

    spec = registry.get(spec_name)

    assert tuple(option.token for option in spec.answer_options) == expected_tokens
    assert spec.constrain_with_grammar is True


@pytest.mark.parametrize(
    "spec_name, raw_text, expected_value_fragment, expected_risk_level",
    [
        ("Caco2", "(B)", "moderate Caco-2 permeability", "medium"),
        ("Lipophilicity", "(C)", "high lipophilicity", "high"),
        ("Solubility", "(B)", "moderate aqueous solubility", "medium"),
        ("PPBR", "(B)", "moderate plasma protein binding", "medium"),
        ("LD50", "(A)", "low acute toxicity risk", "low"),
    ],
)
def test_tuned_classification_specs_normalize_option_outputs_for_recovered_live_tasks(
    spec_name: str,
    raw_text: str,
    expected_value_fragment: str,
    expected_risk_level: str,
) -> None:
    core_specs, _, TxGemmaPredictionPropertyRegistry = _load_registry_symbols()
    registry = TxGemmaPredictionPropertyRegistry(core_specs)

    signal = registry.normalize_response(spec_name=spec_name, raw_text=raw_text)

    assert signal.name == spec_name
    assert expected_value_fragment in str(signal.value)
    assert signal.risk_level == expected_risk_level
