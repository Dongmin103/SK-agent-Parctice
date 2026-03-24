from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from app.domain.models import PredictionSignal

OPTION_RE = re.compile(r"\(([A-Z0-9])\)|\b([A-Z0-9])\b")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class PredictionChoiceOption:
    token: str
    value: str
    risk_level: str | None = None
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class PredictionPromptExample:
    smiles: str
    answer: str


@dataclass(frozen=True, slots=True)
class PredictionPropertySpec:
    name: str
    context: str
    question: str
    answer_options: tuple[PredictionChoiceOption, ...] = ()
    few_shot_examples: tuple[PredictionPromptExample, ...] = ()
    constrain_with_grammar: bool = False
    unit: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    numeric_value_prefix: str | None = None

    def render_prompt(
        self,
        *,
        smiles: str,
        target: str | None = None,
        compound_name: str | None = None,
    ) -> str:
        if self.few_shot_examples:
            lines = [
                "Instructions: Answer the following question about drug properties.",
                "Return only valid JSON.",
                f"Context: {self.context}",
                f"Question: {self.question}",
                'JSON schema: {"property": "<property name>", "answer": "<option token or numeric value>"}',
            ]
            for index, example in enumerate(self.few_shot_examples, start=1):
                lines.extend(
                    [
                        f"Example {index}:",
                        f"Drug SMILES: {example.smiles}",
                        f'Answer: {{"property":"{self.name}","answer":"{self._example_answer_token(example.answer)}"}}',
                    ]
                )
            lines.append(f"Drug SMILES: {smiles}")
            if target:
                lines.append(f"Target: {target}")
            if compound_name:
                lines.append(f"Compound name: {compound_name}")
            lines.append("Answer:")
            return "\n".join(lines)

        lines = [
            "Instructions: Answer the following question about drug properties.",
            "Return only valid JSON.",
            f"Property: {self.name}",
            f"Context: {self.context}",
            f"Question: {self.question}",
            f"Drug SMILES: {smiles}",
        ]
        if target:
            lines.append(f"Target: {target}")
        if compound_name:
            lines.append(f"Compound name: {compound_name}")
        lines.extend(
            [
                'JSON schema: {"property": "<property name>", "answer": "<option token or numeric value>"}',
                "Answer:",
            ]
        )
        return "\n".join(lines)

    def _example_answer_token(self, answer_text: str) -> str:
        normalized = answer_text.strip()
        match = OPTION_RE.search(normalized.upper())
        if match is None:
            return normalized
        return next(group for group in match.groups() if group)


def _two_choice_property_spec(
    *,
    name: str,
    context: str,
    question: str,
    option_a_value: str,
    option_b_value: str,
    option_a_risk: str | None = None,
    option_b_risk: str | None = None,
) -> PredictionPropertySpec:
    return PredictionPropertySpec(
        name=name,
        context=context,
        question=question,
        answer_options=(
            PredictionChoiceOption("A", option_a_value, option_a_risk),
            PredictionChoiceOption("B", option_b_value, option_b_risk),
        ),
    )


def _three_choice_property_spec(
    *,
    name: str,
    context: str,
    question: str,
    option_a_value: str,
    option_b_value: str,
    option_c_value: str,
    option_a_risk: str | None = None,
    option_b_risk: str | None = None,
    option_c_risk: str | None = None,
    few_shot_examples: tuple[PredictionPromptExample, ...] = (),
    constrain_with_grammar: bool = False,
    unit: str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    numeric_value_prefix: str | None = None,
) -> PredictionPropertySpec:
    return PredictionPropertySpec(
        name=name,
        context=context,
        question=question,
        answer_options=(
            PredictionChoiceOption("A", option_a_value, option_a_risk),
            PredictionChoiceOption("B", option_b_value, option_b_risk),
            PredictionChoiceOption("C", option_c_value, option_c_risk),
        ),
        few_shot_examples=few_shot_examples,
        constrain_with_grammar=constrain_with_grammar,
        unit=unit,
        min_value=min_value,
        max_value=max_value,
        numeric_value_prefix=numeric_value_prefix,
    )


def _numeric_property_spec(
    *,
    name: str,
    context: str,
    question: str,
    unit: str,
    min_value: float,
    max_value: float,
    numeric_value_prefix: str | None = None,
) -> PredictionPropertySpec:
    return PredictionPropertySpec(
        name=name,
        context=context,
        question=question,
        unit=unit,
        min_value=min_value,
        max_value=max_value,
        numeric_value_prefix=numeric_value_prefix or name,
    )


class TxGemmaPredictionPropertyRegistry:
    def __init__(self, specs: Iterable[PredictionPropertySpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def get(self, spec_name: str) -> PredictionPropertySpec:
        try:
            return self._specs[spec_name]
        except KeyError as exc:
            raise KeyError(f"unknown prediction property: {spec_name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def normalize_response(self, *, spec_name: str, raw_text: str) -> PredictionSignal:
        spec = self.get(spec_name)
        answer_text = self._extract_answer_text(raw_text)

        option = self._match_option(spec, answer_text)
        if option is not None:
            return PredictionSignal(
                name=spec.name,
                value=option.value,
                unit=option.unit or spec.unit,
                confidence=None,
                risk_level=option.risk_level,
            )

        numeric_value = self._extract_numeric_value(answer_text)
        if numeric_value is None:
            raise ValueError(f"Unsupported response for {spec_name}: {raw_text!r}")

        if spec.min_value is not None and numeric_value < spec.min_value:
            raise ValueError(f"{spec_name} response is outside the allowed range")
        if spec.max_value is not None and numeric_value > spec.max_value:
            raise ValueError(f"{spec_name} response is outside the allowed range")

        value_prefix = spec.numeric_value_prefix or spec.name
        value_text = f"{value_prefix} {numeric_value:.2f}"
        return PredictionSignal(
            name=spec.name,
            value=value_text,
            unit=spec.unit,
            confidence=None,
            risk_level=self._numeric_risk_level(spec, numeric_value),
        )

    def _extract_answer_text(self, raw_text: str) -> str:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end >= start:
            candidate = cleaned[start : end + 1]
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                return cleaned
            if isinstance(payload, dict):
                answer = payload.get("answer") or payload.get("value")
                if answer is not None:
                    return str(answer).strip()
            return cleaned
        return cleaned

    def _match_option(
        self,
        spec: PredictionPropertySpec,
        answer_text: str,
    ) -> PredictionChoiceOption | None:
        if not spec.answer_options:
            return None

        option_map = {option.token.upper(): option for option in spec.answer_options}
        for match in OPTION_RE.finditer(answer_text.upper()):
            token = next(group for group in match.groups() if group)
            option = option_map.get(token)
            if option is not None:
                return option

        normalized_text = answer_text.strip().lower()
        for option in spec.answer_options:
            if option.value.lower() == normalized_text:
                return option
        return None

    def _extract_numeric_value(self, answer_text: str) -> float | None:
        match = NUMBER_RE.search(answer_text)
        if not match:
            return None
        return float(match.group(0))

    def _numeric_risk_level(self, spec: PredictionPropertySpec, value: float) -> str | None:
        if spec.name == "Solubility":
            if value <= -4:
                return "high"
            if value <= -2:
                return "medium"
            return "low"
        if spec.name == "Lipophilicity":
            if value >= 5:
                return "high"
            if value >= 3:
                return "medium"
            return "low"
        if spec.name == "PPBR":
            if value >= 95:
                return "high"
            if value >= 85:
                return "medium"
            return "low"
        if spec.name == "LD50":
            if value >= 3.5:
                return "high"
            if value >= 2.5:
                return "medium"
            return "low"
        return None


CORE_PREDICTION_PROPERTY_SPECS: tuple[PredictionPropertySpec, ...] = (
    _three_choice_property_spec(
        name="Caco2",
        context="Caco-2 permeability is a proxy for intestinal membrane permeability and oral absorption.",
        question=(
            "Given a drug SMILES string, predict whether its Caco-2 permeability is "
            "(A) low (B) moderate (C) high"
        ),
        option_a_value="low Caco-2 permeability",
        option_b_value="moderate Caco-2 permeability",
        option_c_value="high Caco-2 permeability",
        option_a_risk="high",
        option_b_risk="medium",
        option_c_risk="low",
        few_shot_examples=(
            PredictionPromptExample("CO", "(B)"),
            PredictionPromptExample("CCCCCCCC", "(C)"),
        ),
        constrain_with_grammar=True,
        unit="cm/s",
        min_value=-20.0,
        max_value=20.0,
    ),
    _two_choice_property_spec(
        name="Bioavailability",
        context="Oral bioavailability reflects the fraction of drug reaching systemic circulation.",
        question=(
            "Given a drug SMILES string, predict whether its oral bioavailability is "
            "(A) low (B) high"
        ),
        option_a_value="low oral bioavailability",
        option_b_value="high oral bioavailability",
        option_a_risk="high",
        option_b_risk="low",
    ),
    _two_choice_property_spec(
        name="HIA",
        context="Human intestinal absorption reflects how efficiently a small molecule is absorbed enterally.",
        question=(
            "Given a drug SMILES string, predict whether its human intestinal absorption is "
            "(A) low (B) high"
        ),
        option_a_value="low human intestinal absorption",
        option_b_value="high human intestinal absorption",
        option_a_risk="high",
        option_b_risk="low",
    ),
    _two_choice_property_spec(
        name="Pgp",
        context="P-glycoprotein efflux can limit permeability and bioavailability.",
        question=(
            "Given a drug SMILES string, predict whether it has P-glycoprotein efflux liability "
            "(A) low (B) high"
        ),
        option_a_value="low P-glycoprotein efflux liability",
        option_b_value="high P-glycoprotein efflux liability",
        option_a_risk="low",
        option_b_risk="high",
    ),
    _three_choice_property_spec(
        name="Lipophilicity",
        context="Lipophilicity is commonly measured as log-ratio and influences permeability and developability.",
        question=(
            "Given a drug SMILES string, predict whether its lipophilicity is "
            "(A) low (B) moderate (C) high"
        ),
        option_a_value="low lipophilicity",
        option_b_value="moderate lipophilicity",
        option_c_value="high lipophilicity",
        option_a_risk="low",
        option_b_risk="medium",
        option_c_risk="high",
        few_shot_examples=(
            PredictionPromptExample("CO", "(A)"),
            PredictionPromptExample("CCCCCCCC", "(C)"),
        ),
        constrain_with_grammar=True,
        unit="log-ratio",
        min_value=-5.0,
        max_value=10.0,
    ),
    _three_choice_property_spec(
        name="Solubility",
        context="Aqueous solubility is an important determinant of developability and exposure.",
        question=(
            "Given a drug SMILES string, predict whether its aqueous solubility is "
            "(A) low (B) moderate (C) high"
        ),
        option_a_value="low aqueous solubility",
        option_b_value="moderate aqueous solubility",
        option_c_value="high aqueous solubility",
        option_a_risk="high",
        option_b_risk="medium",
        option_c_risk="low",
        few_shot_examples=(
            PredictionPromptExample("CO", "(C)"),
            PredictionPromptExample("CCCCCCCC", "(A)"),
        ),
        constrain_with_grammar=True,
        unit="log mol/L",
        min_value=-12.0,
        max_value=4.0,
        numeric_value_prefix="Solubility",
    ),
    _two_choice_property_spec(
        name="BBB",
        context=(
            "As a membrane separating circulating blood and brain extracellular fluid, the blood-brain "
            "barrier blocks many foreign drugs and shapes CNS exposure."
        ),
        question=(
            "Given a drug SMILES string, predict whether it "
            "(A) does not cross the BBB (B) crosses the BBB"
        ),
        option_a_value="does not cross the BBB",
        option_b_value="crosses the BBB",
        option_a_risk="low",
        option_b_risk="high",
    ),
    _three_choice_property_spec(
        name="PPBR",
        context="Plasma protein binding rate affects free drug exposure and distribution.",
        question=(
            "Given a drug SMILES string, predict whether its plasma protein binding is "
            "(A) low (B) moderate (C) high"
        ),
        option_a_value="low plasma protein binding",
        option_b_value="moderate plasma protein binding",
        option_c_value="high plasma protein binding",
        option_a_risk="low",
        option_b_risk="medium",
        option_c_risk="high",
        few_shot_examples=(
            PredictionPromptExample("CO", "(A)"),
            PredictionPromptExample("CCCCCCCC", "(C)"),
        ),
        constrain_with_grammar=True,
        unit="%",
        min_value=0.0,
        max_value=100.0,
        numeric_value_prefix="PPBR",
    ),
    _numeric_property_spec(
        name="VDss",
        context="Volume of distribution at steady state reflects tissue distribution.",
        question="Given a drug SMILES string, predict its steady-state volume of distribution as a numeric value.",
        unit="L/kg",
        min_value=0.0,
        max_value=100.0,
        numeric_value_prefix="VDss",
    ),
    _two_choice_property_spec(
        name="CYP2C9 inhibition",
        context="CYP2C9 inhibition can increase drug-drug interaction risk during development.",
        question=(
            "Given a drug SMILES string, predict whether it "
            "(A) has low CYP2C9 inhibition risk (B) has high CYP2C9 inhibition risk"
        ),
        option_a_value="low CYP2C9 inhibition risk",
        option_b_value="high CYP2C9 inhibition risk",
        option_a_risk="low",
        option_b_risk="high",
    ),
    _two_choice_property_spec(
        name="CYP2D6 inhibition",
        context="CYP2D6 inhibition can create meaningful clinical DDI risk.",
        question=(
            "Given a drug SMILES string, predict whether it "
            "(A) has low CYP2D6 inhibition risk (B) has high CYP2D6 inhibition risk"
        ),
        option_a_value="low CYP2D6 inhibition risk",
        option_b_value="high CYP2D6 inhibition risk",
        option_a_risk="low",
        option_b_risk="high",
    ),
    _two_choice_property_spec(
        name="CYP3A4 inhibition",
        context="CYP3A4 inhibition can increase interaction risk across many concomitant medications.",
        question=(
            "Given a drug SMILES string, predict whether it "
            "(A) has low CYP3A4 inhibition risk (B) has high CYP3A4 inhibition risk"
        ),
        option_a_value="low CYP3A4 inhibition risk",
        option_b_value="high CYP3A4 inhibition risk",
        option_a_risk="low",
        option_b_risk="high",
    ),
    _two_choice_property_spec(
        name="CYP2C9 substrate",
        context="Being a CYP2C9 substrate can create metabolism-driven exposure variability.",
        question=(
            "Given a drug SMILES string, predict whether it "
            "(A) is not a CYP2C9 substrate (B) is a CYP2C9 substrate"
        ),
        option_a_value="not a CYP2C9 substrate",
        option_b_value="is a CYP2C9 substrate",
        option_a_risk="low",
        option_b_risk="high",
    ),
    _two_choice_property_spec(
        name="CYP2D6 substrate",
        context="Being a CYP2D6 substrate can amplify genotype-sensitive exposure changes.",
        question=(
            "Given a drug SMILES string, predict whether it "
            "(A) is not a CYP2D6 substrate (B) is a CYP2D6 substrate"
        ),
        option_a_value="not a CYP2D6 substrate",
        option_b_value="is a CYP2D6 substrate",
        option_a_risk="low",
        option_b_risk="high",
    ),
    _two_choice_property_spec(
        name="CYP3A4 substrate",
        context="Being a CYP3A4 substrate can create broad interaction and exposure risks.",
        question=(
            "Given a drug SMILES string, predict whether it "
            "(A) is not a CYP3A4 substrate (B) is a CYP3A4 substrate"
        ),
        option_a_value="not a CYP3A4 substrate",
        option_b_value="is a CYP3A4 substrate",
        option_a_risk="low",
        option_b_risk="high",
    ),
    _numeric_property_spec(
        name="Half-life",
        context="Half-life affects dosing frequency and systemic exposure persistence.",
        question="Given a drug SMILES string, predict its half-life as a numeric value.",
        unit="hr",
        min_value=0.0,
        max_value=240.0,
        numeric_value_prefix="Half-life",
    ),
    _numeric_property_spec(
        name="Clearance hepatocyte",
        context="Hepatocyte clearance reflects intrinsic cellular metabolic clearance.",
        question="Given a drug SMILES string, predict its hepatocyte clearance as a numeric value.",
        unit="uL.min-1.(10^6 cells)-1",
        min_value=0.0,
        max_value=5000.0,
        numeric_value_prefix="Clearance hepatocyte",
    ),
    _numeric_property_spec(
        name="Clearance microsome",
        context="Microsomal clearance reflects liver microsome-based metabolic clearance.",
        question="Given a drug SMILES string, predict its microsomal clearance as a numeric value.",
        unit="mL.min-1.g-1",
        min_value=0.0,
        max_value=5000.0,
        numeric_value_prefix="Clearance microsome",
    ),
    _three_choice_property_spec(
        name="LD50",
        context="LD50 is a toxicity benchmark where larger inverse-dose values indicate greater acute toxicity risk.",
        question=(
            "Given a drug SMILES string, predict whether its acute toxicity risk is "
            "(A) low (B) moderate (C) high"
        ),
        option_a_value="low acute toxicity risk",
        option_b_value="moderate acute toxicity risk",
        option_c_value="high acute toxicity risk",
        option_a_risk="low",
        option_b_risk="medium",
        option_c_risk="high",
        few_shot_examples=(
            PredictionPromptExample("CO", "(A)"),
            PredictionPromptExample("C1=CC=CC=C1Cl", "(B)"),
        ),
        constrain_with_grammar=True,
        unit="log(1/(mol/kg))",
        min_value=0.0,
        max_value=10.0,
        numeric_value_prefix="LD50",
    ),
    _two_choice_property_spec(
        name="hERG",
        context="hERG channel inhibition is associated with QT prolongation and cardiotoxicity risk.",
        question=(
            "Given a drug SMILES string, predict whether it "
            "(A) has low hERG inhibition risk (B) has high hERG inhibition risk"
        ),
        option_a_value="low hERG inhibition risk",
        option_b_value="high hERG inhibition risk",
        option_a_risk="low",
        option_b_risk="high",
    ),
    _two_choice_property_spec(
        name="AMES",
        context="AMES mutagenicity is a standard genotoxicity screening task.",
        question=(
            "Given a drug SMILES string, predict whether it "
            "(A) has low AMES mutagenicity risk (B) has high AMES mutagenicity risk"
        ),
        option_a_value="low AMES mutagenicity risk",
        option_b_value="high AMES mutagenicity risk",
        option_a_risk="low",
        option_b_risk="high",
    ),
    _two_choice_property_spec(
        name="DILI",
        context="Drug-induced liver injury is a major safety concern in therapeutic development.",
        question=(
            "Given a drug SMILES string, predict whether it "
            "(A) has low DILI risk (B) has high DILI risk"
        ),
        option_a_value="low DILI risk",
        option_b_value="high DILI risk",
        option_a_risk="low",
        option_b_risk="high",
    ),
)


def get_core_prediction_specs(
    names: tuple[str, ...] | None = None,
) -> tuple[PredictionPropertySpec, ...]:
    if not names:
        return CORE_PREDICTION_PROPERTY_SPECS

    selected: list[PredictionPropertySpec] = []
    remaining = set(names)
    for spec in CORE_PREDICTION_PROPERTY_SPECS:
        if spec.name in remaining:
            selected.append(spec)
            remaining.remove(spec.name)
    if remaining:
        raise KeyError(f"unknown prediction properties: {sorted(remaining)}")
    return tuple(selected)
