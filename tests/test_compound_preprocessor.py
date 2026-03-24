from __future__ import annotations

import pytest

from app.domain.compound import CompoundPreprocessor, InvalidSmilesError


def test_compound_preprocessor_builds_context_with_canonical_smiles_and_svg() -> None:
    preprocessor = CompoundPreprocessor()

    context = preprocessor.build_context(
        smiles="N#CC1=CC=CC=C1",
        target="KRAS G12C",
        compound_name="ABC-101",
    )

    assert context.smiles == "N#CC1=CC=CC=C1"
    assert context.target == "KRAS G12C"
    assert context.compound_name == "ABC-101"
    assert context.canonical_smiles == "N#Cc1ccccc1"
    assert context.molecule_svg is not None
    assert "<svg" in context.molecule_svg


def test_compound_preprocessor_rejects_invalid_smiles() -> None:
    preprocessor = CompoundPreprocessor()

    with pytest.raises(InvalidSmilesError, match="Invalid SMILES"):
        preprocessor.build_context(
            smiles="not-a-smiles",
            target="KRAS G12C",
            compound_name="ABC-101",
        )
