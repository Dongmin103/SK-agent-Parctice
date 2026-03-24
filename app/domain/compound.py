from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from app.domain.models import CompoundContext


class InvalidSmilesError(ValueError):
    """Raised when a SMILES string cannot be parsed into a molecule."""


class CompoundPreprocessor:
    def __init__(
        self,
        *,
        svg_width: int = 320,
        svg_height: int = 240,
    ) -> None:
        self.svg_width = svg_width
        self.svg_height = svg_height

    def build_context(
        self,
        *,
        smiles: str,
        target: str,
        compound_name: str | None = None,
    ) -> CompoundContext:
        normalized_smiles = smiles.strip()
        molecule = self._parse_smiles(normalized_smiles)
        canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
        molecule_svg = self._render_molecule_svg(molecule)
        return CompoundContext(
            smiles=normalized_smiles,
            target=target,
            compound_name=compound_name,
            canonical_smiles=canonical_smiles,
            molecule_svg=molecule_svg,
        )

    def _parse_smiles(self, smiles: str) -> Chem.Mol:
        if not smiles:
            raise InvalidSmilesError("Invalid SMILES: <empty>")

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise InvalidSmilesError(f"Invalid SMILES: {smiles}")
        return molecule

    def _render_molecule_svg(self, molecule: Chem.Mol) -> str:
        drawable = Chem.Mol(molecule)
        rdDepictor.Compute2DCoords(drawable)
        drawer = rdMolDraw2D.MolDraw2DSVG(self.svg_width, self.svg_height)
        drawer.DrawMolecule(drawable)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
