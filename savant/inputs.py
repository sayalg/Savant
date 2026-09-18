"""Reading targets and ligands from the command line or from files.

Every reader returns the same two shapes -- :class:`Target` and :class:`Ligand`
-- so the CLI and the JSON report never have to care whether a sequence arrived
as a literal string, a FASTA record or a PDB chain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

## The 20 standard residues, plus the ambiguity and gap codes that appear in real
## FASTA files. Anything outside this raises, because a sequence that is actually
## a mistyped filename would otherwise be embedded and scored without complaint.
AA_STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
AA_ALLOWED = AA_STANDARD | set("BJOUXZ")

AA3 = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
       "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
       "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
       "TRP": "W", "TYR": "Y"}

## Header names a SMILES-bearing column may go by, lowercased.
SMILES_HEADERS = ("smiles", "smi", "canonical_smiles", "ligand_smiles", "structure")
NAME_HEADERS = ("name", "id", "title", "ligand", "ligand_id", "compound",
                "compound_id", "molecule_chembl_id", "chembl_id")


@dataclass
class Target:
    """A protein to score against."""
    sequence: str
    id: str | None = None
    source: str = "argument"

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class Ligand:
    """A ligand to score. `name` is whatever the input labelled it, if anything."""
    smiles: str
    name: str | None = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# sequences
# --------------------------------------------------------------------------- #

def clean_sequence(raw: str, where: str = "sequence") -> str:
    """Uppercase, strip whitespace and digits, then validate.

    Digits and whitespace are stripped because sequences are so often pasted
    from a numbered alignment or a PDB SEQRES block.
    """
    seq = re.sub(r"[\s\d*]+", "", raw).upper()
    if not seq:
        raise ValueError(f"{where}: empty after cleaning")
    bad = sorted(set(seq) - AA_ALLOWED)
    if bad:
        raise ValueError(
            f"{where}: {''.join(bad)} is not an amino-acid code. "
            f"Expected one-letter residues; got {seq[:40]!r}"
            + ("..." if len(seq) > 40 else "")
        )
    return seq


def parse_fasta(text: str):
    """[(id, sequence)] from FASTA text, or one unnamed record if there is no '>'."""
    text = text.strip()
    if not text:
        return []
    if not text.startswith(">"):
        return [(None, text)]

    records, name, chunks = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if chunks:
                records.append((name, "".join(chunks)))
            name, chunks = line[1:].strip().split()[0] if line[1:].strip() else None, []
        elif line:
            chunks.append(line)
    if chunks:
        records.append((name, "".join(chunks)))
    return records


def read_targets(value: str, protein_chain: str = "A") -> list[Target]:
    """Targets from a literal sequence, a FASTA/text file, or a PDB.

    A value that names an existing file is read as a file; anything else is
    treated as a literal sequence. That is unambiguous in practice -- an
    amino-acid string is not a path that exists -- and a typo in a path
    surfaces as "not an amino-acid code" rather than as silent nonsense.
    """
    path = Path(value)
    if not path.exists():
        return [Target(clean_sequence(value, "--seq"), source="argument")]

    if path.suffix.lower() in (".pdb", ".ent"):
        sequence, _ = read_pdb(path, protein_chain=protein_chain)
        return [Target(clean_sequence(sequence, str(path)), id=path.stem,
                       source=f"{path.name} chain {protein_chain}")]

    records = parse_fasta(path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"{path}: no sequence found")
    return [Target(clean_sequence(seq, f"{path}:{name or i + 1}"),
                   id=name, source=path.name)
            for i, (name, seq) in enumerate(records)]


# --------------------------------------------------------------------------- #
# ligands
# --------------------------------------------------------------------------- #

def _looks_like_header(fields: list[str]) -> bool:
    return bool(fields) and fields[0].strip().lower() in SMILES_HEADERS


def parse_smiles_lines(text: str, where: str = "input") -> list[Ligand]:
    """Ligands from `.smi`-style text: one SMILES per line, optional name after it.

    Blank lines and `#` comments are skipped, and a leading header row naming a
    SMILES column is ignored -- so a file exported from a spreadsheet works
    without editing.
    """
    ligands, seen_header = [], False
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if not seen_header and not ligands and _looks_like_header(fields):
            seen_header = True
            continue
        ligands.append(Ligand(smiles=fields[0],
                              name=" ".join(fields[1:]) or None))
    if not ligands:
        raise ValueError(f"{where}: no SMILES found")
    return ligands


def _read_table(path: Path) -> list[Ligand]:
    """Ligands from a CSV/TSV, picking the SMILES column by name."""
    import pandas as pd

    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep)
    lower = {c.lower(): c for c in df.columns}

    smiles_col = next((lower[h] for h in SMILES_HEADERS if h in lower), None)
    if smiles_col is None:
        raise ValueError(
            f"{path}: no SMILES column. Looked for {list(SMILES_HEADERS)}, "
            f"found {list(df.columns)[:8]}"
        )
    name_col = next((lower[h] for h in NAME_HEADERS if h in lower), None)

    ligands = []
    for _, row in df.iterrows():
        smiles = row[smiles_col]
        if pd.isna(smiles) or not str(smiles).strip():
            continue
        name = None
        if name_col is not None and not pd.isna(row[name_col]):
            name = str(row[name_col])
        ligands.append(Ligand(smiles=str(smiles).strip(), name=name))
    if not ligands:
        raise ValueError(f"{path}: SMILES column {smiles_col!r} is empty")
    return ligands


def read_ligands(values) -> list[Ligand]:
    """Ligands from SMILES given directly, from files, or from any mix of both.

    Each value that names an existing file is read as one (`.csv`/`.tsv` by
    column, anything else line by line); every other value is taken as a literal
    SMILES. So all of these work, including together:

        --smiles "CCO"
        --smiles "CCO" "c1ccccc1"
        --smiles library.smi
        --smiles library.csv extra_hit.smi "CCO"
    """
    if isinstance(values, (str, Path)):
        values = [values]

    ligands = []
    for value in values:
        path = Path(value)
        if not path.exists():
            ligands.append(Ligand(smiles=str(value).strip()))
        elif path.suffix.lower() in (".csv", ".tsv", ".tab"):
            ligands.extend(_read_table(path))
        else:
            ligands.extend(parse_smiles_lines(path.read_text(encoding="utf-8"),
                                              where=str(path)))
    if not ligands:
        raise ValueError("no ligands given")
    return ligands


def dedupe_ligands(ligands: list[Ligand]) -> tuple[list[Ligand], int]:
    """(unique ligands, number dropped), keeping first occurrence and its name.

    Scoring the same SMILES twice costs the same as scoring two ligands and adds
    nothing, and a library file assembled from several sources usually overlaps.
    """
    seen, unique = set(), []
    for lig in ligands:
        if lig.smiles in seen:
            continue
        seen.add(lig.smiles)
        unique.append(lig)
    return unique, len(ligands) - len(unique)


# --------------------------------------------------------------------------- #
# structures
# --------------------------------------------------------------------------- #

def read_pdb(path, protein_chain="A", ligand_chain="B"):
    """(sequence, smiles) from a structure -- a convenience, not a feature source.

    The sequence is exact. The SMILES is inferred from ligand geometry, so bond
    orders are a guess (`C(=O)O` comes back as `C(O)O`) and every ligand feature
    is computed from the SMILES. Pass the SMILES explicitly whenever you have it.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    seen, seq, het = set(), [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("ATOM  ") and line[21] == protein_chain:
            key = line[22:27]
            if key not in seen:
                seen.add(key)
                seq.append(AA3.get(line[17:20].strip(), "X"))
        elif line.startswith("HETATM") and line[21] == ligand_chain:
            het.append(line)

    if not seq:
        raise ValueError(f"no ATOM records on chain {protein_chain!r} in {path}")
    mol = Chem.MolFromPDBBlock("\n".join(het) + "\nEND", sanitize=True) if het else None
    return "".join(seq), (Chem.MolToSmiles(mol) if mol is not None else None)
