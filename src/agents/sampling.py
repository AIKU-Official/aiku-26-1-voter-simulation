"""Step 1 — Nemotron-Personas-Korea sampling + narrative assembly."""
from __future__ import annotations

import glob

import pandas as pd

from . import demographics as dm

# Persona prose fields combined into the narrative shown to the judge / model.
NARRATIVE_FIELDS = [
    "persona",
    "professional_persona",
    "family_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
]
LOAD_COLS = [
    "uuid", "sex", "age", "education_level", "province", "district",
    "marital_status", "military_status", "family_type", "housing_type",
    "occupation", *NARRATIVE_FIELDS,
]


def _assemble_narrative(row: pd.Series) -> str:
    parts = [str(row[f]).strip() for f in NARRATIVE_FIELDS if pd.notna(row.get(f))]
    return "\n".join(p for p in parts if p)


def sample_personas(glob_pattern: str, n: int, seed: int = 42) -> pd.DataFrame:
    """Person-level uniform random sample of `n` personas (proposal §8.1 default).

    Returns a frame with agent_id, recoded demographics, the assembled narrative,
    and the raw persona fields.
    """
    shards = sorted(glob.glob(glob_pattern))
    if not shards:
        raise FileNotFoundError(f"No Nemotron parquet shards at {glob_pattern}")
    df = pd.concat((pd.read_parquet(s, columns=LOAD_COLS) for s in shards), ignore_index=True)

    samp = df.sample(n=n, random_state=seed).reset_index(drop=True)
    samp.insert(0, "agent_id", range(len(samp)))
    samp = dm.recode_nemotron(samp)
    samp["narrative"] = samp.apply(_assemble_narrative, axis=1)
    return samp, len(df)
