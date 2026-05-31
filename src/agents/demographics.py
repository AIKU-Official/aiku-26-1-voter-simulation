"""Shared demographic recoding for Nemotron agents and KGSS respondents.

Both sources are mapped onto a common scheme so the KGSS demographic-conditional
orientation lookup (Step 2a) keys identically on agents and respondents:
  - sex_label        : 남성 / 여성
  - age_bucket       : 6 buckets (configs/config.yaml election.age_buckets)
  - region7          : KGSS REGION's 7 regions (orientation conditional key)
  - edu4             : 중졸이하 / 고졸 / 대졸 / 대학원
  - orientation_5way : VP/P/M/C/VC from KGSS PARTYLR
The 17-sido label (sido17) is kept from Nemotron `province` for the within-sido
swap (Step 4); KGSS has no sido, only the 7-way REGION.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# KGSS REGION value labels (1..7) → name
REGION7_FROM_KGSS = {1: "서울", 2: "경기", 3: "강원", 4: "충청", 5: "경상", 6: "전라", 7: "제주"}

# Nemotron `province` (17 sido) → KGSS 7-way region
PROVINCE_TO_REGION7 = {
    "서울": "서울",
    "경기": "경기", "인천": "경기",
    "강원": "강원",
    "대전": "충청", "세종": "충청", "충청남": "충청", "충청북": "충청",
    "부산": "경상", "대구": "경상", "울산": "경상", "경상남": "경상", "경상북": "경상",
    "광주": "전라", "전라남": "전라", "전북": "전라",
    "제주": "제주",
}

# KGSS EDUC (0..8) → 4 levels
EDU4_FROM_KGSS = {
    0: "중졸이하", 1: "중졸이하", 2: "중졸이하",
    3: "고졸",
    4: "대졸", 5: "대졸",
    6: "대학원", 7: "대학원",
    8: None,  # 'Other (서당)' — drop
}

# Nemotron education_level → 4 levels
EDU4_FROM_NEMOTRON = {
    "무학": "중졸이하", "초등학교": "중졸이하", "중학교": "중졸이하",
    "고등학교": "고졸",
    "2~3년제 전문대학": "대졸", "4년제 대학교": "대졸",
    "대학원": "대학원",
}

SEX_FROM_NEMOTRON = {"남자": "남성", "여자": "여성"}
SEX_FROM_KGSS = {1: "남성", 2: "여성"}
ORIENT_FROM_PARTYLR = {1: "VP", 2: "P", 3: "M", 4: "C", 5: "VC"}

AGE_BUCKETS = ["19-29", "30-39", "40-49", "50-59", "60-69", "70+"]
_AGE_BINS = [18, 29, 39, 49, 59, 69, 200]

ORIENT_5WAY = ["VP", "P", "M", "C", "VC"]
# 5-way → Gallup 3-way
THREEWAY = {"VP": "진보", "P": "진보", "M": "중도", "C": "보수", "VC": "보수"}


def age_bucket(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=_AGE_BINS, labels=AGE_BUCKETS, right=True)


def recode_nemotron(df: pd.DataFrame) -> pd.DataFrame:
    """Add common demographic columns to a Nemotron persona frame."""
    out = df.copy()
    out["sex_label"] = out["sex"].map(SEX_FROM_NEMOTRON)
    out["age_bucket"] = age_bucket(out["age"])
    out["sido17"] = out["province"]
    out["region7"] = out["province"].map(PROVINCE_TO_REGION7)
    out["edu4"] = out["education_level"].map(EDU4_FROM_NEMOTRON)
    return out


def recode_kgss(df: pd.DataFrame) -> pd.DataFrame:
    """Add common demographic + orientation columns to a KGSS frame (all waves).
    Drops rows missing orientation or any conditioning demographic."""
    out = df.copy()
    out["sex_label"] = out["SEX"].map(SEX_FROM_KGSS)
    out["age_bucket"] = age_bucket(out["AGE"])
    out["region7"] = out["REGION"].map(REGION7_FROM_KGSS)
    out["edu4"] = out["EDUC"].map(EDU4_FROM_KGSS)
    out["orientation_5way"] = out["PARTYLR"].map(ORIENT_FROM_PARTYLR)
    return out.dropna(subset=["sex_label", "age_bucket", "region7", "edu4", "orientation_5way"])
