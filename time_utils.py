# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import torch


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATE_TXT = PACKAGE_ROOT / "data" / "ndvi_dates.txt"


def load_date_strings(date_txt_path: Optional[str] = None) -> List[str]:
    """
Read the timestamp from the TXT file. Support
- One date per line (YYYYMMDD)
Blank lines are automatically ignored
Comment lines starting with # are automatically ignored

    """
    txt_path = Path(date_txt_path) if date_txt_path is not None else DEFAULT_DATE_TXT
    if not txt_path.exists():
        raise FileNotFoundError(f"The date file was not found: {txt_path}")

    date_strs: List[str] = []
 with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if len(s) != 8 or not s.isdigit():
                raise ValueError(f"Illegal date format {s}，requesting for YYYYMMDD")
            date_strs.append(s)

    if not date_strs:
 raise ValueError(f"The date file is empty {txt_path}")
    return date_strs


def month_to_season(month: int) -> int:
    if month in (3, 4, 5):
        return 0
    if month in (6, 7, 8):
        return 1
    if month in (9, 10, 11):
        return 2
    return 3


def build_time_metadata(date_strs: List[str]) -> Dict[str, torch.Tensor]:
    dates = [dt.datetime.strptime(s, "%Y%m%d").date() for s in date_strs]

    years = torch.tensor([d.year for d in dates], dtype=torch.long)
    months = torch.tensor([d.month for d in dates], dtype=torch.long)
    doy = torch.tensor([d.timetuple().tm_yday for d in dates], dtype=torch.float32)
    delta_days = torch.tensor([1] + [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))],
        dtype=torch.float32,
    )
    tau_days = torch.cumsum(delta_days, dim=0)

    uniq_years = sorted(set(int(y) for y in years.tolist()))
    year_to_id = {y: i for i, y in enumerate(uniq_years)}
    year_ids = torch.tensor([year_to_id[int(y)] for y in years.tolist()], dtype=torch.long)

    return {
        "years": years,
        "months": months,
        "doy": doy,
        "delta_days": delta_days,
        "tau_days": tau_days,
        "year_ids": year_ids,
        "num_years": torch.tensor(len(uniq_years), dtype=torch.long),
    }


def build_phase_metadata(date_strs: List[str]) -> Dict[str, torch.Tensor]:
    meta = build_time_metadata(date_strs)
    year_ids = meta["year_ids"]
    months = meta["months"]

    seasons = torch.tensor([month_to_season(int(m)) for m in months.tolist()], dtype=torch.long)
    num_years = int(meta["num_years"].item())
    num_seasons = 4
    num_phase_tokens = num_years * num_seasons

    bucket_masks = []
    for y in range(num_years):
    for s in range(num_seasons):
    mask = ((year_ids == y) & (seasons == s)).float()
    bucket_masks.append(mask)

    meta.update({
        "season_ids": seasons,
        "num_seasons": torch.tensor(num_seasons, dtype=torch.long),
        "num_phase_tokens": torch.tensor(num_phase_tokens, dtype=torch.long),
        "bucket_masks": torch.stack(bucket_masks, dim=0),
    })
    return meta