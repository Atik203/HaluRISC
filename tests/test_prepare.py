"""Grouped split tests: no source question (item_idx) may span multiple partitions."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.prepare import build_integrity_report, group_split_by_item  # noqa: E402


def make_df(n_groups: int = 200, seed: int = 7, missing_halucinated_p: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_groups):
        rows.append({"item_idx": i, "label": 0, "x": 1.0})
        if rng.random() > missing_halucinated_p:
            rows.append({"item_idx": i, "label": 1, "x": 2.0})
    return pd.DataFrame(rows)


def test_no_group_overlap():
    df, report = group_split_by_item(make_df())
    assert report["groups_spanning_multiple_splits"] == 0
    assert report["leakage_free"] is True
    assert df.groupby("item_idx")["split"].nunique().max() == 1


def test_splits_exist_and_reproducible():
    df1, r1 = group_split_by_item(make_df())
    df2, r2 = group_split_by_item(make_df())
    assert (df1["split"].values == df2["split"].values).all()
    counts = df1.groupby("split").size()
    assert set(counts.index) == {"train", "val", "test"}
    assert len(df1) == len(df1["split"]) and df1["split"].notna().all()


def test_label_balance_per_split():
    df, report = group_split_by_item(make_df(n_groups=400, seed=11))
    mean = df.groupby("split")["label"].mean()
    assert mean.sub(0.5).abs().max() < 0.15


def test_integrity_report_detects_leakage():
    df = pd.DataFrame(
        {
            "item_idx": [0, 0, 1, 1, 2, 2],
            "label": [0, 1, 0, 1, 0, 1],
            "split": ["train", "train", "val", "val", "test", "test"],
        }
    )
    report = build_integrity_report(df)
    assert report["leakage_free"] is True
    assert report["groups_spanning_multiple_splits"] == 0

    df.loc[0, "split"] = "test"  # group 0 now spans train + test
    with pytest.raises(AssertionError):
        build_integrity_report(df)
