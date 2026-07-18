from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "python" / "scfi_analysis.py"
SPEC = importlib.util.spec_from_file_location("scfi_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
scfi_analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scfi_analysis)


def test_load_data_sorts_rows_and_removes_invalid_values(tmp_path: Path) -> None:
    source = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "Date": ["2024-03-01", "invalid", "2024-01-01"],
            "SCFI": ["2100", "1900", "1800"],
        }
    ).to_csv(source, index=False)

    result = scfi_analysis.load_data(source)

    assert result["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-03-01",
    ]
    assert result["SCFI"].tolist() == [1800, 2100]


def test_load_data_rejects_missing_required_columns(tmp_path: Path) -> None:
    source = tmp_path / "sample.csv"
    pd.DataFrame({"Date": ["2024-01-01"], "Rate": [1800]}).to_csv(
        source, index=False
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        scfi_analysis.load_data(source)


def test_add_metrics_calculates_returns_averages_and_drawdown() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=13, freq="MS"),
            "SCFI": [1000, 1100, 1200, 900, 1000, 1050, 1150, 1250, 1300, 1280, 1350, 1400, 1500],
        }
    )

    result = scfi_analysis.add_metrics(frame)

    assert result.loc[1, "MoM"] == pytest.approx(0.10)
    assert result.loc[12, "YoY"] == pytest.approx(0.50)
    assert result.loc[3, "Drawdown"] == pytest.approx(-0.25)
    assert result.loc[2, "MA_3M"] == pytest.approx(1100)
    assert str(result.loc[0, "Market_Phase"]) == "Normal"


def test_annual_summary_reports_yearly_observations() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-12-01", "2024-01-01", "2024-02-01"]),
            "SCFI": [1000, 1200, 1800],
        }
    )
    metrics = scfi_analysis.add_metrics(frame)

    result = scfi_analysis.annual_summary(metrics)

    assert result["Year"].tolist() == [2023, 2024]
    assert result["Observations"].tolist() == [1, 2]
    assert result.loc[result["Year"] == 2024, "Average_SCFI"].item() == pytest.approx(1500)
