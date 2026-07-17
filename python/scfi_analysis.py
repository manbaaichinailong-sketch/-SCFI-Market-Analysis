"""
SCFI Container Shipping Market Analysis (2020-2026)

Rebuilds analytical metrics and charts from the monthly reference CSV.
The dataset is for learning/portfolio use and is not the licensed complete
weekly raw series published by Shanghai Shipping Exchange.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "SCFI_Monthly_Reference_2020_2026.csv"
IMAGE_DIR = ROOT / "images"
OUTPUT_DIR = ROOT / "data"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run the script from the project structure described in README.md."
        )

    df = pd.read_csv(path)
    required = {"Date", "SCFI"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["SCFI"] = pd.to_numeric(df["SCFI"], errors="coerce")
    df = df.dropna(subset=["Date", "SCFI"]).sort_values("Date").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid SCFI observations were found.")

    return df


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MoM"] = out["SCFI"].pct_change()
    out["YoY"] = out["SCFI"].pct_change(12)
    out["MA_3M"] = out["SCFI"].rolling(3, min_periods=1).mean()
    out["MA_12M"] = out["SCFI"].rolling(12, min_periods=1).mean()
    out["Rolling_12M_Ann_Vol"] = out["MoM"].rolling(12, min_periods=2).std(ddof=0) * (12 ** 0.5)
    out["Running_Peak"] = out["SCFI"].cummax()
    out["Drawdown"] = out["SCFI"] / out["Running_Peak"] - 1
    out["Year"] = out["Date"].dt.year
    out["Market_Phase"] = pd.cut(
        out["SCFI"],
        bins=[-float("inf"), 1000, 1800, 3000, 4500, float("inf")],
        labels=["Low", "Normal", "Elevated", "High", "Extreme"],
        right=False,
    )
    return out


def annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Year")
        .agg(
            Average_SCFI=("SCFI", "mean"),
            Minimum=("SCFI", "min"),
            Maximum=("SCFI", "max"),
            Monthly_Volatility=("MoM", "std"),
            Worst_Drawdown=("Drawdown", "min"),
            Observations=("SCFI", "size"),
        )
        .reset_index()
    )
    summary["Annual_Average_YoY"] = summary["Average_SCFI"].pct_change()
    return summary


def save_charts(df: pd.DataFrame, annual: pd.DataFrame) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13, 6.5))
    plt.plot(df["Date"], df["SCFI"], linewidth=2, label="SCFI")
    plt.plot(df["Date"], df["MA_12M"], linestyle="--", linewidth=1.8, label="12M moving average")
    plt.title("Shanghai Containerized Freight Index (SCFI), 2020-2026")
    plt.xlabel("Date")
    plt.ylabel("Index points")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "python_scfi_trend.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    plt.bar(annual["Year"].astype(str), annual["Average_SCFI"])
    plt.title("SCFI Annual Average")
    plt.xlabel("Year")
    plt.ylabel("Average index points")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "python_scfi_annual_average.png", dpi=220)
    plt.close()

    plt.figure(figsize=(13, 5.5))
    plt.bar(df["Date"], df["MoM"] * 100, width=22)
    plt.axhline(0, linewidth=0.8)
    plt.title("SCFI Monthly Change")
    plt.xlabel("Date")
    plt.ylabel("MoM change (%)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "python_scfi_monthly_change.png", dpi=220)
    plt.close()

    plt.figure(figsize=(13, 5.5))
    plt.plot(df["Date"], df["Rolling_12M_Ann_Vol"] * 100, linewidth=2)
    plt.title("SCFI Rolling 12-Month Annualized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "python_scfi_rolling_volatility.png", dpi=220)
    plt.close()

    plt.figure(figsize=(13, 5.5))
    plt.fill_between(df["Date"], df["Drawdown"] * 100, 0, alpha=0.5)
    plt.title("SCFI Drawdown from Running Peak")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "python_scfi_drawdown.png", dpi=220)
    plt.close()


def print_summary(df: pd.DataFrame, annual: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    peak = df.loc[df["SCFI"].idxmax()]
    trough = df.loc[df["SCFI"].idxmin()]

    print("SCFI analysis completed")
    print("-" * 60)
    print(f"Observations: {len(df)}")
    print(f"Period: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Latest: {latest['SCFI']:,.2f}")
    print(f"Historical peak in dataset: {peak['SCFI']:,.2f} ({peak['Date'].date()})")
    print(f"Historical low in dataset: {trough['SCFI']:,.2f} ({trough['Date'].date()})")
    print(f"Latest drawdown from peak: {latest['Drawdown']:.1%}")
    print()
    print(annual.to_string(index=False))


def main() -> int:
    try:
        raw = load_data(DATA_FILE)
        analysis = add_metrics(raw)
        annual = annual_summary(analysis)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        analysis.to_csv(OUTPUT_DIR / "SCFI_Analysis_Output.csv", index=False, encoding="utf-8-sig")
        annual.to_csv(OUTPUT_DIR / "SCFI_Annual_Summary.csv", index=False, encoding="utf-8-sig")
        save_charts(analysis, annual)
        print_summary(analysis, annual)
        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
