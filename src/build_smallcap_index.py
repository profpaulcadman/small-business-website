from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"


@dataclass
class Settings:
    source_type: str
    source_path: str
    date_column: str
    returns_column: Optional[str]
    price_column: Optional[str]
    date_format: Optional[str]
    ticker: Optional[str]
    interval: Optional[str]
    auto_adjust: bool
    ar_lags: int
    rolling_window: int
    ewma_lambda: float
    chart_title: str
    threshold: float


def load_settings() -> Settings:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return Settings(
        source_type=raw["source"]["type"],
        source_path=raw["source"]["path"],
        date_column=raw["source"]["date_column"],
        returns_column=raw["source"].get("returns_column"),
        price_column=raw["source"].get("price_column"),
        date_format=raw["source"].get("date_format"),
        ticker=raw["source"].get("ticker"),
        interval=raw["source"].get("interval"),
        auto_adjust=raw["source"].get("auto_adjust", False),
        ar_lags=raw["model"]["ar_lags"],
        rolling_window=raw["model"]["rolling_window"],
        ewma_lambda=raw["model"]["ewma_lambda"],
        chart_title=raw["site"]["chart_title"],
        threshold=raw["site"]["threshold"],
    )


def load_source(settings: Settings) -> pd.DataFrame:
    source_path = settings.source_path
    if settings.source_type == "yahoo":
        if not settings.ticker:
            raise ValueError("Set source.ticker in config.json for Yahoo downloads.")

        raw = yf.download(
            settings.ticker,
            period="max",
            interval=settings.interval or "1mo",
            auto_adjust=settings.auto_adjust,
            progress=False,
            actions=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise ValueError(f"No Yahoo Finance data returned for ticker {settings.ticker}.")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        price_column = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if price_column not in raw.columns:
            raise KeyError("Yahoo Finance did not return a Close or Adj Close series.")

        series = raw.reset_index()[["Date", price_column]].copy()
        series.columns = ["Date", "Price"]
        series["Returns"] = series["Price"].pct_change() * 100.0
        series = series[["Date", "Returns"]]
    elif settings.source_type == "csv_url":
        df = pd.read_csv(source_path)
    elif settings.source_type == "local_csv":
        df = pd.read_csv(ROOT / source_path)
    else:
        raise ValueError(f"Unsupported source type: {settings.source_type}")

    if settings.source_type in {"csv_url", "local_csv"}:
        if settings.date_column not in df.columns:
            raise KeyError(f"Date column '{settings.date_column}' was not found.")

        if settings.returns_column and settings.returns_column in df.columns:
            series = df[[settings.date_column, settings.returns_column]].copy()
            series.columns = ["Date", "Returns"]
        elif settings.price_column and settings.price_column in df.columns:
            price_df = df[[settings.date_column, settings.price_column]].copy()
            price_df.columns = ["Date", "Price"]
            price_df["Returns"] = price_df["Price"].pct_change() * 100.0
            series = price_df[["Date", "Returns"]]
        else:
            raise KeyError(
                "Provide either a returns column or a price column in config.json."
            )

    series["Date"] = pd.to_datetime(
        series["Date"],
        format=settings.date_format,
        errors="coerce",
        dayfirst=True,
    )
    series["Returns"] = pd.to_numeric(series["Returns"], errors="coerce")
    series = series.dropna(subset=["Date", "Returns"]).sort_values("Date")

    if len(series) <= settings.ar_lags:
        raise ValueError("Not enough observations for the selected AR lag count.")

    return series.reset_index(drop=True)


def fit_ar_residuals(returns: np.ndarray, p: int) -> np.ndarray:
    y = returns[p:]
    x = np.ones((len(returns) - p, p + 1))
    for lag in range(1, p + 1):
        x[:, lag] = returns[p - lag : len(returns) - lag]

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(values, fill_value=np.nan, dtype=float)
    for idx in range(window - 1, len(values)):
        out[idx] = np.std(values[idx - window + 1 : idx + 1], ddof=0)
    return out


def ewma_std(values: np.ndarray, lam: float) -> np.ndarray:
    out = np.full_like(values, fill_value=np.nan, dtype=float)
    variance = values[0] ** 2
    out[0] = np.sqrt(variance)
    for idx in range(1, len(values)):
        variance = lam * variance + (1.0 - lam) * (values[idx - 1] ** 2)
        out[idx] = np.sqrt(variance)
    return out


def zscore(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    out = np.full_like(values, fill_value=np.nan, dtype=float)
    if valid.sum() < 2:
        return out
    subset = values[valid]
    std = subset.std(ddof=0)
    if std == 0:
        out[valid] = 0.0
    else:
        out[valid] = (subset - subset.mean()) / std
    return out


def build_outputs(series: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    residuals = fit_ar_residuals(series["Returns"].to_numpy(dtype=float), settings.ar_lags)
    dates = series["Date"].iloc[settings.ar_lags :].reset_index(drop=True)

    u_roll = rolling_std(residuals, settings.rolling_window)
    u_ewma = ewma_std(residuals, settings.ewma_lambda)
    u_roll_z = zscore(u_roll)
    u_ewma_z = zscore(u_ewma)

    return pd.DataFrame(
        {
            "Date": dates,
            "ForecastError": residuals,
            "U_Rolling": u_roll,
            "U_EWMA": u_ewma,
            "U_Rolling_Z": u_roll_z,
            "U_EWMA_Z": u_ewma_z,
        }
    )


def write_json(series: pd.DataFrame, source_meta: pd.DataFrame, settings: Settings) -> None:
    output_path = ROOT / "site" / "data" / "index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "title": settings.chart_title,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "source_rows": int(len(source_meta)),
        "series": [
            {
                "date": row.Date.strftime("%Y-%m-%d"),
                "forecast_error": None if pd.isna(row.ForecastError) else float(row.ForecastError),
                "u_rolling": None if pd.isna(row.U_Rolling) else float(row.U_Rolling),
                "u_ewma": None if pd.isna(row.U_EWMA) else float(row.U_EWMA),
                "u_rolling_z": None if pd.isna(row.U_Rolling_Z) else float(row.U_Rolling_Z),
                "u_ewma_z": None if pd.isna(row.U_EWMA_Z) else float(row.U_EWMA_Z),
            }
            for row in series.itertuples(index=False)
        ],
        "settings": {
            "ar_lags": settings.ar_lags,
            "rolling_window": settings.rolling_window,
            "ewma_lambda": settings.ewma_lambda,
            "threshold": settings.threshold,
        },
        "latest": None,
    }

    if payload["series"]:
        latest = payload["series"][-1]
        payload["latest"] = {
            "date": latest["date"],
            "u_rolling_z": latest["u_rolling_z"],
            "u_ewma_z": latest["u_ewma_z"],
        }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_plot(series: pd.DataFrame, settings: Settings) -> None:
    output_path = ROOT / "site" / "data" / "plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(series["Date"], series["U_Rolling_Z"], linewidth=2, label="Rolling window")
    ax.plot(series["Date"], series["U_EWMA_Z"], linewidth=2, label="EWMA")
    ax.axhline(settings.threshold, color="black", linestyle=":", linewidth=1.2, label=f"{settings.threshold:.2f} threshold")
    ax.set_title(settings.chart_title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Standardized uncertainty")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    settings = load_settings()
    source_df = load_source(settings)
    output_df = build_outputs(source_df, settings)
    write_json(output_df, source_df, settings)
    write_plot(output_df, settings)
    print(f"Built {len(output_df)} monthly observations.")
    print(f"Latest source date: {source_df['Date'].iloc[-1].date()}")


if __name__ == "__main__":
    main()
