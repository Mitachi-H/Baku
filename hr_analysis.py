#!/usr/bin/env python3
"""MMASHデータセットを用いた心拍・HRV解析スクリプト"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import interpolate, signal

DATASET_URL = "https://physionet.org/static/published-projects/mmash/mmash-1.0.0.zip"


@dataclass
class HRVMetrics:
    """HRVに関する主要指標"""

    mean_hr: float
    sdnn: float
    rmssd: float
    pnn50: float
    lf_power: float
    hf_power: float
    lf_hf_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMASHのRRデータからHR/HRVを解析")
    parser.add_argument("--subject", type=int, default=1, help="対象被験者番号 (1-22)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="解析結果を保存するディレクトリ",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="MMASH zipをキャッシュするディレクトリ (未指定時は ~/.cache/mmash)",
    )
    parser.add_argument(
        "--interp-freq",
        type=float,
        default=4.0,
        help="RRタコグラム補間時のサンプリング周波数 [Hz]",
    )
    return parser.parse_args()


def ensure_dataset(cache_dir: Path | None) -> Path:
    """MMASHデータセットをローカルに展開し、そのルートディレクトリを返す"""

    cache_root = cache_dir or Path.home() / ".cache" / "mmash"
    cache_root.mkdir(parents=True, exist_ok=True)

    data_root = cache_root / "mmash-1.0.0"
    if data_root.exists():
        return data_root

    zip_path = cache_root / "mmash-1.0.0.zip"
    if not zip_path.exists():
        print(f"Downloading MMASH dataset to {zip_path} ...")
        response = requests.get(DATASET_URL, stream=True, timeout=60)
        response.raise_for_status()
        with zip_path.open("wb") as fp:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fp.write(chunk)

    with zipfile.ZipFile(zip_path) as outer:
        inner_name = next(
            (name for name in outer.namelist() if name.endswith("MMASH.zip")),
            None,
        )
        if inner_name is None:
            raise RuntimeError("MMASH.zipが外側のアーカイブから見つかりません")
        outer.extract(inner_name, path=cache_root)

    inner_zip_path = cache_root / inner_name
    data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(inner_zip_path) as inner:
        inner.extractall(path=data_root)

    return data_root


def load_rr_series(data_root: Path, subject: int) -> pd.Series:
    """被験者ごとのRRデータをミリ秒単位のSeriesで返す"""

    candidates = [
        data_root / "DataPaper" / f"user_{subject:02d}" / "RR.csv",
        data_root / "DataPaper" / f"user_{subject}" / "RR.csv",
        data_root / f"Subject_{subject}" / "RR.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if "RR" in df.columns:
            series = pd.to_numeric(df["RR"], errors="coerce")
        elif "ibi_s" in df.columns:
            series = pd.to_numeric(df["ibi_s"], errors="coerce") * 1000.0
        else:
            continue
        series.name = "RR"
        return series

    raise FileNotFoundError(f"subject {subject} のRR.csvが見つかりません")


def compute_hrv(
    rr: pd.Series,
    interp_freq: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HRVMetrics]:
    """RR間隔からHRV指標を算出"""

    rr_clean = rr.dropna().to_numpy()
    if len(rr_clean) < 2:
        raise ValueError("RR間隔が不足しています")

    rr_sec = rr_clean / 1000.0
    t_rr = np.cumsum(rr_sec)
    t_rr -= t_rr[0]

    hr_bpm = 60.0 / rr_sec
    diff_rr = np.diff(rr_clean)

    sdnn = float(np.std(rr_clean, ddof=1))
    rmssd = float(np.sqrt(np.mean(diff_rr**2)))
    pnn50 = float(np.mean(np.abs(diff_rr) > 50.0) * 100.0)

    t_interp = np.arange(0, t_rr[-1], 1.0 / interp_freq)
    interp_rr = interpolate.interp1d(
        t_rr,
        rr_sec,
        kind="cubic",
        fill_value="extrapolate",
    )(t_interp)
    interp_rr = signal.detrend(interp_rr)

    freqs, psd = signal.welch(
        interp_rr,
        fs=interp_freq,
        nperseg=min(len(interp_rr), 1024),
    )

    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.40)

    lf_mask = (freqs >= lf_band[0]) & (freqs <= lf_band[1])
    hf_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])

    lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask]))
    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask]))
    lf_hf_ratio = float(lf_power / hf_power) if hf_power > 0 else float("nan")

    metrics = HRVMetrics(
        mean_hr=float(np.mean(hr_bpm)),
        sdnn=sdnn,
        rmssd=rmssd,
        pnn50=pnn50,
        lf_power=lf_power,
        hf_power=hf_power,
        lf_hf_ratio=lf_hf_ratio,
    )

    return hr_bpm, t_rr, freqs, psd, metrics


def save_figure(
    output_path: Path,
    t_rr: np.ndarray,
    hr_bpm: np.ndarray,
    rr_series: np.ndarray,
    freqs: np.ndarray,
    psd: np.ndarray,
) -> None:
    """Save overview plots (heart rate, tachogram, PSD) as PNG."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    axes[0].plot(t_rr / 60.0, hr_bpm, color="#2E86DE")
    axes[0].set_xlabel("Elapsed time [min]")
    axes[0].set_ylabel("Heart rate [bpm]")
    axes[0].set_title("Heart rate trend")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_rr / 60.0, rr_series, color="#27AE60")
    axes[1].set_xlabel("Elapsed time [min]")
    axes[1].set_ylabel("RR interval [ms]")
    axes[1].set_title("RR tachogram")
    axes[1].grid(alpha=0.3)

    axes[2].semilogy(freqs, psd, color="#8E44AD")
    axes[2].axvspan(0.04, 0.15, color="orange", alpha=0.15, label="LF")
    axes[2].axvspan(0.15, 0.40, color="skyblue", alpha=0.2, label="HF")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("PSD")
    axes[2].set_title("RR tachogram spectrum")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = ensure_dataset(args.cache_dir)
    rr_series = load_rr_series(data_root, args.subject)

    hr_bpm, t_rr, freqs, psd, metrics = compute_hrv(
        rr_series,
        args.interp_freq,
    )

    rr_ms = rr_series.dropna().to_numpy()

    figure_path = output_dir / f"hrv_summary_subject{args.subject}.png"
    save_figure(figure_path, t_rr, hr_bpm, rr_ms, freqs, psd)

    csv_path = output_dir / f"hr_timeseries_subject{args.subject}.csv"
    pd.DataFrame(
        {
            "time_sec": t_rr,
            "heart_rate_bpm": hr_bpm,
            "rr_interval_ms": rr_ms,
        }
    ).to_csv(csv_path, index=False)

    summary: Dict[str, object] = {
        "subject": args.subject,
        "mean_hr_bpm": metrics.mean_hr,
        "sdnn_ms": metrics.sdnn,
        "rmssd_ms": metrics.rmssd,
        "pnn50_percent": metrics.pnn50,
        "lf_power": metrics.lf_power,
        "hf_power": metrics.hf_power,
        "lf_hf_ratio": metrics.lf_hf_ratio,
        "interp_freq_hz": args.interp_freq,
        "figure_path": str(figure_path),
        "timeseries_csv": str(csv_path),
    }

    json_path = output_dir / f"hrv_summary_subject{args.subject}.json"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("=== 解析結果 ===")
    for key, value in summary.items():
        if key in {"subject", "figure_path", "timeseries_csv", "interp_freq_hz"}:
            continue
        print(f"{key}: {value:.4f}")
    print(f"保存: {figure_path}")
    print(f"保存: {csv_path}")
    print(f"保存: {json_path}")


if __name__ == "__main__":
    main()
