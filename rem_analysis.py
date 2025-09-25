#!/usr/bin/env python3
"""REM sleep analysis pipeline for PhysioNet sleep data.

This script downloads a subject recording from the PhysioNet Sleep-EDF
Database (sleep_physionet.age), extracts EOG/EEG channels, computes
band-limited power spectra, detects REM sleep based on simple features,
and visualises the results.

Usage example:
    python rem_analysis.py --subject 0 --recording 1 --output-dir results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.datasets import sleep_physionet
import mne
from scipy import signal
from sklearn.preprocessing import StandardScaler

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


@dataclass
class ChannelSelection:
    eog: List[str]
    eeg: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REM sleep analysis demo")
    parser.add_argument("--subject", type=int, required=True, help="Subject ID (0-82 in dataset)")
    parser.add_argument("--recording", type=int, choices=[1, 2], default=1, help="Recording index (1 or 2)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory to write figures and summary files",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional cache directory for PhysioNet downloads",
    )
    parser.add_argument("--window-size", type=float, default=30.0, help="Sliding window size in seconds")
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Window overlap ratio (0-1). 0 means no overlap, 0.5 means 50%% overlap.",
    )
    return parser.parse_args()


def fetch_dataset(subject: int, recording: int, data_dir: Path | None) -> Tuple[Path, Path | None]:
    """Download PhysioNet data for the requested subject/recording."""
    dataset = sleep_physionet.age.fetch_data(
        subjects=[subject],
        recording=[recording],
        on_missing="raise",
        path=str(data_dir) if data_dir else None,
    )

    if not dataset:
        raise RuntimeError("No data returned from PhysioNet fetch_data")

    first_entry = dataset[0]
    if isinstance(first_entry, (tuple, list)):
        raw_path = Path(first_entry[0])
        ann_path = Path(first_entry[1]) if len(first_entry) > 1 else None
    else:
        raw_path = Path(first_entry)
        ann_path = None

    return raw_path, ann_path


def select_channels(raw: mne.io.BaseRaw, limit: int = 2) -> ChannelSelection:
    """Pick EOG and EEG channel names with sensible fallbacks."""
    picks_eog = mne.pick_types(raw.info, eog=True, include=[], exclude="bads")
    eog_channels = [raw.ch_names[idx] for idx in picks_eog][:limit]

    if not eog_channels:
        eog_channels = [ch for ch in raw.ch_names if "EOG" in ch.upper()][:limit]

    if not eog_channels:
        eeg_fallback = mne.pick_types(raw.info, eeg=True, include=[], exclude="bads")
        eog_channels = [raw.ch_names[idx] for idx in eeg_fallback][:limit]

    picks_eeg = mne.pick_types(raw.info, eeg=True, include=[], exclude="bads")
    eeg_channels = [raw.ch_names[idx] for idx in picks_eeg][:limit]

    if not eeg_channels:
        eeg_channels = [ch for ch in raw.ch_names if "EEG" in ch.upper()][:limit]

    if not eeg_channels:
        remaining = [ch for ch in raw.ch_names if ch not in eog_channels]
        eeg_channels = remaining[:limit]

    return ChannelSelection(eog=eog_channels, eeg=eeg_channels)


def compute_band_powers(data: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Compute PSD and aggregate power within predefined bands."""
    nperseg = int(min(max(sfreq, 128), len(data)))
    freqs, psd = signal.welch(data, sfreq, nperseg=nperseg)
    band_powers: Dict[str, float] = {}
    for band, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        band_powers[band] = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
    return freqs, psd, band_powers


def sliding_windows(n_samples: int, window_samples: int, step_samples: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n_samples - window_samples + 1, step_samples):
        yield start, start + window_samples


def extract_rem_features(
    eog: np.ndarray,
    eeg: np.ndarray,
    sfreq: float,
    window_size: float,
    overlap: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute REM detection features for sliding windows."""
    window_samples = int(window_size * sfreq)
    if window_samples <= 0:
        raise ValueError("window_size must be positive")

    step_samples = max(int(window_samples * (1 - overlap)), 1)
    features: List[List[float]] = []
    centers: List[float] = []
    step_seconds = step_samples / sfreq

    for start, end in sliding_windows(eog.shape[1], window_samples, step_samples):
        segment_center = (start + end) / 2 / sfreq
        eog_segment = eog[:, start:end]
        eeg_segment = eeg[:, start:end]

        eog_std = np.std(eog_segment, axis=1)
        eog_range = np.ptp(eog_segment, axis=1)

        theta_over_delta: List[float] = []
        alpha_power: List[float] = []
        beta_power: List[float] = []

        for channel in eeg_segment:
            freqs, psd = signal.welch(channel, sfreq, nperseg=min(len(channel), 256))
            delta_mask = (freqs >= BANDS["delta"][0]) & (freqs < BANDS["delta"][1])
            theta_mask = (freqs >= BANDS["theta"][0]) & (freqs < BANDS["theta"][1])
            alpha_mask = (freqs >= BANDS["alpha"][0]) & (freqs < BANDS["alpha"][1])
            beta_mask = (freqs >= BANDS["beta"][0]) & (freqs < BANDS["beta"][1])

            delta_power = np.trapz(psd[delta_mask], freqs[delta_mask]) if np.any(delta_mask) else 0.0
            theta_power = np.trapz(psd[theta_mask], freqs[theta_mask]) if np.any(theta_mask) else 0.0
            alpha_power.append(np.trapz(psd[alpha_mask], freqs[alpha_mask]) if np.any(alpha_mask) else 0.0)
            beta_power.append(np.trapz(psd[beta_mask], freqs[beta_mask]) if np.any(beta_mask) else 0.0)
            theta_over_delta.append(theta_power / (delta_power + 1e-10))

        features.append(
            [
                float(np.mean(eog_std)),
                float(np.mean(eog_range)),
                float(np.mean(theta_over_delta) if theta_over_delta else 0.0),
                float(np.mean(alpha_power) if alpha_power else 0.0),
                float(np.mean(beta_power) if beta_power else 0.0),
            ]
        )
        centers.append(segment_center)

    return np.asarray(features), np.asarray(centers), float(step_seconds)


def detect_rem_sleep(features: np.ndarray, percentile: float = 75.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Detect REM segments based on scaled feature scores."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    eog_activity = scaled[:, 0] + scaled[:, 1]
    theta_delta_ratio = scaled[:, 2]
    rem_score = eog_activity + theta_delta_ratio
    threshold = np.percentile(rem_score, percentile)
    rem_detected = rem_score > threshold
    return rem_detected, rem_score, float(threshold)


def summarise_rem(
    rem_flags: np.ndarray,
    window_size: float,
    step_seconds: float,
    centers: np.ndarray,
) -> Dict[str, object]:
    window_minutes = step_seconds / 60.0
    total_rem_minutes = float(np.sum(rem_flags) * window_minutes)
    total_minutes = float(len(centers) * window_minutes)
    percentage = (total_rem_minutes / total_minutes * 100.0) if total_minutes else 0.0

    segments: List[Dict[str, float]] = []
    for value, group in groupby(enumerate(rem_flags), key=lambda item: item[1]):
        if not value:
            continue
        items = list(group)
        start_idx = items[0][0]
        end_idx = items[-1][0]
        start_time = float(centers[start_idx] - window_size / 2)
        end_time = float(centers[end_idx] + window_size / 2)
        segments.append({"start_sec": max(start_time, 0.0), "end_sec": max(end_time, 0.0)})

    return {
        "total_minutes": total_minutes,
        "rem_minutes": total_rem_minutes,
        "rem_percentage": percentage,
        "episode_count": len(segments),
        "segments": segments,
    }


def channels_band_summary(raw: mne.io.BaseRaw, channels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    data = raw.copy().pick(list(channels)).get_data()
    sfreq = raw.info["sfreq"]
    for idx, ch_name in enumerate(channels):
        _, _, band_powers = compute_band_powers(data[idx], sfreq)
        summary[ch_name] = band_powers
    return summary


def create_figure(
    eog_summary: Dict[str, Dict[str, float]],
    eeg_summary: Dict[str, Dict[str, float]],
    times_hours: np.ndarray,
    rem_score: np.ndarray,
    threshold: float,
    rem_flags: np.ndarray,
    segments: Sequence[Dict[str, float]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)

    def plot_band_bars(ax, summary: Dict[str, Dict[str, float]], title: str) -> None:
        if not summary:
            ax.text(0.5, 0.5, "No channels", ha="center", va="center")
            ax.set_axis_off()
            return
        band_names = list(BANDS.keys())
        x = np.arange(len(band_names))
        width = 0.8 / max(1, len(summary))
        for idx, (channel, powers) in enumerate(summary.items()):
            values = [powers.get(band, 0.0) for band in band_names]
            ax.bar(x + idx * width, values, width=width, label=channel)
        ax.set_xticks(x + width * (len(summary) - 1) / 2)
        ax.set_xticklabels(band_names)
        ax.set_ylabel("Power")
        ax.set_title(title)
        ax.legend()

    plot_band_bars(axes[0], eog_summary, "EOG band power (Welch)")
    plot_band_bars(axes[1], eeg_summary, "EEG band power (Welch)")

    axes[2].plot(times_hours, rem_score, label="REM score", color="#2E86DE")
    axes[2].axhline(threshold, linestyle="--", color="#E74C3C", label="threshold")

    for segment in segments:
        axes[2].axvspan(
            segment["start_sec"] / 3600.0,
            segment["end_sec"] / 3600.0,
            color="#AED6F1",
            alpha=0.3,
        )

    axes[2].set_xlabel("Time [hours]")
    axes[2].set_ylabel("Score")
    axes[2].set_title("REM detection")
    axes[2].legend()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def resample_for_plot(data: np.ndarray, sfreq: float, target_sfreq: float = 25.0) -> Tuple[np.ndarray, float]:
    """Down-sample long signals for plotting."""
    if sfreq <= target_sfreq:
        return data, sfreq
    decim = max(int(np.floor(sfreq / target_sfreq)), 1)
    if decim <= 1:
        return data, sfreq
    resampled = signal.resample_poly(data, up=1, down=decim)
    return resampled, sfreq / decim


def create_eog_overview(
    eog_data: np.ndarray,
    sfreq: float,
    channels: Sequence[str],
    segments: Sequence[Dict[str, float]],
    centers_sec: np.ndarray,
    eog_std: np.ndarray,
    eog_range: np.ndarray,
    output_path: Path,
) -> None:
    # EOG波形を可視化し、REM区間をハイライトする
    data_for_plot, plot_sfreq = resample_for_plot(eog_data, sfreq)
    times_min = np.arange(data_for_plot.shape[1]) / plot_sfreq / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    # 上段: チャンネル別にEOG波形を表示
    for idx, channel in enumerate(channels[: data_for_plot.shape[0]]):
        axes[0].plot(times_min, data_for_plot[idx], label=channel, linewidth=0.7)
    for segment in segments:
        axes[0].axvspan(
            segment["start_sec"] / 60.0,
            segment["end_sec"] / 60.0,
            color="#AED6F1",
            alpha=0.25,
        )
    axes[0].set_ylabel("EOG amplitude")
    axes[0].set_title("EOG waveforms with REM segments")
    axes[0].legend(loc="upper right")

    # 下段: 抽出したEOG特徴量の推移
    feature_times_min = centers_sec / 60.0
    axes[1].plot(feature_times_min, eog_std, label="Std", color="#8E44AD")
    axes[1].plot(feature_times_min, eog_range, label="Range", color="#16A085")
    for segment in segments:
        axes[1].axvspan(
            segment["start_sec"] / 60.0,
            segment["end_sec"] / 60.0,
            color="#AED6F1",
            alpha=0.25,
        )
    axes[1].set_xlabel("Elapsed time [min]")
    axes[1].set_ylabel("Feature value")
    axes[1].set_title("EOG feature trajectories")
    axes[1].legend(loc="upper right")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_eeg_psd(
    eeg_data: np.ndarray,
    sfreq: float,
    channels: Sequence[str],
    output_path: Path,
) -> None:
    # EEGチャンネルごとのパワースペクトルを描画する
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    for idx, channel in enumerate(channels[: eeg_data.shape[0]]):
        freqs, psd = signal.welch(eeg_data[idx], sfreq, nperseg=min(len(eeg_data[idx]), int(sfreq * 4)))
        ax.semilogy(freqs, psd, label=channel)

    y_top = ax.get_ylim()[1]
    for band, (low, high) in BANDS.items():
        ax.axvspan(low, high, color="grey", alpha=0.08)
        ax.text((low + high) / 2, y_top * 0.8, band, ha="center", va="center", fontsize=9, alpha=0.7)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.set_title("EEG power spectrum (Welch)")
    ax.legend(loc="upper right")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching PhysioNet data (subject={args.subject}, recording={args.recording})...")
    raw_path, ann_path = fetch_dataset(args.subject, args.recording, args.data_dir)

    print(f"Loading EDF file: {raw_path}")
    raw = mne.io.read_raw_edf(raw_path, preload=True, stim_channel=None, verbose="error")

    if ann_path and ann_path.exists():
        try:
            annotations = mne.read_annotations(ann_path)
            raw.set_annotations(annotations)
            print(f"Applied annotations: {ann_path}")
        except Exception as exc:  # pragma: no cover
            print(f"Annotation load failed ({ann_path}): {exc}")

    selection = select_channels(raw)
    print(f"Using EOG channels: {selection.eog}")
    print(f"Using EEG channels: {selection.eeg}")

    eog_data = raw.copy().pick(selection.eog).get_data()
    eeg_data = raw.copy().pick(selection.eeg).get_data()
    sfreq = raw.info["sfreq"]

    print("Computing REM features...")
    features, centers_sec, step_seconds = extract_rem_features(
        eog=eog_data,
        eeg=eeg_data,
        sfreq=sfreq,
        window_size=args.window_size,
        overlap=args.overlap,
    )

    rem_flags, rem_score, threshold = detect_rem_sleep(features)
    summary = summarise_rem(rem_flags, args.window_size, step_seconds, centers_sec)

    eog_band_summary = channels_band_summary(raw, selection.eog)
    eeg_band_summary = channels_band_summary(raw, selection.eeg)

    times_hours = centers_sec / 3600.0
    figure_path = output_dir / f"rem_analysis_subject{args.subject}_rec{args.recording}.png"
    create_figure(
        eog_summary=eog_band_summary,
        eeg_summary=eeg_band_summary,
        times_hours=times_hours,
        rem_score=rem_score,
        threshold=threshold,
        rem_flags=rem_flags,
        segments=summary["segments"],
        output_path=figure_path,
    )

    print(f"Saved visualisation to {figure_path}")

    eog_plot_path = output_dir / f"eog_overview_subject{args.subject}_rec{args.recording}.png"
    create_eog_overview(
        eog_data=eog_data,
        sfreq=sfreq,
        channels=selection.eog,
        segments=summary["segments"],
        centers_sec=centers_sec,
        eog_std=features[:, 0],
        eog_range=features[:, 1],
        output_path=eog_plot_path,
    )
    print(f"Saved EOG overview to {eog_plot_path}")

    eeg_psd_path = output_dir / f"eeg_psd_subject{args.subject}_rec{args.recording}.png"
    create_eeg_psd(
        eeg_data=eeg_data,
        sfreq=sfreq,
        channels=selection.eeg,
        output_path=eeg_psd_path,
    )
    print(f"Saved EEG PSD to {eeg_psd_path}")

    df = pd.DataFrame(
        {
            "time_hours": times_hours,
            "rem_score": rem_score,
            "rem_detected": rem_flags.astype(int),
            "eog_std": features[:, 0],
            "eog_range": features[:, 1],
            "theta_delta_ratio": features[:, 2],
            "alpha_power": features[:, 3],
            "beta_power": features[:, 4],
        }
    )

    csv_path = output_dir / f"rem_features_subject{args.subject}_rec{args.recording}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved feature CSV to {csv_path}")

    result = {
        "subject": args.subject,
        "recording": args.recording,
        "sampling_rate": sfreq,
        "channels": {
            "eog": selection.eog,
            "eeg": selection.eeg,
        },
        "band_power": {
            "eog": eog_band_summary,
            "eeg": eeg_band_summary,
        },
        "rem_summary": summary,
        "threshold": threshold,
        "window_seconds": args.window_size,
        "step_seconds": step_seconds,
        "overlap": args.overlap,
        "csv_path": str(csv_path),
        "figure_path": str(figure_path),
        "eog_overview_path": str(eog_plot_path),
        "eeg_psd_path": str(eeg_psd_path),
    }

    json_path = output_dir / f"rem_summary_subject{args.subject}_rec{args.recording}.json"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(f"Saved summary JSON to {json_path}")


if __name__ == "__main__":
    main()
