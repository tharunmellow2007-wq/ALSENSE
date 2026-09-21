"""
extract_literature_features.py
================================
Adds the literature-identified high-value features from Vashkevich &
Rushkevich (2021) to your existing 82-feature CSV, WITHOUT re-running the
full hour-long extraction:

  - PVI (Pathological Vibrato Index): ratio of F0-contour spectral energy in
    the 9-14 Hz 'pathological vibrato' band to the 3-8 Hz 'normal vibrato'
    band. One of the two features their paper's own analysis found best
    separated ALS from HC.
  - Harmonic-energy structure (H1-H8 mean, H1-H8 std, RelH1-RelH8): spectral
    magnitude at each of the first 8 harmonics per frame, aggregated. Their
    LASSO-selected 32-feature model included 10 harmonic-structure features -
    the single largest contributing group.

These merge into your existing feature CSV by filename, adding 25 new
columns (1 PVI + 8 H_mean + 8 H_std + 8 RelH) to your existing 82.

USAGE
-----
1. Edit AUDIO_FOLDER and EXISTING_CSV_PATH below.
2. Run: python extract_literature_features.py
3. Output: als_voice_features_with_literature_feats.csv, next to your
   existing CSV, with all 82 original + 25 new columns.

Requires: praat-parselmouth, librosa, soundfile, numpy, scipy, pandas, tqdm
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import librosa
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
AUDIO_FOLDER = r"D:\ABDM\Compressed\ALS_VOICE_SYNAPSE_extracted"
EXISTING_CSV_PATH = r"D:\ABDM\Compressed\als_voice_features.csv"
F0_MIN, F0_MAX = 75, 500
N_HARMONICS = 8


# =============================================================================
# Feature functions
# =============================================================================
def compute_f0_contour(snd, f0min=F0_MIN, f0max=F0_MAX, time_step=0.005):
    pitch = call(snd, "To Pitch", time_step, f0min, f0max)
    times = pitch.xs()
    f0 = np.array([pitch.get_value_at_time(t) for t in times])
    return times, f0


def compute_pvi(times, f0, normal_band=(3, 8), pathological_band=(9, 14)):
    voiced = f0 > 0
    if voiced.sum() < 20:
        return np.nan
    f0v = f0[voiced]
    t_step = np.median(np.diff(times))
    fs = 1 / t_step
    f0_detrended = f0v - np.mean(f0v)
    spec = np.abs(rfft(f0_detrended)) ** 2
    freqs = rfftfreq(len(f0_detrended), d=1 / fs)
    normal_mask = (freqs >= normal_band[0]) & (freqs <= normal_band[1])
    path_mask = (freqs >= pathological_band[0]) & (freqs <= pathological_band[1])
    normal_energy = spec[normal_mask].sum()
    path_energy = spec[path_mask].sum()
    if normal_energy == 0:
        return np.nan
    return path_energy / normal_energy


def compute_harmonic_energy_features(y, sr, f0_times, f0_values, n_harmonics=N_HARMONICS,
                                      frame_len=0.04):
    frame_samples = int(frame_len * sr)
    if frame_samples < 8:
        frame_samples = 8
    window = np.hanning(frame_samples)
    harmonic_db = [[] for _ in range(n_harmonics)]

    for t, f0 in zip(f0_times, f0_values):
        if f0 <= 0:
            continue
        center = int(t * sr)
        start, end = center - frame_samples // 2, center + frame_samples // 2
        if start < 0 or end > len(y):
            continue
        frame = y[start:end] * window
        spec = np.abs(rfft(frame))
        freqs = rfftfreq(frame_samples, d=1 / sr)
        for k in range(1, n_harmonics + 1):
            target_freq = k * f0
            if target_freq >= freqs[-1]:
                continue
            idx = np.argmin(np.abs(freqs - target_freq))
            mag = spec[idx]
            harmonic_db[k - 1].append(20 * np.log10(mag + 1e-12))

    feats = {}
    h_means = []
    for k in range(n_harmonics):
        vals = harmonic_db[k]
        feats[f"H{k+1}_mean"] = np.mean(vals) if vals else np.nan
        feats[f"H{k+1}_std"] = np.std(vals) if vals else np.nan
        h_means.append(feats[f"H{k+1}_mean"])

    h1 = h_means[0]
    for k in range(n_harmonics):
        valid = h1 is not None and not np.isnan(h1) and not np.isnan(h_means[k])
        feats[f"RelH{k+1}"] = (h_means[k] - h1) if valid else np.nan
    return feats


def extract_literature_features(filepath: Path):
    y, sr = librosa.load(str(filepath), sr=None, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < sr * 0.2:
        raise ValueError("Signal too short after silence trimming")

    snd = parselmouth.Sound(str(filepath))
    times, f0 = compute_f0_contour(snd)

    feats = {"PVI": compute_pvi(times, f0)}
    feats.update(compute_harmonic_energy_features(y, sr, times, f0))
    return feats


# =============================================================================
# MAIN
# =============================================================================
def main():
    audio_folder = Path(AUDIO_FOLDER if len(sys.argv) < 2 else sys.argv[1])
    existing_csv = Path(EXISTING_CSV_PATH if len(sys.argv) < 3 else sys.argv[2])

    if not existing_csv.exists():
        print(f"[error] Existing features CSV not found: {existing_csv}")
        return

    existing_df = pd.read_csv(existing_csv)
    print(f"[info] Loaded existing CSV: {existing_df.shape[0]} rows, {existing_df.shape[1]} columns")

    out_path = existing_csv.parent / "als_voice_features_with_literature_feats.csv"
    fieldnames = list(existing_df.columns) + ["PVI"] + \
        [f"H{k}_mean" for k in range(1, N_HARMONICS + 1)] + \
        [f"H{k}_std" for k in range(1, N_HARMONICS + 1)] + \
        [f"RelH{k}" for k in range(1, N_HARMONICS + 1)]

    import csv
    fh = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()

    n_ok, failed = 0, []
    try:
        for _, row in tqdm(existing_df.iterrows(), total=len(existing_df), desc="Adding literature features"):
            filepath = audio_folder / row["filepath"]
            try:
                lit_feats = extract_literature_features(filepath)
                merged = {**row.to_dict(), **lit_feats}
                writer.writerow(merged)
                fh.flush()
                n_ok += 1
            except Exception as e:
                failed.append((str(filepath), str(e)))
    finally:
        fh.close()

    print(f"\n[done] Added literature features for {n_ok} / {len(existing_df)} files.")
    print(f"[done] Output: {out_path}")
    if failed:
        print(f"[warning] {len(failed)} files failed. First few:")
        for fp, err in failed[:5]:
            print(f"  - {fp}: {err}")


if __name__ == "__main__":
    main()
