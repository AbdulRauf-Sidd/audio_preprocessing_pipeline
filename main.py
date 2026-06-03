#!/usr/bin/env python3
"""
preprocess_audio.py
────────────────────────────────────────────────────────────────────────────────
Production-ready audio preprocessing for deepfake detection datasets.

Designed for LONG source files (15-20+ minutes). Each source file is
exhaustively segmented into many non-overlapping clips, each randomly sized
between --min-dur and --max-dur seconds. A 20-minute file at 4s average
clip length yields ~300 clips.

Pipeline per clip:
  1. ffmpeg decode → 16 kHz mono PCM WAV
  2. Skip first/last --skip-edges seconds (intro/outro noise)
  3. Exhaustive segmentation with random clip lengths [min-dur, max-dur]
  4. VAD check  — discard if speech ratio < --min-speech
  5. RMS check  — discard if too quiet or too loud
  6. SNR check  — discard if too noisy
  7. Loudness normalisation to --lufs
  8. Deduplication via mel fingerprint
  9. Write 16 kHz mono PCM_16 WAV + manifest CSV

Usage:
  python preprocess_audio.py \
      --input  /data/raw/youtube_fake \
      --output /data/processed/youtube_fake \
      --label  fake \
      --min-dur 2.0 \
      --max-dur 7.0 \
      --workers 4

Requirements:
  pip install librosa soundfile numpy scipy tqdm webrtcvad
  System: ffmpeg must be on PATH
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import hashlib
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# ── Optional: webrtcvad for frame-level VAD ───────────────────────────────────
try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    input_dir:          Path
    output_dir:         Path
    label:              str             # "real" or "fake"
    min_dur:            float = 2.0     # seconds — minimum clip length
    max_dur:            float = 7.0     # seconds — maximum clip length
    skip_edges:         float = 30.0   # seconds to skip at start and end of file
    target_sr:          int   = 16_000
    vad_aggressiveness: int   = 2       # webrtcvad 0-3
    min_speech_ratio:   float = 0.40   # fraction of clip that must be speech
    min_rms_db:         float = -45.0
    max_rms_db:         float = -5.0
    target_lufs:        float = -23.0
    min_snr_db:         float = 5.0
    dedup:              bool  = True
    workers:            int   = 4
    seed:               int   = 42
    extensions: tuple = field(default_factory=lambda: (
        ".wav", ".mp3", ".mp4", ".m4a", ".aac",
        ".ogg", ".flac", ".webm", ".opus", ".mkv", ".mov",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Convert source file → 16 kHz mono WAV
# ─────────────────────────────────────────────────────────────────────────────
def convert_to_wav(src: Path, dst: Path, target_sr: int = 16_000) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ar", str(target_sr),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-vn",
        str(dst),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0 and dst.exists() and dst.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Exhaustive segmentation
#
# Walk the audio from left to right. At each position pick a random clip
# length in [min_dur, max_dur]. This gives natural variation in clip length
# rather than uniform fixed-size chunks, which is better for training.
# ─────────────────────────────────────────────────────────────────────────────
def segment_audio(
    audio: np.ndarray,
    sr:    int,
    min_dur: float,
    max_dur: float,
    skip_edges: float,
    rng: random.Random,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Returns list of (clip_array, start_sample, end_sample).
    Skips the first and last `skip_edges` seconds of the file.
    """
    total_samples  = len(audio)
    edge_samples   = int(skip_edges * sr)
    min_samples    = int(min_dur * sr)
    max_samples    = int(max_dur * sr)

    start = edge_samples
    end   = total_samples - edge_samples

    # Not enough audio after trimming edges
    if (end - start) < min_samples:
        return []

    segments = []
    cursor   = start
    while cursor + min_samples <= end:
        remaining   = end - cursor
        clip_len    = rng.randint(min_samples, min(max_samples, remaining))
        segment     = audio[cursor: cursor + clip_len]
        segments.append((segment, cursor, cursor + clip_len))
        cursor += clip_len          # non-overlapping — move forward by full clip

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — VAD
# ─────────────────────────────────────────────────────────────────────────────
def speech_ratio_webrtcvad(audio: np.ndarray, sr: int, aggressiveness: int = 2) -> float:
    vad       = webrtcvad.Vad(aggressiveness)
    frame_ms  = 30
    frame_len = int(sr * frame_ms / 1000)
    pcm       = (audio * 32768).astype(np.int16).tobytes()

    speech = total = 0
    for start in range(0, len(audio) - frame_len, frame_len):
        frame = pcm[start * 2: (start + frame_len) * 2]
        if len(frame) < frame_len * 2:
            break
        try:
            speech += int(vad.is_speech(frame, sample_rate=sr))
        except Exception:
            continue
        total += 1

    return speech / total if total > 0 else 0.0


def speech_ratio_energy(audio: np.ndarray, sr: int, threshold_db: float = -40.0) -> float:
    frame_len = int(sr * 0.03)
    hop_len   = frame_len // 2
    frames    = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
    rms       = np.sqrt(np.mean(frames ** 2, axis=0))
    rms_db    = 20 * np.log10(np.maximum(rms, 1e-9))
    return float(np.mean(rms_db > threshold_db))


def compute_speech_ratio(audio: np.ndarray, sr: int, aggressiveness: int = 2) -> float:
    if HAS_WEBRTCVAD:
        return speech_ratio_webrtcvad(audio, sr, aggressiveness)
    return speech_ratio_energy(audio, sr)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Quality checks
# ─────────────────────────────────────────────────────────────────────────────
def rms_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    return float(20 * np.log10(max(rms, 1e-9)))


def estimate_snr(audio: np.ndarray, sr: int) -> float:
    frame_len = int(sr * 0.03)
    hop_len   = frame_len // 2
    if len(audio) < frame_len * 2:
        return 0.0

    frames   = librosa.util.frame(audio.astype(np.float32),
                                  frame_length=frame_len, hop_length=hop_len)
    energies = np.mean(frames ** 2, axis=0)
    thresh   = np.percentile(energies, 30)

    signal_e = energies[energies >= thresh]
    noise_e  = energies[energies <  thresh]

    if len(noise_e) == 0 or np.mean(noise_e) == 0:
        return 99.0

    return float(10 * np.log10(np.mean(signal_e) / np.mean(noise_e)))


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Loudness normalisation
# ─────────────────────────────────────────────────────────────────────────────
def normalise_loudness(audio: np.ndarray, target_lufs: float = -23.0) -> np.ndarray:
    current_rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if current_rms < 1e-9:
        return audio
    target_rms = 10 ** (target_lufs / 20.0)
    gain       = target_rms / current_rms
    normalised = np.clip(audio.astype(np.float64) * gain, -1.0, 1.0)
    return normalised.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Deduplication fingerprint
# ─────────────────────────────────────────────────────────────────────────────
def audio_fingerprint(audio: np.ndarray, sr: int, n_mels: int = 32) -> str:
    mel      = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=512)
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    quantised = np.round(mel_db.mean(axis=1), decimals=1).tobytes()
    return hashlib.md5(quantised).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Core per-file worker  ← KEY CHANGE: returns list of results, one per clip
# ─────────────────────────────────────────────────────────────────────────────
def process_file(
    src:               Path,
    output_dir:        Path,
    cfg:               Config,
    seen_fingerprints: Optional[set],
    rng_seed:          int,
) -> List[dict]:
    """
    Processes one long source file and returns a list of result dicts —
    one entry per extracted clip (kept or skipped).
    """
    rng = random.Random(rng_seed)

    def skipped(reason: str) -> List[dict]:
        return [{"src": str(src), "out": None, "skipped": True,
                 "reason": reason, "duration": None, "rms_db": None,
                 "snr_db": None, "speech_ratio": None, "label": cfg.label}]

    # ── 1. Convert whole file to 16 kHz mono WAV ─────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    try:
        if not convert_to_wav(src, tmp_wav, cfg.target_sr):
            return skipped("ffmpeg_error")

        try:
            audio, sr = sf.read(str(tmp_wav), dtype="float32", always_2d=False)
        except Exception:
            return skipped("read_error")

        if sr != cfg.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg.target_sr)
            sr    = cfg.target_sr

        # ── 2. Exhaustive segmentation ────────────────────────────────────────
        segments = segment_audio(
            audio, sr, cfg.min_dur, cfg.max_dur, cfg.skip_edges, rng
        )
        if not segments:
            return skipped("too_short")

        # ── 3-8. Per-clip quality pipeline ───────────────────────────────────
        results = []
        try:
            rel = src.relative_to(cfg.input_dir)
        except ValueError:
            rel = Path(src.name)

        stem = rel.with_suffix("")   # e.g. "channel1/video_001"

        for clip_idx, (clip, start_s, end_s) in enumerate(segments):

            clip_result = {
                "src":          str(src),
                "out":          None,
                "skipped":      False,
                "reason":       None,
                "duration":     round(len(clip) / sr, 3),
                "rms_db":       None,
                "snr_db":       None,
                "speech_ratio": None,
                "label":        cfg.label,
            }

            # VAD
            ratio = compute_speech_ratio(clip, sr, cfg.vad_aggressiveness)
            clip_result["speech_ratio"] = round(ratio, 3)
            if ratio < cfg.min_speech_ratio:
                clip_result.update(skipped=True, reason="low_speech_ratio")
                results.append(clip_result)
                continue

            # RMS
            rms = rms_db(clip)
            clip_result["rms_db"] = round(rms, 2)
            if not (cfg.min_rms_db <= rms <= cfg.max_rms_db):
                clip_result.update(skipped=True, reason="rms_out_of_range")
                results.append(clip_result)
                continue

            # SNR
            snr = estimate_snr(clip, sr)
            clip_result["snr_db"] = round(snr, 2)
            if snr < cfg.min_snr_db:
                clip_result.update(skipped=True, reason="low_snr")
                results.append(clip_result)
                continue

            # Loudness normalisation
            normalised = normalise_loudness(clip, cfg.target_lufs)

            # Deduplication
            if cfg.dedup and seen_fingerprints is not None:
                fp = audio_fingerprint(normalised, sr)
                if fp in seen_fingerprints:
                    clip_result.update(skipped=True, reason="duplicate")
                    results.append(clip_result)
                    continue
                seen_fingerprints.add(fp)

            # Write clip — named as  stem_clip0001.wav
            out_path = output_dir / stem.parent / f"{stem.name}_clip{clip_idx:05d}.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), normalised, sr, subtype="PCM_16")

            clip_result["out"] = str(out_path)
            results.append(clip_result)

        return results

    finally:
        tmp_wav.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────
SKIP_REASONS = [
    "ffmpeg_error", "read_error", "too_short",
    "low_speech_ratio", "rms_out_of_range", "low_snr", "duplicate",
]


def collect_files(input_dir: Path, extensions: tuple) -> List[Path]:
    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
        files.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def run(cfg: Config) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found on PATH. Please install ffmpeg.")
        sys.exit(1)

    if not HAS_WEBRTCVAD:
        log.warning(
            "webrtcvad not installed — using energy-based VAD fallback. "
            "pip install webrtcvad for better accuracy."
        )

    log.info(f"Scanning {cfg.input_dir} ...")
    files = collect_files(cfg.input_dir, cfg.extensions)
    if not files:
        log.error("No audio/video files found.")
        sys.exit(1)
    log.info(f"Found {len(files):,} source file(s).")

    seen_fingerprints: set = set()
    skip_counts  = {k: 0 for k in SKIP_REASONS}
    clips_kept   = 0
    results_rows = []

    manifest_path = cfg.output_dir / "manifest.csv"

    # Note: dedup requires sequential processing (shared fingerprint set).
    # Without dedup, parallel processing is safe and much faster.
    use_parallel = cfg.workers > 1 and not cfg.dedup

    if cfg.dedup and cfg.workers > 1:
        log.warning("Deduplication requires sequential processing — workers ignored.")

    log.info(
        f"Mode: {'parallel (' + str(cfg.workers) + ' workers)' if use_parallel else 'sequential'} | "
        f"Dedup: {cfg.dedup} | "
        f"Clip: {cfg.min_dur}–{cfg.max_dur}s | "
        f"Skip edges: {cfg.skip_edges}s"
    )

    def handle_results(all_clip_results: List[dict]) -> None:
        nonlocal clips_kept
        for r in all_clip_results:
            if r["skipped"]:
                reason = r.get("reason", "unknown")
                if reason in skip_counts:
                    skip_counts[reason] += 1
            else:
                clips_kept += 1
                results_rows.append(r)

    if use_parallel:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {
                pool.submit(process_file, src, cfg.output_dir, cfg, None, cfg.seed + i): src
                for i, src in enumerate(files)
            }
            with tqdm(total=len(futures), unit="file", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    handle_results(future.result())
                    pbar.update(1)
                    pbar.set_postfix(clips=clips_kept, skip=sum(skip_counts.values()))
    else:
        with tqdm(files, unit="file", dynamic_ncols=True) as pbar:
            for i, src in enumerate(pbar):
                handle_results(
                    process_file(src, cfg.output_dir, cfg, seen_fingerprints, cfg.seed + i)
                )
                pbar.set_postfix(clips=clips_kept, skip=sum(skip_counts.values()))

    # ── Manifest ──────────────────────────────────────────────────────────────
    if results_rows:
        fieldnames = ["src", "out", "label", "duration", "rms_db", "snr_db", "speech_ratio"]
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results_rows)
        log.info(f"Manifest → {manifest_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_skipped = sum(skip_counts.values())
    total_clips   = clips_kept + total_skipped

    avg_dur = (
        sum(r["duration"] for r in results_rows if r["duration"]) / clips_kept
        if clips_kept else 0
    )

    print("\n" + "─" * 60)
    print("  PREPROCESSING SUMMARY")
    print("─" * 60)
    print(f"  Source files          : {len(files):>8,}")
    print(f"  Total clips extracted : {total_clips:>8,}")
    print(f"  Clips kept            : {clips_kept:>8,}")
    print(f"  Clips skipped         : {total_skipped:>8,}")
    print(f"  Average clip duration : {avg_dur:>7.2f}s")
    print()
    print("  Skip breakdown:")
    for reason, count in skip_counts.items():
        if count:
            print(f"    {reason:<24}: {count:>6,}")
    print("─" * 60)
    print(f"  Output : {cfg.output_dir}")
    print(f"  Manifest : {manifest_path}")
    print("─" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess long-form audio into short clips for deepfake detection training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      required=True,  type=Path,  help="Input directory")
    p.add_argument("--output",     required=True,  type=Path,  help="Output directory")
    p.add_argument("--label",      required=True,  choices=["real", "fake"])
    p.add_argument("--min-dur",    type=float, default=2.0,    help="Min clip duration (s)")
    p.add_argument("--max-dur",    type=float, default=7.0,    help="Max clip duration (s)")
    p.add_argument("--skip-edges", type=float, default=30.0,   help="Seconds to skip at start/end of each file")
    p.add_argument("--sr",         type=int,   default=16_000, help="Target sample rate")
    p.add_argument("--vad-mode",   type=int,   default=2,      help="webrtcvad aggressiveness 0-3")
    p.add_argument("--min-speech", type=float, default=0.40,   help="Minimum speech ratio 0-1")
    p.add_argument("--min-rms",    type=float, default=-45.0,  help="Min RMS dBFS")
    p.add_argument("--max-rms",    type=float, default=-5.0,   help="Max RMS dBFS")
    p.add_argument("--lufs",       type=float, default=-23.0,  help="Target loudness dBFS")
    p.add_argument("--min-snr",    type=float, default=5.0,    help="Min SNR dB")
    p.add_argument("--no-dedup",   action="store_true",        help="Disable dedup (enables parallel processing)")
    p.add_argument("--workers",    type=int,   default=4,      help="Workers for parallel mode (requires --no-dedup)")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = Config(
        input_dir          = args.input.resolve(),
        output_dir         = args.output.resolve(),
        label              = args.label,
        min_dur            = args.min_dur,
        max_dur            = args.max_dur,
        skip_edges         = args.skip_edges,
        target_sr          = args.sr,
        vad_aggressiveness = args.vad_mode,
        min_speech_ratio   = args.min_speech,
        min_rms_db         = args.min_rms,
        max_rms_db         = args.max_rms,
        target_lufs        = args.lufs,
        min_snr_db         = args.min_snr,
        dedup              = not args.no_dedup,
        workers            = args.workers,
        seed               = args.seed,
    )

    log.info("Audio preprocessing pipeline starting")
    t0 = time.time()
    run(cfg)
    log.info(f"Finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()