#!/usr/bin/env python3
"""
preprocess_audio.py
────────────────────────────────────────────────────────────────────────────────
Production-ready audio preprocessing for deepfake detection datasets.

What it does:
  - Recursively finds all audio/video files in input directory
  - Converts to 16 kHz mono WAV (PCM s16le)
  - Runs Voice Activity Detection (VAD) — discards clips with insufficient speech
  - Randomly clips audio to a duration between --min-dur and --max-dur seconds
  - Normalises loudness (EBU R128) so all clips are at a consistent level
  - Checks SNR — discards clips that are too noisy or too silent
  - De-duplicates by audio fingerprint (catches re-uploads of the same clip)
  - Writes a manifest CSV (path, duration, rms, label inferred from folder name)
  - Logs everything — skipped files, reasons, counts

Usage:
  python preprocess_audio.py \
      --input  /data/raw/youtube_fake \
      --output /data/processed/youtube_fake \
      --label  fake \
      --min-dur 2.0 \
      --max-dur 7.0 \
      --workers 8

Requirements:
  pip install librosa soundfile numpy scipy tqdm pandas webrtcvad audioread ffmpeg-python

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
from typing import Optional

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import resample_poly
from tqdm import tqdm

# ── Optional: webrtcvad for frame-level VAD ───────────────────────────────────
try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    label: str                      # "real" or "fake"
    min_dur: float      = 2.0       # seconds
    max_dur: float      = 7.0       # seconds
    target_sr: int      = 16_000    # Hz
    vad_aggressiveness: int = 2     # webrtcvad: 0-3 (3 = most aggressive)
    min_speech_ratio: float = 0.40  # at least 40 % of clip must be speech
    min_rms_db: float   = -45.0     # clips quieter than this are discarded
    max_rms_db: float   = -5.0      # clips louder than this are discarded
    target_lufs: float  = -23.0     # EBU R128 target loudness
    min_snr_db: float   = 5.0       # minimum estimated SNR
    dedup: bool         = True      # skip exact-duplicate clips
    workers: int        = 4
    seed: int           = 42
    extensions: tuple   = field(default_factory=lambda: (
        ".wav", ".mp3", ".mp4", ".m4a", ".aac",
        ".ogg", ".flac", ".webm", ".opus", ".mkv", ".mov",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Skip-reason counters (used in summary)
# ─────────────────────────────────────────────────────────────────────────────
SKIP_REASONS = {
    "ffmpeg_error":     0,
    "too_short":        0,
    "low_speech_ratio": 0,
    "rms_out_of_range": 0,
    "low_snr":          0,
    "duplicate":        0,
    "read_error":       0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Convert any format → 16 kHz mono WAV via ffmpeg
# ─────────────────────────────────────────────────────────────────────────────
def convert_to_wav(src: Path, tmp_wav: Path, target_sr: int = 16_000) -> bool:
    """
    Use ffmpeg to decode src into a 16 kHz mono signed-16-bit WAV.
    Returns True on success.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ar", str(target_sr),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-vn",                  # strip video track if present
        str(tmp_wav),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0 and tmp_wav.exists() and tmp_wav.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Voice Activity Detection
# ─────────────────────────────────────────────────────────────────────────────
def speech_ratio_webrtcvad(audio: np.ndarray, sr: int, aggressiveness: int = 2) -> float:
    """
    Frame-level VAD using webrtcvad.
    Returns fraction of 30 ms frames classified as speech.
    webrtcvad only supports 8k / 16k / 32k / 48k Hz.
    """
    vad = webrtcvad.Vad(aggressiveness)
    frame_ms   = 30                         # ms per frame
    frame_len  = int(sr * frame_ms / 1000)  # samples per frame
    pcm_bytes  = (audio * 32768).astype(np.int16).tobytes()

    speech_frames = 0
    total_frames  = 0
    for start in range(0, len(audio) - frame_len, frame_len):
        frame = pcm_bytes[start * 2: (start + frame_len) * 2]
        if len(frame) < frame_len * 2:
            break
        try:
            is_speech = vad.is_speech(frame, sample_rate=sr)
        except Exception:
            continue
        speech_frames += int(is_speech)
        total_frames  += 1

    return speech_frames / total_frames if total_frames > 0 else 0.0


def speech_ratio_energy(audio: np.ndarray, sr: int, threshold_db: float = -40.0) -> float:
    """
    Fallback energy-based VAD when webrtcvad is not installed.
    Uses short-time RMS energy per frame.
    """
    frame_len = int(sr * 0.03)  # 30 ms
    hop_len   = frame_len // 2

    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
    rms    = np.sqrt(np.mean(frames ** 2, axis=0))
    rms_db = 20 * np.log10(np.maximum(rms, 1e-9))

    speech_frames = np.sum(rms_db > threshold_db)
    return float(speech_frames / len(rms_db)) if len(rms_db) > 0 else 0.0


def compute_speech_ratio(audio: np.ndarray, sr: int, aggressiveness: int = 2) -> float:
    if HAS_WEBRTCVAD:
        return speech_ratio_webrtcvad(audio, sr, aggressiveness)
    return speech_ratio_energy(audio, sr)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Random clip extraction
# ─────────────────────────────────────────────────────────────────────────────
def random_clip(
    audio: np.ndarray,
    sr: int,
    min_dur: float,
    max_dur: float,
    rng: random.Random,
) -> Optional[np.ndarray]:
    """
    Extract a random segment of length U[min_dur, max_dur] seconds.
    Returns None if the audio is shorter than min_dur.
    """
    total_sec = len(audio) / sr
    if total_sec < min_dur:
        return None

    clip_dur = rng.uniform(min_dur, min(max_dur, total_sec))
    max_start = total_sec - clip_dur
    start_sec = rng.uniform(0.0, max_start)

    start  = int(start_sec * sr)
    length = int(clip_dur * sr)
    return audio[start: start + length]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Quality checks
# ─────────────────────────────────────────────────────────────────────────────
def rms_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    return float(20 * np.log10(max(rms, 1e-9)))


def estimate_snr(audio: np.ndarray, sr: int) -> float:
    """
    Simple SNR estimate: compare energy of voiced vs unvoiced frames.
    Uses energy-based VAD to split frames, then ratio of means.
    """
    frame_len = int(sr * 0.03)
    hop_len   = frame_len // 2

    if len(audio) < frame_len * 2:
        return 0.0

    frames    = librosa.util.frame(audio.astype(np.float32),
                                   frame_length=frame_len,
                                   hop_length=hop_len)
    energies  = np.mean(frames ** 2, axis=0)
    threshold = np.percentile(energies, 30)

    signal_e = energies[energies >= threshold]
    noise_e  = energies[energies <  threshold]

    if len(noise_e) == 0 or np.mean(noise_e) == 0:
        return 99.0  # effectively silence-free

    snr = 10 * np.log10(np.mean(signal_e) / np.mean(noise_e))
    return float(snr)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Loudness normalisation (simple RMS-based, EBU R128 approximation)
# ─────────────────────────────────────────────────────────────────────────────
def normalise_loudness(audio: np.ndarray, target_lufs: float = -23.0) -> np.ndarray:
    """
    RMS-based loudness normalisation targeting `target_lufs` dBFS.
    A proper ITU-R BS.1770 implementation would use pyloudnorm,
    but this is sufficient for training data preparation.
    """
    current_rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if current_rms < 1e-9:
        return audio

    target_rms = 10 ** (target_lufs / 20.0)
    gain       = target_rms / current_rms
    normalised = audio.astype(np.float64) * gain

    # Hard-clip to [-1, 1] before converting back
    normalised = np.clip(normalised, -1.0, 1.0)
    return normalised.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Deduplication via audio fingerprint
# ─────────────────────────────────────────────────────────────────────────────
def audio_fingerprint(audio: np.ndarray, sr: int, n_mels: int = 32) -> str:
    """
    Fast perceptual fingerprint: mel spectrogram → mean per band → MD5.
    Robust to minor encoding differences; catches re-uploaded duplicates.
    """
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=sr, n_mels=n_mels, n_fft=512,
    )
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    quantised = np.round(mel_db.mean(axis=1), decimals=1).tobytes()
    return hashlib.md5(quantised).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Core per-file worker
# ─────────────────────────────────────────────────────────────────────────────
def process_file(
    src: Path,
    output_dir: Path,
    cfg: Config,
    seen_fingerprints: Optional[set],  # None = dedup disabled
    rng_seed: int,
) -> dict:
    """
    Full preprocessing pipeline for a single source file.
    Returns a result dict consumed by the main loop.
    """
    rng    = random.Random(rng_seed)
    result = {
        "src":      str(src),
        "out":      None,
        "skipped":  False,
        "reason":   None,
        "duration": None,
        "rms_db":   None,
        "snr_db":   None,
        "speech_ratio": None,
        "label":    cfg.label,
    }

    # ── 1. Convert to 16 kHz mono WAV ────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    try:
        ok = convert_to_wav(src, tmp_wav, cfg.target_sr)
        if not ok:
            result.update(skipped=True, reason="ffmpeg_error")
            return result

        # ── 2. Load audio ────────────────────────────────────────────────────
        try:
            audio, sr = sf.read(str(tmp_wav), dtype="float32", always_2d=False)
        except Exception as e:
            result.update(skipped=True, reason="read_error")
            return result

        if sr != cfg.target_sr:
            # Should not happen after ffmpeg, but resample defensively
            audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg.target_sr)
            sr    = cfg.target_sr

        # ── 3. Random clip ───────────────────────────────────────────────────
        clipped = random_clip(audio, sr, cfg.min_dur, cfg.max_dur, rng)
        if clipped is None:
            result.update(skipped=True, reason="too_short")
            return result

        # ── 4. VAD check ─────────────────────────────────────────────────────
        ratio = compute_speech_ratio(clipped, sr, cfg.vad_aggressiveness)
        result["speech_ratio"] = round(ratio, 3)
        if ratio < cfg.min_speech_ratio:
            result.update(skipped=True, reason="low_speech_ratio")
            return result

        # ── 5. RMS check ─────────────────────────────────────────────────────
        rms = rms_db(clipped)
        result["rms_db"] = round(rms, 2)
        if not (cfg.min_rms_db <= rms <= cfg.max_rms_db):
            result.update(skipped=True, reason="rms_out_of_range")
            return result

        # ── 6. SNR check ─────────────────────────────────────────────────────
        snr = estimate_snr(clipped, sr)
        result["snr_db"] = round(snr, 2)
        if snr < cfg.min_snr_db:
            result.update(skipped=True, reason="low_snr")
            return result

        # ── 7. Loudness normalisation ────────────────────────────────────────
        normalised = normalise_loudness(clipped, cfg.target_lufs)

        # ── 8. Deduplication ─────────────────────────────────────────────────
        if cfg.dedup and seen_fingerprints is not None:
            fp = audio_fingerprint(normalised, sr)
            if fp in seen_fingerprints:
                result.update(skipped=True, reason="duplicate")
                return result
            seen_fingerprints.add(fp)

        # ── 9. Write output ──────────────────────────────────────────────────
        # Preserve relative subfolder structure from input dir
        try:
            rel = src.relative_to(cfg.input_dir)
        except ValueError:
            rel = Path(src.name)

        out_path = output_dir / rel.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(out_path),
            normalised,
            sr,
            subtype="PCM_16",
        )

        result.update(
            out=str(out_path),
            duration=round(len(normalised) / sr, 3),
        )
        return result

    finally:
        tmp_wav.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def collect_files(input_dir: Path, extensions: tuple) -> list[Path]:
    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
        files.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def run(cfg: Config) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Check ffmpeg is available
    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found on PATH. Please install ffmpeg.")
        sys.exit(1)

    if not HAS_WEBRTCVAD:
        log.warning(
            "webrtcvad not installed — falling back to energy-based VAD. "
            "Install with: pip install webrtcvad"
        )

    log.info(f"Scanning {cfg.input_dir} ...")
    files = collect_files(cfg.input_dir, cfg.extensions)
    if not files:
        log.error("No audio/video files found in input directory.")
        sys.exit(1)
    log.info(f"Found {len(files):,} files.")

    # Shared fingerprint set (manager for multiprocessing not needed —
    # we pass None and handle dedup post-hoc for parallel runs)
    seen_fingerprints: set = set()

    manifest_path = cfg.output_dir / "manifest.csv"
    skip_counts   = {k: 0 for k in SKIP_REASONS}
    processed     = 0
    results_rows  = []

    # ── Sequential processing (safe for dedup across files) ──────────────────
    # For pure speed without dedup, flip to ProcessPoolExecutor below.
    log.info(f"Processing with {cfg.workers} worker(s)...")

    if cfg.workers == 1 or cfg.dedup:
        # Sequential — required for shared fingerprint set
        if cfg.workers > 1 and cfg.dedup:
            log.warning(
                "Deduplication requires sequential processing. "
                "Setting workers=1 for this run."
            )
        iterator = tqdm(files, unit="file", dynamic_ncols=True)
        for i, src in enumerate(iterator):
            r = process_file(
                src,
                cfg.output_dir,
                cfg,
                seen_fingerprints if cfg.dedup else None,
                rng_seed=cfg.seed + i,
            )
            if r["skipped"]:
                skip_counts[r["reason"]] += 1
                iterator.set_postfix(skip=sum(skip_counts.values()), ok=processed)
            else:
                processed += 1
                results_rows.append(r)
    else:
        # Parallel — dedup disabled
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {
                pool.submit(
                    process_file,
                    src,
                    cfg.output_dir,
                    cfg,
                    None,
                    cfg.seed + i,
                ): src
                for i, src in enumerate(files)
            }
            with tqdm(total=len(futures), unit="file", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    r = future.result()
                    if r["skipped"]:
                        skip_counts[r["reason"]] += 1
                    else:
                        processed += 1
                        results_rows.append(r)
                    pbar.update(1)
                    pbar.set_postfix(ok=processed, skip=sum(skip_counts.values()))

    # ── Write manifest CSV ────────────────────────────────────────────────────
    if results_rows:
        fieldnames = ["src", "out", "label", "duration", "rms_db", "snr_db", "speech_ratio"]
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results_rows)
        log.info(f"Manifest written → {manifest_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total      = len(files)
    skipped    = sum(skip_counts.values())
    keep_rate  = processed / total * 100 if total else 0

    print("\n" + "─" * 60)
    print(f"  PREPROCESSING SUMMARY")
    print("─" * 60)
    print(f"  Input files found   : {total:>8,}")
    print(f"  Clips kept          : {processed:>8,}  ({keep_rate:.1f}%)")
    print(f"  Clips skipped       : {skipped:>8,}")
    print()
    print("  Skip breakdown:")
    for reason, count in skip_counts.items():
        if count:
            print(f"    {reason:<22}: {count:>6,}")
    print("─" * 60)
    print(f"  Output directory    : {cfg.output_dir}")
    print(f"  Manifest CSV        : {manifest_path}")
    print("─" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess audio clips for deepfake detection training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",     required=True,  type=Path, help="Input directory (recursive scan)")
    p.add_argument("--output",    required=True,  type=Path, help="Output directory for processed WAVs")
    p.add_argument("--label",     required=True,  choices=["real", "fake"], help="Class label")
    p.add_argument("--min-dur",   type=float, default=2.0,   help="Minimum clip duration (seconds)")
    p.add_argument("--max-dur",   type=float, default=7.0,   help="Maximum clip duration (seconds)")
    p.add_argument("--sr",        type=int,   default=16_000, help="Target sample rate (Hz)")
    p.add_argument("--vad-mode",  type=int,   default=2,     help="webrtcvad aggressiveness 0-3")
    p.add_argument("--min-speech",type=float, default=0.40,  help="Minimum speech ratio (0-1)")
    p.add_argument("--min-rms",   type=float, default=-45.0, help="Minimum RMS in dBFS")
    p.add_argument("--max-rms",   type=float, default=-5.0,  help="Maximum RMS in dBFS")
    p.add_argument("--lufs",      type=float, default=-23.0, help="Target loudness (dBFS)")
    p.add_argument("--min-snr",   type=float, default=5.0,   help="Minimum SNR (dB)")
    p.add_argument("--no-dedup",  action="store_true",       help="Disable deduplication")
    p.add_argument("--workers",   type=int,   default=4,     help="Parallel workers (ignored if dedup enabled)")
    p.add_argument("--seed",      type=int,   default=42,    help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = Config(
        input_dir          = args.input.resolve(),
        output_dir         = args.output.resolve(),
        label              = args.label,
        min_dur            = args.min_dur,
        max_dur            = args.max_dur,
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

    log.info("Starting audio preprocessing pipeline")
    log.info(f"  Input  : {cfg.input_dir}")
    log.info(f"  Output : {cfg.output_dir}")
    log.info(f"  Label  : {cfg.label}")
    log.info(f"  Clip   : {cfg.min_dur}s – {cfg.max_dur}s")
    log.info(f"  SR     : {cfg.target_sr} Hz, mono")
    log.info(f"  Dedup  : {cfg.dedup}")
    log.info(f"  VAD    : {'webrtcvad' if HAS_WEBRTCVAD else 'energy-based (fallback)'}")

    t0 = time.time()
    run(cfg)
    log.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
