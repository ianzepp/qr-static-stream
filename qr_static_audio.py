"""
QR Static Stream - Audio Static Variant

Embeds QR pattern into audio white noise.
Audio samples are grouped into "frames" that map to 2D QR modules.

Cover medium: Pre-generated white noise WAV (seeded for reproducibility)
Encoding: Add small offset to samples based on QR module polarity
Decoding: Accumulate frames, signal emerges from noise

Frame structure:
    frame_size samples → sqrt(frame_size) × sqrt(frame_size) virtual 2D frame
    e.g., 4096 samples → 64×64 virtual frame
"""

from __future__ import annotations

import numpy as np
import wave
import struct
from pathlib import Path

# Reuse QR generation and scanning from binary module
from qr_static_binary import generate_qr_matrix, scan_qr


SAMPLE_RATE = 44100
FRAME_SIZE = 64 * 64  # 4096 samples = 64x64 virtual frame
FRAME_SHAPE = (64, 64)


def generate_cover_wav(
    filepath: str | Path,
    duration_seconds: float = 30.0,
    seed: int = 42,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """
    Generate a seeded white noise WAV file as cover medium.

    Args:
        filepath: Output WAV file path
        duration_seconds: Duration of audio
        seed: Random seed for reproducibility
        sample_rate: Audio sample rate (default 44100 Hz)
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_seconds * sample_rate)

    # Generate white noise in [-1, 1] range
    samples = rng.uniform(-1, 1, n_samples).astype(np.float32)

    # Convert to 16-bit PCM
    pcm_samples = (samples * 32767).astype(np.int16)

    # Write WAV file
    with wave.open(str(filepath), 'wb') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_samples.tobytes())

    print(f"Generated cover WAV: {filepath}")
    print(f"  Duration: {duration_seconds}s")
    print(f"  Samples: {n_samples}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Seed: {seed}")


def load_cover_wav(filepath: str | Path) -> tuple[np.ndarray, int]:
    """
    Load cover WAV file as float samples.

    Returns:
        (samples as float32 in [-1, 1], sample_rate)
    """
    with wave.open(str(filepath), 'rb') as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()

        raw_data = wav.readframes(n_frames)

    if sample_width == 2:
        samples = np.frombuffer(raw_data, dtype=np.int16)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if n_channels == 2:
        samples = samples[::2]  # Take left channel only

    # Convert to float [-1, 1]
    samples = samples.astype(np.float32) / 32767.0

    return samples, sample_rate


def encode_audio(
    cover_samples: np.ndarray,
    qr_key: str,
    n_frames: int = 60,
    flip_probability: float = 0.4,
    frame_size: int = FRAME_SIZE,
    seed: int = 12345,
) -> np.ndarray:
    """
    Encode QR pattern into audio samples using sign biasing.

    Instead of adding a small offset (which gets lost in noise), we
    probabilistically flip sample signs to match the QR pattern:
    - White modules: bias toward positive samples
    - Black modules: bias toward negative samples

    Args:
        cover_samples: Original white noise samples (float32)
        qr_key: String to encode in QR code
        n_frames: Number of audio frames per QR decode cycle
        flip_probability: Probability of flipping "wrong" signs (0.0-1.0)
        frame_size: Samples per virtual frame (must be perfect square)
        seed: Random seed for flip decisions

    Returns:
        Modified audio samples with embedded QR pattern
    """
    frame_dim = int(np.sqrt(frame_size))
    assert frame_dim * frame_dim == frame_size, "frame_size must be perfect square"

    # Generate QR and create sign pattern
    qr = generate_qr_matrix(qr_key, frame_dim)
    # white=0 → want positive (+1), black=1 → want negative (-1)
    desired_signs = np.where(qr == 0, +1, -1).astype(np.float32)
    desired_flat = desired_signs.flatten()

    # Apply sign biasing
    rng = np.random.default_rng(seed)
    modified = cover_samples.copy()
    n_samples = len(modified)

    for i in range(n_samples):
        module_idx = i % frame_size
        desired = desired_flat[module_idx]
        current_sign = np.sign(modified[i])

        # If sample sign doesn't match desired, maybe flip it
        if current_sign != desired and current_sign != 0:
            if rng.random() < flip_probability:
                modified[i] = -modified[i]

    return modified


def save_wav(
    filepath: str | Path,
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Save float samples as WAV file."""
    pcm_samples = (np.clip(samples, -1, 1) * 32767).astype(np.int16)

    with wave.open(str(filepath), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_samples.tobytes())


def decode_audio(
    samples: np.ndarray,
    n_frames: int = 60,
    frame_size: int = FRAME_SIZE,
) -> tuple[str | None, np.ndarray]:
    """
    Decode QR pattern from audio samples.

    Args:
        samples: Audio samples (should contain n_frames worth)
        n_frames: Number of frames to accumulate
        frame_size: Samples per virtual frame

    Returns:
        (decoded QR content or None, accumulated 2D array)
    """
    frame_dim = int(np.sqrt(frame_size))
    samples_needed = n_frames * frame_size

    if len(samples) < samples_needed:
        raise ValueError(f"Need {samples_needed} samples, got {len(samples)}")

    # Accumulate frames
    accumulated = np.zeros(frame_size, dtype=np.float32)

    for frame_idx in range(n_frames):
        start = frame_idx * frame_size
        end = start + frame_size
        frame = samples[start:end]
        accumulated += frame

    # Reshape to 2D
    accumulated_2d = accumulated.reshape((frame_dim, frame_dim))

    # Extract QR (positive = white module, negative = black module)
    qr_matrix = (accumulated_2d < 0).astype(np.uint8)
    qr_content = scan_qr(qr_matrix)

    return qr_content, accumulated_2d


class AudioStreamEncoder:
    """Streaming encoder for audio QR embedding."""

    def __init__(
        self,
        qr_key: str,
        flip_probability: float = 0.4,
        frame_size: int = FRAME_SIZE,
        seed: int = 12345,
    ):
        self.qr_key = qr_key
        self.flip_probability = flip_probability
        self.frame_size = frame_size

        frame_dim = int(np.sqrt(frame_size))
        qr = generate_qr_matrix(qr_key, frame_dim)
        self.desired_flat = np.where(qr == 0, +1, -1).astype(np.float32).flatten()

        self.rng = np.random.default_rng(seed)
        self.sample_index = 0

    def encode_sample(self, sample: float) -> float:
        """Encode a single sample with sign biasing."""
        module_idx = self.sample_index % self.frame_size
        desired = self.desired_flat[module_idx]
        current_sign = np.sign(sample)

        self.sample_index += 1

        # If sample sign doesn't match desired, maybe flip it
        if current_sign != desired and current_sign != 0:
            if self.rng.random() < self.flip_probability:
                return -sample
        return sample

    def encode_chunk(self, samples: np.ndarray) -> np.ndarray:
        """Encode a chunk of samples."""
        modified = samples.copy()
        for i in range(len(modified)):
            modified[i] = self.encode_sample(modified[i])
        return modified


class AudioStreamDecoder:
    """Streaming decoder for audio QR extraction."""

    def __init__(
        self,
        n_frames: int = 60,
        frame_size: int = FRAME_SIZE,
    ):
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.frame_dim = int(np.sqrt(frame_size))

        self.accumulated = np.zeros(frame_size, dtype=np.float32)
        self.sample_count = 0
        self.frame_count = 0

    def push_sample(self, sample: float) -> tuple[str | None, np.ndarray | None]:
        """
        Push a sample. Returns result when N frames accumulated.

        Returns:
            (qr_content, accumulated_2d) when complete, else (None, None)
        """
        module_idx = self.sample_count % self.frame_size
        self.accumulated[module_idx] += sample
        self.sample_count += 1

        # Check if we completed a frame
        if self.sample_count % self.frame_size == 0:
            self.frame_count += 1

            if self.frame_count >= self.n_frames:
                # Decode
                accumulated_2d = self.accumulated.reshape((self.frame_dim, self.frame_dim))
                qr_matrix = (accumulated_2d < 0).astype(np.uint8)
                qr_content = scan_qr(qr_matrix)

                result = (qr_content, accumulated_2d.copy())

                # Reset
                self.accumulated = np.zeros(self.frame_size, dtype=np.float32)
                self.sample_count = 0
                self.frame_count = 0

                return result

        return None, None

    def push_chunk(self, samples: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Push a chunk of samples. Returns list of decoded results."""
        results = []
        for sample in samples:
            result, acc = self.push_sample(sample)
            if result is not None:
                results.append((result, acc))
        return results


def get_audio_stats(n_frames: int = 60, frame_size: int = FRAME_SIZE) -> dict:
    """Get statistics about audio encoding parameters."""
    frame_dim = int(np.sqrt(frame_size))
    samples_per_cycle = n_frames * frame_size
    duration_per_cycle = samples_per_cycle / SAMPLE_RATE

    return {
        'frame_size': frame_size,
        'frame_dim': frame_dim,
        'n_frames': n_frames,
        'samples_per_cycle': samples_per_cycle,
        'duration_per_cycle_seconds': duration_per_cycle,
        'sample_rate': SAMPLE_RATE,
    }
