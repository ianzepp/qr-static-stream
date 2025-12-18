"""
QR Static Stream - Analog/Continuous Variant

Frames contain continuous grayscale values (floats), not binary ±1.
The accumulated sum is continuous. QR emerges from thresholding at zero.
Magnitude carries payload. Grayscale precision increases information density.

Three-layer steganography:
  1. Know N → accumulate frames → continuous height field
  2. Threshold at 0 → QR emerges from sign
  3. Scan QR → get key → decode magnitude → hidden payload
"""

from __future__ import annotations

import numpy as np
from typing import Optional
import hashlib
import struct


def generate_qr_matrix(data: str, size: int = None) -> np.ndarray:
    """Generate a binary matrix from QR code data."""
    import qrcode

    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=1,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    modules = qr.get_matrix()
    matrix = np.array(modules, dtype=np.uint8)

    if size and matrix.shape[0] != size:
        matrix = _pad_matrix(matrix, size)

    return matrix


def _pad_matrix(matrix: np.ndarray, target_size: int) -> np.ndarray:
    """Pad matrix to target size with zeros (white)."""
    from_size = matrix.shape[0]

    if target_size == from_size:
        return matrix

    if target_size > from_size:
        pad_total = target_size - from_size
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(matrix, ((pad_before, pad_after), (pad_before, pad_after)),
                      mode='constant', constant_values=0)

    crop_total = from_size - target_size
    crop_start = crop_total // 2
    return matrix[crop_start:crop_start + target_size, crop_start:crop_start + target_size]


def scan_qr(matrix: np.ndarray) -> str | None:
    """Scan a QR code from a binary matrix."""
    import cv2

    img = ((1 - matrix) * 255).astype(np.uint8)

    min_size = 100
    if img.shape[0] < min_size:
        scale = (min_size // img.shape[0]) + 1
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)

    return data if data else None


def _seed_to_rng(seed: str) -> np.random.Generator:
    """Convert a string seed to a numpy RNG."""
    hash_bytes = hashlib.sha256(seed.encode()).digest()
    seed_int = struct.unpack('<Q', hash_bytes[:8])[0]
    return np.random.default_rng(seed_int)


def encode(
    qr_seed: str,
    payload: bytes,
    n_frames: int,
    frame_shape: tuple[int, int],
    noise_amplitude: float = 0.3,
    signal_strength: float = 5.0,
) -> list[np.ndarray]:
    """
    Encode a QR code and hidden payload into N continuous-valued frames.

    Each frame contains grayscale noise with a tiny embedded signal.
    After N frames, signal accumulates while noise partially cancels.

    Args:
        qr_seed: String to encode in QR (also seeds PRNG for decoding)
        payload: Bytes to hide in magnitude data
        n_frames: Number of frames to generate
        frame_shape: (height, width) of each frame
        noise_amplitude: Amplitude of per-frame noise (visual randomness)
        signal_strength: Target magnitude for encoded bits (larger = more robust)

    Returns:
        List of N frames, each float32 in roughly [-1, 1] range
    """
    if n_frames < 2:
        raise ValueError("Need at least 2 frames")

    # Generate QR matrix and pad to frame size
    qr = generate_qr_matrix(qr_seed)
    qr = _pad_matrix(qr, frame_shape[0])

    # Target signs: 0 (white) -> +1, 1 (black) -> -1
    target_signs = np.where(qr == 0, 1.0, -1.0).astype(np.float32)

    # Convert payload to bits and create magnitude targets
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    n_cells = frame_shape[0] * frame_shape[1]

    # Magnitude encoding: bit 1 -> higher magnitude, bit 0 -> lower magnitude
    magnitude_bias = np.zeros(frame_shape, dtype=np.float32)
    if len(payload_bits) > 0:
        flat_bias = magnitude_bias.flatten()
        for i in range(n_cells):
            bit_idx = i % len(payload_bits)
            # Bit 1 -> +0.5 bias, Bit 0 -> -0.5 bias (relative to base magnitude)
            flat_bias[i] = 0.5 if payload_bits[bit_idx] else -0.5
        magnitude_bias = flat_bias.reshape(frame_shape)

    # Target sum at each cell: sign * (base_magnitude + payload_bias)
    target_sum = target_signs * (signal_strength + magnitude_bias)

    # Generate frames: each frame = (target / N) + noise
    # Signal accumulates to target, noise partially cancels (sqrt(N) growth vs N growth)
    rng = _seed_to_rng(qr_seed)
    signal_per_frame = target_sum / n_frames

    frames = []
    for _ in range(n_frames):
        noise = rng.uniform(-noise_amplitude, noise_amplitude, size=frame_shape).astype(np.float32)
        frame = signal_per_frame + noise
        frames.append(frame)

    return frames


def accumulate(frames: list[np.ndarray]) -> np.ndarray:
    """
    Accumulate frames via summation.

    Returns:
        Accumulated sum at each point (float32)
    """
    result = np.zeros(frames[0].shape, dtype=np.float32)
    for frame in frames:
        result += frame
    return result


def extract_qr(accumulated: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Extract QR code from accumulated values via threshold.

    Returns:
        Binary matrix: 0 where sum > threshold (white), 1 where sum <= threshold (black)
    """
    return (accumulated <= threshold).astype(np.uint8)


def decode_payload(
    accumulated: np.ndarray,
    qr_seed: str,
    n_frames: int,
    payload_length: int,
    noise_amplitude: float = 0.3,
    signal_strength: float = 5.0,
) -> bytes:
    """
    Decode hidden payload from accumulated magnitudes.

    Args:
        accumulated: The accumulated sum from all frames
        qr_seed: The seed from the QR code
        n_frames: Number of frames that were accumulated
        payload_length: Expected payload length in bytes
        noise_amplitude: Same noise amplitude used during encoding

    Returns:
        Decoded payload bytes
    """
    # Get the QR pattern to know expected signs
    qr = generate_qr_matrix(qr_seed)
    qr = _pad_matrix(qr, accumulated.shape[0])
    expected_signs = np.where(qr == 0, 1.0, -1.0)

    # Magnitude is |accumulated|, but we need to account for sign
    # The payload is encoded in whether magnitude is above or below average
    magnitudes = np.abs(accumulated)

    # Expected base magnitude (without payload bias)
    # With signal_strength=2.0 default, base is around 2.0
    # Noise contribution: N frames of uniform[-0.5, 0.5] sums to roughly 0 with std sqrt(N * 1/12)
    # So magnitude should be close to signal_strength

    # Compute expected magnitude from the reconstruction
    rng = _seed_to_rng(qr_seed)
    expected_noise_sum = np.zeros(accumulated.shape, dtype=np.float32)
    for _ in range(n_frames):
        noise = rng.uniform(-noise_amplitude, noise_amplitude, size=accumulated.shape).astype(np.float32)
        expected_noise_sum += noise

    # The accumulated signal without payload would be: expected_signs * signal_strength + expected_noise_sum
    # With payload: expected_signs * (signal_strength + bias) + expected_noise_sum
    # So: accumulated - expected_noise_sum = expected_signs * (signal_strength + bias)
    # And: |accumulated - expected_noise_sum| / |expected_signs| = signal_strength + bias
    # Since |expected_signs| = 1: magnitude_cleaned = |accumulated - expected_noise_sum|

    cleaned = accumulated - expected_noise_sum
    magnitude_cleaned = np.abs(cleaned)

    # The bias is relative to signal_strength
    # bit 1 -> magnitude ~signal_strength+0.5, bit 0 -> magnitude ~signal_strength-0.5
    # Threshold at signal_strength
    threshold = signal_strength

    flat_magnitudes = magnitude_cleaned.flatten()
    n_cells = len(flat_magnitudes)
    n_bits = payload_length * 8

    # Majority vote across repetitions
    bit_votes = np.zeros(n_bits, dtype=np.float32)
    vote_counts = np.zeros(n_bits, dtype=np.int32)

    for i in range(n_cells):
        bit_idx = i % n_bits
        # Higher magnitude -> bit 1, lower -> bit 0
        vote = 1.0 if flat_magnitudes[i] > threshold else 0.0
        bit_votes[bit_idx] += vote
        vote_counts[bit_idx] += 1

    # Determine bits by majority
    bits = (bit_votes > vote_counts / 2).astype(np.uint8)

    # Pack into bytes
    padded_bits = np.zeros(((len(bits) + 7) // 8) * 8, dtype=np.uint8)
    padded_bits[:len(bits)] = bits

    return np.packbits(padded_bits).tobytes()[:payload_length]


def frames_to_video_frames(frames: list[np.ndarray], scale: int = 4) -> list[np.ndarray]:
    """
    Convert internal float frames to uint8 video frames.

    Args:
        frames: List of float32 frames
        scale: Upscale factor for visibility

    Returns:
        List of uint8 frames suitable for video encoding
    """
    video_frames = []
    for frame in frames:
        # Map from typical range [-1.5, 1.5] to [0, 255]
        normalized = (frame + 1.5) / 3.0  # Map to [0, 1]
        normalized = np.clip(normalized, 0, 1)
        uint8_frame = (normalized * 255).astype(np.uint8)

        # Upscale for visibility
        if scale > 1:
            uint8_frame = np.repeat(np.repeat(uint8_frame, scale, axis=0), scale, axis=1)

        video_frames.append(uint8_frame)

    return video_frames


def height_field_to_image(accumulated: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Convert accumulated height field to a viewable grayscale image.

    Args:
        accumulated: Float32 accumulated values
        scale: Upscale factor

    Returns:
        uint8 grayscale image
    """
    # Normalize to [0, 255]
    vmin, vmax = accumulated.min(), accumulated.max()
    if vmax > vmin:
        normalized = (accumulated - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(accumulated)

    img = (normalized * 255).astype(np.uint8)

    if scale > 1:
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

    return img


class AnalogStreamEncoder:
    """Streaming encoder for analog/continuous frames."""

    def __init__(
        self,
        n_frames: int,
        frame_shape: tuple[int, int],
        noise_amplitude: float = 0.3,
        signal_strength: float = 5.0,
    ):
        self.n_frames = n_frames
        self.frame_shape = frame_shape
        self.noise_amplitude = noise_amplitude
        self.signal_strength = signal_strength
        self._queue: list[tuple[str, bytes]] = []
        self._current_frames: list[np.ndarray] = []
        self._frame_index = 0

    def queue_message(self, qr_seed: str, payload: bytes):
        """Queue a message (QR seed + hidden payload)."""
        self._queue.append((qr_seed, payload))

    def next_frame(self) -> np.ndarray:
        """Get the next frame."""
        if self._frame_index >= len(self._current_frames):
            self._start_new_cycle()

        frame = self._current_frames[self._frame_index]
        self._frame_index += 1
        return frame

    def _start_new_cycle(self):
        """Begin a new cycle."""
        self._frame_index = 0

        if self._queue:
            qr_seed, payload = self._queue.pop(0)
            self._current_frames = encode(
                qr_seed,
                payload,
                self.n_frames,
                self.frame_shape,
                self.noise_amplitude,
                self.signal_strength,
            )
        else:
            # Pure noise when no messages queued
            rng = np.random.default_rng()
            self._current_frames = [
                rng.uniform(-self.noise_amplitude, self.noise_amplitude,
                           size=self.frame_shape).astype(np.float32)
                for _ in range(self.n_frames)
            ]


class AnalogStreamDecoder:
    """Streaming decoder for analog/continuous frames."""

    def __init__(
        self,
        n_frames: int,
        expected_payload_length: int = 32,
        noise_amplitude: float = 0.3,
        signal_strength: float = 5.0,
    ):
        self.n_frames = n_frames
        self.expected_payload_length = expected_payload_length
        self.noise_amplitude = noise_amplitude
        self.signal_strength = signal_strength
        self._buffer: list[np.ndarray] = []
        self._accumulated: np.ndarray | None = None

    def push_frame(self, frame: np.ndarray) -> tuple[str, bytes] | None:
        """
        Push a frame and attempt decode.

        Returns:
            (qr_content, payload) if cycle complete and valid, else None
        """
        self._buffer.append(frame)

        if self._accumulated is None:
            self._accumulated = frame.astype(np.float32).copy()
        else:
            self._accumulated = self._accumulated + frame

        if len(self._buffer) >= self.n_frames:
            # Extract QR from threshold
            qr_matrix = extract_qr(self._accumulated)
            qr_content = scan_qr(qr_matrix)

            if qr_content:
                payload = decode_payload(
                    self._accumulated,
                    qr_content,
                    self.n_frames,
                    self.expected_payload_length,
                    self.noise_amplitude,
                    self.signal_strength,
                )
                self._reset()
                return (qr_content, payload)

            self._reset()

        return None

    def _reset(self):
        """Reset for next cycle."""
        self._buffer = []
        self._accumulated = None

    def peek_accumulation(self) -> np.ndarray | None:
        """Get current accumulated state."""
        return self._accumulated.copy() if self._accumulated is not None else None
