"""
QR Static Stream - Signed Accumulation Variant

Instead of XOR, frames contribute +1 or -1 at each point. The sign of the
accumulated sum reveals the QR code, while the magnitude carries a hidden
payload. The QR itself contains the key to decode the magnitude data.

Two-layer steganography:
  1. Know N → accumulate frames → QR emerges from sign
  2. Scan QR → get seed → decode magnitude → hidden payload
"""

from __future__ import annotations

import numpy as np
from functools import reduce
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


def _generate_expected_accumulation(seed: str, shape: tuple[int, int], n_frames: int) -> np.ndarray:
    """
    Generate the expected accumulation if we had N frames of pure noise.

    This is what the accumulation "should" look like without any signal.
    By comparing actual vs expected, we can extract the hidden payload.
    """
    rng = _seed_to_rng(seed)

    # Generate what each frame "would have been" and accumulate
    expected = np.zeros(shape, dtype=np.int32)
    for _ in range(n_frames):
        frame = rng.choice([-1, 1], size=shape).astype(np.int8)
        expected += frame

    return expected


def encode(
    qr_seed: str,
    payload: bytes,
    n_frames: int,
    frame_shape: tuple[int, int],
    signal_strength: int = 3
) -> list[np.ndarray]:
    """
    Encode a QR code and hidden payload into N signed frames.

    Args:
        qr_seed: String to encode in QR (also seeds the PRNG for decoding)
        payload: Bytes to hide in the magnitude data
        n_frames: Number of frames to generate
        frame_shape: (height, width) of each frame
        signal_strength: How strongly to bias magnitudes (higher = more robust)

    Returns:
        List of N frames, each containing values of -1 or +1
    """
    if n_frames < 4:
        raise ValueError("Need at least 4 frames for reliable encoding")

    # Generate the QR matrix for this seed
    qr = generate_qr_matrix(qr_seed)
    qr = _pad_matrix(qr, frame_shape[0])

    # Convert QR from 0/1 to target signs: 0 (white) -> +1, 1 (black) -> -1
    target_signs = np.where(qr == 0, 1, -1).astype(np.int8)

    # Convert payload to bit array
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

    # Create a magnitude target map
    # We'll encode payload bits as magnitude deviations from expected
    # Positive deviation = 1 bit, negative deviation = 0 bit
    magnitude_target = np.zeros(frame_shape, dtype=np.int8)
    n_cells = frame_shape[0] * frame_shape[1]

    # Spread payload bits across cells (with repetition for error correction)
    if len(payload_bits) > 0:
        # Simple repetition: cycle through payload bits
        flat_target = magnitude_target.flatten()
        for i in range(n_cells):
            bit_idx = i % len(payload_bits)
            # +1 if bit is 1, -1 if bit is 0
            flat_target[i] = 1 if payload_bits[bit_idx] else -1
        magnitude_target = flat_target.reshape(frame_shape)

    # Generate frames using seeded RNG, but bias toward targets
    rng = _seed_to_rng(qr_seed)

    frames = []
    accumulated = np.zeros(frame_shape, dtype=np.int32)

    for frame_idx in range(n_frames):
        if frame_idx < n_frames - 1:
            # Generate mostly random frames, but track accumulation
            frame = rng.choice([-1, 1], size=frame_shape).astype(np.int8)
        else:
            # Final frame: force correct signs and boost magnitude toward target
            frame = np.zeros(frame_shape, dtype=np.int8)

            for y in range(frame_shape[0]):
                for x in range(frame_shape[1]):
                    current_sum = accumulated[y, x]
                    target_sign = target_signs[y, x]
                    mag_target = magnitude_target[y, x]

                    # We need final sum to have correct sign
                    # And we want magnitude to reflect payload

                    # Desired final sum: correct sign, biased magnitude
                    desired_magnitude = abs(current_sum) + signal_strength
                    if mag_target > 0:
                        desired_magnitude += signal_strength
                    else:
                        desired_magnitude -= signal_strength
                    desired_magnitude = max(1, desired_magnitude)

                    desired_sum = target_sign * desired_magnitude
                    needed = desired_sum - current_sum

                    # We can only contribute +1 or -1
                    frame[y, x] = 1 if needed > 0 else -1

        frames.append(frame)
        accumulated += frame

    return frames


def accumulate(frames: list[np.ndarray]) -> np.ndarray:
    """
    Accumulate frames via signed sum.

    Returns:
        Accumulated sum at each point (int32, can be negative)
    """
    result = np.zeros(frames[0].shape, dtype=np.int32)
    for frame in frames:
        result += frame
    return result


def extract_qr(accumulated: np.ndarray) -> np.ndarray:
    """
    Extract QR code from accumulated values via sign.

    Returns:
        Binary matrix: 0 where sum > 0 (white), 1 where sum <= 0 (black)
    """
    return (accumulated <= 0).astype(np.uint8)


def decode_payload(accumulated: np.ndarray, qr_seed: str, n_frames: int, payload_length: int) -> bytes:
    """
    Decode the hidden payload from accumulated magnitudes.

    Args:
        accumulated: The accumulated sum from all frames
        qr_seed: The seed from the QR code (for regenerating expected noise)
        n_frames: Number of frames that were accumulated
        payload_length: Expected length of payload in bytes

    Returns:
        Decoded payload bytes
    """
    # Generate what the accumulation "should" have been with pure noise
    expected = _generate_expected_accumulation(qr_seed, accumulated.shape, n_frames)

    # The residual (actual - expected) contains our signal
    residual = accumulated - expected

    # Extract bits from residual: positive = 1, negative = 0
    flat_residual = residual.flatten()
    n_cells = len(flat_residual)
    n_bits = payload_length * 8

    # Majority vote across repetitions
    bit_votes = np.zeros(n_bits, dtype=np.int32)
    vote_counts = np.zeros(n_bits, dtype=np.int32)

    for i in range(n_cells):
        bit_idx = i % n_bits
        vote = 1 if flat_residual[i] > 0 else 0
        bit_votes[bit_idx] += vote
        vote_counts[bit_idx] += 1

    # Determine bits by majority
    bits = (bit_votes > vote_counts // 2).astype(np.uint8)

    # Pack bits into bytes
    # Pad to multiple of 8
    padded_bits = np.zeros(((len(bits) + 7) // 8) * 8, dtype=np.uint8)
    padded_bits[:len(bits)] = bits

    return np.packbits(padded_bits).tobytes()[:payload_length]


class SignedStreamEncoder:
    """
    Streaming encoder for signed accumulation with payload.
    """

    def __init__(
        self,
        n_frames: int,
        frame_shape: tuple[int, int],
        signal_strength: int = 3
    ):
        self.n_frames = n_frames
        self.frame_shape = frame_shape
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
                self.signal_strength
            )
        else:
            # Pure noise when no messages queued
            rng = np.random.default_rng()
            self._current_frames = [
                rng.choice([-1, 1], size=self.frame_shape).astype(np.int8)
                for _ in range(self.n_frames)
            ]


class SignedStreamDecoder:
    """
    Streaming decoder for signed accumulation.
    """

    def __init__(self, n_frames: int, expected_payload_length: int = 32):
        self.n_frames = n_frames
        self.expected_payload_length = expected_payload_length
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
            self._accumulated = frame.astype(np.int32)
        else:
            self._accumulated = self._accumulated + frame

        if len(self._buffer) >= self.n_frames:
            # Extract QR from signs
            qr_matrix = extract_qr(self._accumulated)
            qr_content = scan_qr(qr_matrix)

            if qr_content:
                # Decode payload using QR content as seed
                payload = decode_payload(
                    self._accumulated,
                    qr_content,
                    self.n_frames,
                    self.expected_payload_length
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
