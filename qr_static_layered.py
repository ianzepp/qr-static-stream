"""
QR Static Stream - Two-Layer Steganography

Layer 1: Accumulate N₁ carrier frames → QR₁ emerges from sign
Layer 2: Accumulate magnitude deviations across N₂ Layer 1 outputs → QR₂ + payload

The carrier video looks like static. Accumulate N₁ frames to see QR₁.
The Layer 1 magnitude varies slightly around a baseline. Accumulate those
variations across N₂ Layer 1 outputs and QR₂ emerges, with payload in depth.

Four secrets to unlock:
  1. N₁ (Layer 1 frame count)
  2. N₂ (Layer 2 frame count)
  3. QR₁ content (Layer 1 key - currently just validates structure)
  4. QR₂ content (Layer 2 key - used to decode payload)
"""

from __future__ import annotations

import numpy as np
import hashlib
import struct
from typing import Optional


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


def encode_two_layer(
    layer1_qr_key: str,
    layer2_qr_key: str,
    layer2_payload: bytes,
    frame_shape: tuple[int, int],
    n1: int,
    n2: int,
    layer1_signal: float = 5.0,
    layer1_noise: float = 0.2,
    layer2_signal: float = 2.0,
    layer2_noise: float = 0.1,
) -> list[np.ndarray]:
    """
    Encode two layers of steganography into carrier frames.

    Args:
        layer1_qr_key: Key/content for Layer 1 QR code
        layer2_qr_key: Key/content for Layer 2 QR code (used to decode payload)
        layer2_payload: Bytes to hide in Layer 2 magnitude
        frame_shape: (height, width) of carrier frames
        n1: Number of carrier frames per Layer 1 output
        n2: Number of Layer 1 outputs per Layer 2 output
        layer1_signal: Base magnitude for Layer 1 outputs
        layer1_noise: Per-frame noise amplitude for Layer 1
        layer2_signal: Signal strength for Layer 2 encoding
        layer2_noise: Noise amplitude for Layer 2 contributions

    Returns:
        List of n1 * n2 carrier frames (float32)
    """
    # Generate QR patterns
    qr1 = generate_qr_matrix(layer1_qr_key)
    qr1 = _pad_matrix(qr1, frame_shape[0])
    qr1_signs = np.where(qr1 == 0, 1.0, -1.0).astype(np.float32)

    qr2 = generate_qr_matrix(layer2_qr_key)
    qr2 = _pad_matrix(qr2, frame_shape[0])
    qr2_signs = np.where(qr2 == 0, 1.0, -1.0).astype(np.float32)

    # Encode Layer 2 payload into magnitude bias
    payload_bits = np.unpackbits(np.frombuffer(layer2_payload, dtype=np.uint8))
    n_cells = frame_shape[0] * frame_shape[1]

    magnitude_bias = np.zeros(frame_shape, dtype=np.float32)
    if len(payload_bits) > 0:
        flat_bias = magnitude_bias.flatten()
        for i in range(n_cells):
            bit_idx = i % len(payload_bits)
            flat_bias[i] = 0.5 if payload_bits[bit_idx] else -0.5
        magnitude_bias = flat_bias.reshape(frame_shape)

    # Target Layer 2 accumulated deviation (what magnitude deviations should sum to)
    target_layer2_deviation = qr2_signs * (layer2_signal + magnitude_bias)

    # Generate carrier frames
    all_frames = []
    rng_l2 = _seed_to_rng(layer2_qr_key)
    rng_l1 = _seed_to_rng(layer1_qr_key)

    # For each Layer 1 output (N2 of them)
    for l2_idx in range(n2):
        # This Layer 1 output's contribution to Layer 2 signal
        l2_noise = rng_l2.uniform(-layer2_noise, layer2_noise, size=frame_shape).astype(np.float32)
        deviation_contribution = target_layer2_deviation / n2 + l2_noise

        # Target magnitude for this Layer 1 output (base + deviation)
        target_l1_magnitude = layer1_signal + deviation_contribution

        # Target sum for this Layer 1 output: sign from QR1, magnitude from above
        target_l1_sum = qr1_signs * target_l1_magnitude

        # Encode this Layer 1 target into N1 carrier frames
        signal_per_frame = target_l1_sum / n1

        for l1_idx in range(n1):
            noise = rng_l1.uniform(-layer1_noise, layer1_noise, size=frame_shape).astype(np.float32)
            frame = signal_per_frame + noise
            all_frames.append(frame)

    return all_frames


def decode_layer1(frames: list[np.ndarray], n1: int) -> list[np.ndarray]:
    """
    Decode Layer 1 by accumulating groups of n1 frames.

    Args:
        frames: All carrier frames
        n1: Frames per Layer 1 output

    Returns:
        List of Layer 1 accumulated outputs
    """
    layer1_outputs = []

    for i in range(0, len(frames), n1):
        group = frames[i:i + n1]
        if len(group) == n1:
            accumulated = np.sum(group, axis=0)
            layer1_outputs.append(accumulated)

    return layer1_outputs


def extract_qr_from_accumulated(accumulated: np.ndarray) -> np.ndarray:
    """Extract QR matrix from accumulated values via sign threshold."""
    return (accumulated <= 0).astype(np.uint8)


def decode_layer2(
    layer1_outputs: list[np.ndarray],
    layer1_qr_key: str,
    n1: int,
    layer1_signal: float = 5.0,
    layer1_noise: float = 0.2,
) -> np.ndarray:
    """
    Decode Layer 2 by extracting magnitude deviations from Layer 1 outputs
    and accumulating them.

    Args:
        layer1_outputs: List of Layer 1 accumulated frames
        layer1_qr_key: Key used for Layer 1 (to compute expected magnitudes)
        n1: Frames per Layer 1 output (for expected magnitude calculation)
        layer1_signal: Base signal strength used in encoding
        layer1_noise: Noise amplitude used in encoding (for noise cancellation)

    Returns:
        Layer 2 accumulated deviation field
    """
    if not layer1_outputs:
        raise ValueError("No Layer 1 outputs provided")

    frame_shape = layer1_outputs[0].shape
    n2 = len(layer1_outputs)

    # Get QR1 pattern to know expected signs
    qr1 = generate_qr_matrix(layer1_qr_key)
    qr1 = _pad_matrix(qr1, frame_shape[0])
    expected_signs = np.where(qr1 == 0, 1.0, -1.0)

    # Reconstruct expected L1 noise contributions
    rng_l1 = _seed_to_rng(layer1_qr_key)

    # Accumulate magnitude deviations with noise cancellation
    layer2_accumulated = np.zeros(frame_shape, dtype=np.float32)

    for l1_idx, l1_output in enumerate(layer1_outputs):
        # Compute expected noise sum for this L1 output
        expected_noise_sum = np.zeros(frame_shape, dtype=np.float32)
        for _ in range(n1):
            noise = rng_l1.uniform(-layer1_noise, layer1_noise, size=frame_shape).astype(np.float32)
            expected_noise_sum += noise

        # Remove expected noise from L1 output
        cleaned_l1 = l1_output - expected_noise_sum

        # Get actual magnitude (accounting for sign)
        actual_magnitude = cleaned_l1 * expected_signs

        # Deviation from expected base
        deviation = actual_magnitude - layer1_signal

        layer2_accumulated += deviation

    return layer2_accumulated


def decode_layer2_payload(
    layer2_accumulated: np.ndarray,
    layer2_qr_key: str,
    n2: int,
    payload_length: int,
    layer2_signal: float = 3.0,
    layer2_noise: float = 0.2,
) -> bytes:
    """
    Decode payload from Layer 2 accumulated magnitudes.

    Args:
        layer2_accumulated: Accumulated Layer 2 deviation field
        layer2_qr_key: Key from Layer 2 QR
        n2: Number of Layer 1 outputs that were accumulated
        payload_length: Expected payload length in bytes
        layer2_signal: Signal strength used in encoding
        layer2_noise: Noise amplitude used in encoding

    Returns:
        Decoded payload bytes
    """
    frame_shape = layer2_accumulated.shape

    # Reconstruct expected noise contribution
    rng_l2 = _seed_to_rng(layer2_qr_key)
    expected_noise_sum = np.zeros(frame_shape, dtype=np.float32)
    for _ in range(n2):
        noise = rng_l2.uniform(-layer2_noise, layer2_noise, size=frame_shape).astype(np.float32)
        expected_noise_sum += noise

    # Remove expected noise
    cleaned = layer2_accumulated - expected_noise_sum

    # Get QR2 pattern for expected signs
    qr2 = generate_qr_matrix(layer2_qr_key)
    qr2 = _pad_matrix(qr2, frame_shape[0])
    expected_signs = np.where(qr2 == 0, 1.0, -1.0)

    # Extract magnitude (multiply by expected sign to get positive magnitude)
    magnitude = cleaned * expected_signs

    # Threshold at base signal strength
    threshold = layer2_signal

    # Decode bits via majority vote
    flat_magnitude = magnitude.flatten()
    n_bits = payload_length * 8
    n_cells = len(flat_magnitude)

    bit_votes = np.zeros(n_bits, dtype=np.float32)
    vote_counts = np.zeros(n_bits, dtype=np.int32)

    for i in range(n_cells):
        bit_idx = i % n_bits
        vote = 1.0 if flat_magnitude[i] > threshold else 0.0
        bit_votes[bit_idx] += vote
        vote_counts[bit_idx] += 1

    bits = (bit_votes > vote_counts / 2).astype(np.uint8)

    padded_bits = np.zeros(((len(bits) + 7) // 8) * 8, dtype=np.uint8)
    padded_bits[:len(bits)] = bits

    return np.packbits(padded_bits).tobytes()[:payload_length]


def encode(
    layer1_key: str,
    layer2_key: str,
    payload: bytes,
    frame_shape: tuple[int, int],
    n1: int = 30,
    n2: int = 30,
) -> list[np.ndarray]:
    """
    Convenience function for two-layer encoding.

    Args:
        layer1_key: Visible QR content for Layer 1
        layer2_key: Hidden QR content for Layer 2 (decryption key)
        payload: Bytes to hide in Layer 2
        frame_shape: (height, width) of carrier frames
        n1: Carrier frames per Layer 1 output (default 30 for 30fps = 1 sec)
        n2: Layer 1 outputs per Layer 2 output (default 30 = 30 sec)

    Returns:
        List of n1 * n2 carrier frames
    """
    return encode_two_layer(
        layer1_qr_key=layer1_key,
        layer2_qr_key=layer2_key,
        layer2_payload=payload,
        frame_shape=frame_shape,
        n1=n1,
        n2=n2,
    )


def decode(
    frames: list[np.ndarray],
    n1: int = 30,
    n2: int = 30,
    payload_length: int = None,
    layer1_signal: float = 5.0,
    layer1_noise: float = 0.2,
    layer2_signal: float = 2.0,
    layer2_noise: float = 0.1,
) -> tuple[str | None, str | None, bytes | None]:
    """
    Convenience function for two-layer decoding.

    Args:
        frames: All carrier frames
        n1: Carrier frames per Layer 1 output
        n2: Layer 1 outputs per Layer 2 output
        payload_length: Expected payload length (required for payload decoding)
        layer1_signal: Base signal strength for Layer 1
        layer1_noise: Noise amplitude for Layer 1 (for noise cancellation)
        layer2_signal: Signal strength for Layer 2
        layer2_noise: Noise amplitude for Layer 2

    Returns:
        (layer1_qr_content, layer2_qr_content, payload)
        Any may be None if decoding fails at that layer
    """
    # Decode Layer 1
    layer1_outputs = decode_layer1(frames, n1)

    if not layer1_outputs:
        return None, None, None

    # Extract QR1 from first Layer 1 output
    qr1_matrix = extract_qr_from_accumulated(layer1_outputs[0])
    layer1_qr = scan_qr(qr1_matrix)

    if not layer1_qr:
        return None, None, None

    # Decode Layer 2
    layer2_accumulated = decode_layer2(
        layer1_outputs, layer1_qr, n1,
        layer1_signal=layer1_signal,
        layer1_noise=layer1_noise,
    )

    # Extract QR2
    qr2_matrix = extract_qr_from_accumulated(layer2_accumulated)
    layer2_qr = scan_qr(qr2_matrix)

    if not layer2_qr:
        return layer1_qr, None, None

    # Decode payload if length provided
    payload = None
    if payload_length:
        payload = decode_layer2_payload(
            layer2_accumulated,
            layer2_qr,
            n2,
            payload_length,
            layer2_signal=layer2_signal,
            layer2_noise=layer2_noise,
        )

    return layer1_qr, layer2_qr, payload


class TwoLayerEncoder:
    """Streaming encoder for two-layer steganography."""

    def __init__(
        self,
        frame_shape: tuple[int, int],
        n1: int = 30,
        n2: int = 30,
    ):
        self.frame_shape = frame_shape
        self.n1 = n1
        self.n2 = n2
        self._frames: list[np.ndarray] = []
        self._frame_idx = 0

    def set_message(
        self,
        layer1_key: str,
        layer2_key: str,
        payload: bytes,
    ):
        """Set the message to encode. Call before getting frames."""
        self._frames = encode(
            layer1_key,
            layer2_key,
            payload,
            self.frame_shape,
            self.n1,
            self.n2,
        )
        self._frame_idx = 0

    def next_frame(self) -> np.ndarray | None:
        """Get next carrier frame, or None if complete."""
        if self._frame_idx >= len(self._frames):
            return None

        frame = self._frames[self._frame_idx]
        self._frame_idx += 1
        return frame

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    @property
    def remaining_frames(self) -> int:
        return len(self._frames) - self._frame_idx


class TwoLayerDecoder:
    """Streaming decoder for two-layer steganography."""

    def __init__(
        self,
        n1: int = 30,
        n2: int = 30,
        expected_payload_length: int = None,
    ):
        self.n1 = n1
        self.n2 = n2
        self.expected_payload_length = expected_payload_length
        self._frames: list[np.ndarray] = []

    def push_frame(self, frame: np.ndarray) -> tuple[str, str, bytes] | None:
        """
        Push a carrier frame. Returns result when complete.

        Returns:
            (layer1_qr, layer2_qr, payload) when n1*n2 frames accumulated,
            None otherwise
        """
        self._frames.append(frame)

        expected_total = self.n1 * self.n2
        if len(self._frames) >= expected_total:
            result = decode(
                self._frames[:expected_total],
                self.n1,
                self.n2,
                self.expected_payload_length,
            )
            self._frames = self._frames[expected_total:]
            return result

        return None

    def reset(self):
        """Clear accumulated frames."""
        self._frames = []
