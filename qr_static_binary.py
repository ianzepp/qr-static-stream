"""
QR Static Stream - Binary Static Variant

Each frame is pure binary static (1 bit per pixel).
Grayscale emerges from temporal accumulation, not per-frame values.

Frame structure:
    Each pixel: +1 or -1 (black or white static)

Accumulator:
    Sums pixel votes over N frames → values from -N to +N
    Sign → QR pattern
    Magnitude → payload data

Encoding via probability bias:
    White QR module: P(+1) > 0.5 → trends positive
    Black QR module: P(+1) < 0.5 → trends negative
    Bias strength encodes payload magnitude
"""

from __future__ import annotations

import numpy as np
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
        matrix = _resize_qr_to_frame(matrix, size)

    return matrix


def _resize_qr_to_frame(qr_matrix: np.ndarray, target_size: int) -> np.ndarray:
    """Resize QR matrix to fill target frame size."""
    qr_size = qr_matrix.shape[0]
    scale = target_size // qr_size
    if scale < 1:
        scale = 1

    scaled = np.repeat(np.repeat(qr_matrix, scale, axis=0), scale, axis=1)

    scaled_size = scaled.shape[0]
    if scaled_size < target_size:
        pad_total = target_size - scaled_size
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        scaled = np.pad(scaled, ((pad_before, pad_after), (pad_before, pad_after)),
                       mode='constant', constant_values=0)
    elif scaled_size > target_size:
        crop_start = (scaled_size - target_size) // 2
        scaled = scaled[crop_start:crop_start + target_size,
                       crop_start:crop_start + target_size]

    return scaled


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


def _frame_rng(seed: str, frame_index: int) -> np.random.Generator:
    """Get deterministic RNG for a specific frame."""
    combined = f"{seed}:{frame_index}"
    hash_bytes = hashlib.sha256(combined.encode()).digest()
    seed_int = struct.unpack('<Q', hash_bytes[:8])[0]
    return np.random.default_rng(seed_int)


def encode_binary(
    qr_key: str,
    frame_shape: tuple[int, int],
    n_frames: int,
    base_bias: float = 0.6,
    payload: bytes = None,
    payload_bias_delta: float = 0.1,
) -> list[np.ndarray]:
    """
    Encode QR pattern into binary static frames.

    Args:
        qr_key: String to encode in QR code
        frame_shape: (height, width) of frames
        n_frames: Number of frames to generate
        base_bias: Base probability of +1 for white modules (0.5-1.0)
        payload: Optional bytes to encode in magnitude
        payload_bias_delta: How much to adjust bias for payload bits

    Returns:
        List of binary frames (int8 with values +1/-1)
    """
    qr = generate_qr_matrix(qr_key, frame_shape[0])

    # Base bias: white modules (qr=0) get P(+1) = base_bias
    #            black modules (qr=1) get P(+1) = 1 - base_bias
    bias_map = np.where(qr == 0, base_bias, 1.0 - base_bias).astype(np.float32)

    # Modulate bias with payload
    if payload is not None:
        payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        n_pixels = frame_shape[0] * frame_shape[1]

        # Create payload modulation map
        flat_bias = bias_map.flatten()
        for i in range(n_pixels):
            bit_idx = i % len(payload_bits)
            # bit=1: increase bias magnitude (stronger signal)
            # bit=0: decrease bias magnitude (weaker signal)
            if payload_bits[bit_idx]:
                # Move bias further from 0.5
                if flat_bias[i] > 0.5:
                    flat_bias[i] = min(0.95, flat_bias[i] + payload_bias_delta)
                else:
                    flat_bias[i] = max(0.05, flat_bias[i] - payload_bias_delta)
            else:
                # Move bias closer to 0.5
                if flat_bias[i] > 0.5:
                    flat_bias[i] = max(0.5, flat_bias[i] - payload_bias_delta)
                else:
                    flat_bias[i] = min(0.5, flat_bias[i] + payload_bias_delta)

        bias_map = flat_bias.reshape(frame_shape)

    # Generate binary frames
    frames = []
    for i in range(n_frames):
        rng = _frame_rng(qr_key, i)
        random_vals = rng.random(frame_shape).astype(np.float32)

        # +1 where random < bias, -1 otherwise
        frame = np.where(random_vals < bias_map, 1, -1).astype(np.int8)
        frames.append(frame)

    return frames


def accumulate(frames: list[np.ndarray]) -> np.ndarray:
    """
    Accumulate binary frames into int16 sum.

    Args:
        frames: List of binary frames (+1/-1 values)

    Returns:
        Accumulated sum as int16 array
    """
    result = np.zeros(frames[0].shape, dtype=np.int16)
    for frame in frames:
        result += frame.astype(np.int16)
    return result


def extract_qr(accumulated: np.ndarray) -> np.ndarray:
    """Extract QR pattern from accumulated values (sign)."""
    return (accumulated < 0).astype(np.uint8)


def decode_payload(
    accumulated: np.ndarray,
    qr_key: str,
    n_frames: int,
    payload_length: int,
    base_bias: float = 0.6,
) -> bytes:
    """
    Decode payload from accumulated magnitude.

    Args:
        accumulated: Accumulated sum array
        qr_key: QR key (to determine expected sign pattern)
        n_frames: Number of frames that were accumulated
        payload_length: Expected payload length in bytes
        base_bias: Base bias used during encoding

    Returns:
        Decoded payload bytes
    """
    qr = generate_qr_matrix(qr_key, accumulated.shape[0])

    # Expected magnitude for base_bias with no payload modulation
    # E[sum] = n_frames * (2 * base_bias - 1) for white modules
    expected_magnitude = n_frames * (2 * base_bias - 1)

    # Get actual magnitudes (make all positive by multiplying by expected sign)
    expected_signs = np.where(qr == 0, 1, -1)
    magnitudes = accumulated * expected_signs

    # Threshold: values above expected = bit 1, below = bit 0
    n_bits = payload_length * 8
    flat_magnitudes = magnitudes.flatten()
    n_pixels = len(flat_magnitudes)

    bit_votes = np.zeros(n_bits, dtype=np.float32)
    vote_counts = np.zeros(n_bits, dtype=np.int32)

    for i in range(n_pixels):
        bit_idx = i % n_bits
        vote = 1.0 if flat_magnitudes[i] > expected_magnitude else 0.0
        bit_votes[bit_idx] += vote
        vote_counts[bit_idx] += 1

    # Majority vote
    bits = (bit_votes > vote_counts / 2).astype(np.uint8)

    # Pack bits into bytes
    padded_bits = np.zeros(((len(bits) + 7) // 8) * 8, dtype=np.uint8)
    padded_bits[:len(bits)] = bits
    payload = np.packbits(padded_bits).tobytes()[:payload_length]

    return payload


class BinaryStreamEncoder:
    """Streaming encoder for binary static."""

    def __init__(
        self,
        frame_shape: tuple[int, int],
        qr_key: str,
        n_frames: int = 60,
        base_bias: float = 0.6,
    ):
        self.frame_shape = frame_shape
        self.qr_key = qr_key
        self.n_frames = n_frames
        self.base_bias = base_bias

        qr = generate_qr_matrix(qr_key, frame_shape[0])
        self.bias_map = np.where(qr == 0, base_bias, 1.0 - base_bias).astype(np.float32)

        self.frame_index = 0

    def next_frame(self) -> np.ndarray:
        """Generate next binary static frame."""
        rng = _frame_rng(self.qr_key, self.frame_index)
        random_vals = rng.random(self.frame_shape).astype(np.float32)
        frame = np.where(random_vals < self.bias_map, 1, -1).astype(np.int8)
        self.frame_index += 1
        return frame


class BinaryStreamDecoder:
    """Streaming decoder for binary static."""

    def __init__(self, n_frames: int = 60):
        self.n_frames = n_frames
        self.accumulated: np.ndarray | None = None
        self.frame_count = 0

    def push_frame(self, frame: np.ndarray) -> tuple[str | None, np.ndarray | None]:
        """
        Push a frame. Returns result when N frames accumulated.

        Returns:
            (qr_content, accumulated) when complete, else (None, None)
        """
        if self.accumulated is None:
            self.accumulated = np.zeros(frame.shape, dtype=np.int16)

        self.accumulated += frame.astype(np.int16)
        self.frame_count += 1

        if self.frame_count >= self.n_frames:
            qr_matrix = extract_qr(self.accumulated)
            qr_content = scan_qr(qr_matrix)
            result = (qr_content, self.accumulated.copy())

            # Reset for next window
            self.accumulated = None
            self.frame_count = 0

            return result

        return None, None
