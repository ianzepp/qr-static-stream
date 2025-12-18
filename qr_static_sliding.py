"""
QR Static Stream - Sliding Window Variant

Layer 1 uses overlapping windows for smooth carrier video:
- Each frame contributes to multiple L1 outputs
- No detectable boundaries in the carrier
- Decoder can lock on at any stride boundary

Layer 2 (optional) overlays on top:
- L1 and L2 signals are additive and composable
- Same L1 stream can carry different L2 payloads
- L2 uses discrete windows (no need for sliding at this layer)

Frame structure:
    final_frame = L1_signal + L2_shift + noise

Where:
    L1_signal = (QR1_signs * L1_magnitude) / N1
    L2_shift = (QR2_signs * L2_magnitude) / (N1 * N2)  # tiny per-frame
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
        matrix = _resize_qr_to_frame(matrix, size)

    return matrix


def _resize_qr_to_frame(qr_matrix: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize QR matrix to fill target frame size.

    Uses nearest-neighbor scaling to maintain sharp module boundaries.
    Each QR module becomes multiple pixels.
    """
    qr_size = qr_matrix.shape[0]

    # Calculate scale factor (pixels per QR module)
    scale = target_size // qr_size
    if scale < 1:
        scale = 1

    # Scale up using repeat
    scaled = np.repeat(np.repeat(qr_matrix, scale, axis=0), scale, axis=1)

    # Center in target size if needed
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


def _frame_index_rng(seed: str, frame_index: int) -> np.random.Generator:
    """Get deterministic RNG for a specific frame index."""
    combined = f"{seed}:{frame_index}"
    hash_bytes = hashlib.sha256(combined.encode()).digest()
    seed_int = struct.unpack('<Q', hash_bytes[:8])[0]
    return np.random.default_rng(seed_int)


def get_qr_module_size(frame_size: int, qr_key: str) -> int:
    """Calculate how many pixels per QR module for a given frame size."""
    qr = generate_qr_matrix(qr_key)
    qr_size = qr.shape[0]
    return frame_size // qr_size


def encode_l1_sliding(
    qr_key: str,
    frame_shape: tuple[int, int],
    total_frames: int,
    n1: int = 60,
    stride: int = 30,
    signal_strength: float = 5.0,
    noise_amplitude: float = 0.2,
) -> list[np.ndarray]:
    """
    Encode Layer 1 using sliding window.

    Each frame contributes 1/N1 of the QR signal plus noise.
    Any N1 consecutive frames accumulate to the full QR.
    Overlapping windows create smooth transitions.

    Args:
        qr_key: String to encode in QR code
        frame_shape: (height, width) of frames
        total_frames: Number of frames to generate
        n1: Window size (frames per L1 output)
        stride: Window stride (frames between L1 outputs)
        signal_strength: Target magnitude after N1 accumulation
        noise_amplitude: Per-frame noise amplitude

    Returns:
        List of carrier frames
    """
    # Generate QR pattern scaled to frame size
    qr = generate_qr_matrix(qr_key, frame_shape[0])

    # Target signs: 0 (white) -> +1, 1 (black) -> -1
    qr_signs = np.where(qr == 0, 1.0, -1.0).astype(np.float32)

    # Signal per frame (accumulates to signal_strength over N1 frames)
    signal_per_frame = (qr_signs * signal_strength) / n1

    frames = []
    for i in range(total_frames):
        rng = _frame_index_rng(qr_key, i)
        noise = rng.uniform(-noise_amplitude, noise_amplitude,
                           size=frame_shape).astype(np.float32)
        frame = signal_per_frame + noise
        frames.append(frame)

    return frames


def apply_l2_overlay(
    l1_frames: list[np.ndarray],
    qr_key: str,
    payload: bytes,
    n1: int = 60,
    stride: int = 30,
    n2: int = 20,
    signal_strength: float = 2.0,
    noise_amplitude: float = 0.05,
) -> list[np.ndarray]:
    """
    Apply Layer 2 signal as an overlay on L1 frames.

    L2 signal is additive - it shifts the L1 magnitude slightly.
    L2 uses discrete (non-overlapping) L1 windows for extraction,
    so L2 contribution is applied per N1-sized block.

    The L1 sliding window provides smooth carrier video.
    L2 extraction samples at N1 intervals for simplicity.

    Args:
        l1_frames: Base L1 frames from encode_l1_sliding
        qr_key: String to encode in L2 QR code
        payload: Bytes to hide in L2 magnitude
        n1: L1 window size
        stride: L1 stride (not used for L2, but kept for API consistency)
        n2: Number of non-overlapping L1 windows per L2 output
        signal_strength: Target L2 magnitude after full accumulation
        noise_amplitude: Per-frame L2 noise

    Returns:
        Frames with L2 overlay applied
    """
    if not l1_frames:
        return []

    frame_shape = l1_frames[0].shape
    total_frames = len(l1_frames)

    # Generate L2 QR pattern
    qr2 = generate_qr_matrix(qr_key, frame_shape[0])
    qr2_signs = np.where(qr2 == 0, 1.0, -1.0).astype(np.float32)

    # Encode payload into magnitude bias
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    n_cells = frame_shape[0] * frame_shape[1]

    magnitude_bias = np.zeros(frame_shape, dtype=np.float32)
    if len(payload_bits) > 0:
        flat_bias = magnitude_bias.flatten()
        for i in range(n_cells):
            bit_idx = i % len(payload_bits)
            flat_bias[i] = 0.5 if payload_bits[bit_idx] else -0.5
        magnitude_bias = flat_bias.reshape(frame_shape)

    # L2 target: QR2 sign * (base magnitude + payload bias)
    l2_target = qr2_signs * (signal_strength + magnitude_bias)

    # L2 uses discrete L1 windows: each N1 frames is one L1 output for L2
    # Spread L2 signal across N1 * N2 frames
    total_l2_frames = n1 * n2
    l2_per_frame = l2_target / total_l2_frames

    # Apply overlay to each frame
    result = []
    for i, frame in enumerate(l1_frames):
        rng = _frame_index_rng(f"l2:{qr_key}", i)
        noise = rng.uniform(-noise_amplitude, noise_amplitude,
                           size=frame_shape).astype(np.float32)

        # Only apply L2 within the L2 window
        if i < total_l2_frames:
            overlay = l2_per_frame + noise
        else:
            # Beyond first L2 window
            overlay = noise

        result.append(frame + overlay)

    return result


def decode_l1_at_offset(
    frames: list[np.ndarray],
    start: int,
    n1: int,
    qr_key: str = None,
    noise_amplitude: float = 0.2,
) -> tuple[np.ndarray, str | None]:
    """
    Decode Layer 1 starting at a specific frame offset.

    Args:
        frames: All carrier frames
        start: Starting frame index
        n1: Window size
        qr_key: Optional - if provided, cancels expected noise
        noise_amplitude: Noise amplitude for cancellation

    Returns:
        (accumulated, qr_content) tuple
    """
    if start + n1 > len(frames):
        raise ValueError(f"Not enough frames: need {start + n1}, have {len(frames)}")

    window = frames[start:start + n1]
    accumulated = np.sum(window, axis=0)

    # Noise cancellation if key provided
    if qr_key:
        frame_shape = frames[0].shape
        expected_noise = np.zeros(frame_shape, dtype=np.float32)
        for i in range(n1):
            rng = _frame_index_rng(qr_key, start + i)
            noise = rng.uniform(-noise_amplitude, noise_amplitude,
                               size=frame_shape).astype(np.float32)
            expected_noise += noise
        accumulated = accumulated - expected_noise

    # Extract QR from sign
    qr_matrix = (accumulated <= 0).astype(np.uint8)
    qr_content = scan_qr(qr_matrix)

    return accumulated, qr_content


def decode_l2(
    frames: list[np.ndarray],
    l1_qr_key: str,
    n1: int,
    n2: int,
    payload_length: int,
    l1_signal: float = 5.0,
    l1_noise: float = 0.2,
    l2_signal: float = 2.0,
    l2_noise: float = 0.05,
    stride: int = None,
) -> tuple[str | None, bytes | None]:
    """
    Decode Layer 2 from carrier frames.

    L2 uses discrete (non-overlapping) L1 windows, sampled at N1 intervals.
    This matches the encoding which applies L2 per N1-frame block.

    Args:
        frames: Carrier frames
        l1_qr_key: Known L1 QR key (for noise cancellation)
        n1: L1 window size
        n2: Number of L1 outputs per L2 output
        payload_length: Expected payload length in bytes
        l1_signal: L1 signal strength
        l1_noise: L1 noise amplitude
        l2_signal: L2 signal strength
        l2_noise: L2 noise amplitude
        stride: Ignored for L2 - uses n1 for discrete windows

    Returns:
        (l2_qr_content, payload) tuple
    """
    frame_shape = frames[0].shape

    # Get L1 QR pattern for magnitude extraction
    qr1 = generate_qr_matrix(l1_qr_key, frame_shape[0])
    qr1_signs = np.where(qr1 == 0, 1.0, -1.0)

    # Collect non-overlapping L1 outputs at N1 intervals
    l1_outputs = []
    for i in range(n2):
        offset = i * n1
        if offset + n1 > len(frames):
            break
        accumulated, _ = decode_l1_at_offset(
            frames, offset, n1, l1_qr_key, l1_noise
        )
        l1_outputs.append(accumulated)

    if len(l1_outputs) < n2:
        return None, None

    # Extract magnitude deviations from L1 outputs
    l2_accumulated = np.zeros(frame_shape, dtype=np.float32)

    for l1_output in l1_outputs:
        # Get magnitude (multiply by expected sign to make positive)
        magnitude = l1_output * qr1_signs
        # Deviation from expected L1 magnitude
        deviation = magnitude - l1_signal
        l2_accumulated += deviation

    # L2 accumulated has pattern qr1*qr2, multiply by qr1 to isolate qr2
    l2_corrected = l2_accumulated * qr1_signs
    qr2_matrix = (l2_corrected <= 0).astype(np.uint8)
    l2_qr_content = scan_qr(qr2_matrix)

    if not l2_qr_content:
        return None, None

    # Cancel L2 noise with known key
    l2_noise_sum = np.zeros(frame_shape, dtype=np.float32)
    for i in range(n1 * n2):
        rng = _frame_index_rng(f"l2:{l2_qr_content}", i)
        noise = rng.uniform(-l2_noise, l2_noise,
                           size=frame_shape).astype(np.float32)
        l2_noise_sum += noise

    cleaned_l2 = l2_accumulated - l2_noise_sum

    # Get L2 QR pattern for payload extraction
    qr2 = generate_qr_matrix(l2_qr_content, frame_shape[0])
    qr2_signs = np.where(qr2 == 0, 1.0, -1.0)

    # Extract payload from magnitude
    magnitude = cleaned_l2 * qr2_signs

    # Threshold and majority vote
    flat_magnitude = magnitude.flatten()
    n_bits = payload_length * 8
    n_cells = len(flat_magnitude)

    bit_votes = np.zeros(n_bits, dtype=np.float32)
    vote_counts = np.zeros(n_bits, dtype=np.int32)

    for i in range(n_cells):
        bit_idx = i % n_bits
        vote = 1.0 if flat_magnitude[i] > l2_signal else 0.0
        bit_votes[bit_idx] += vote
        vote_counts[bit_idx] += 1

    bits = (bit_votes > vote_counts / 2).astype(np.uint8)
    padded_bits = np.zeros(((len(bits) + 7) // 8) * 8, dtype=np.uint8)
    padded_bits[:len(bits)] = bits

    payload = np.packbits(padded_bits).tobytes()[:payload_length]

    return l2_qr_content, payload


class SlidingWindowEncoder:
    """
    Streaming encoder with sliding window L1 and optional L2 overlay.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int],
        l1_qr_key: str,
        n1: int = 60,
        stride: int = 30,
        l1_signal: float = 5.0,
        l1_noise: float = 0.2,
    ):
        self.frame_shape = frame_shape
        self.l1_qr_key = l1_qr_key
        self.n1 = n1
        self.stride = stride
        self.l1_signal = l1_signal
        self.l1_noise = l1_noise

        # Precompute L1 signal pattern
        qr = generate_qr_matrix(l1_qr_key, frame_shape[0])
        qr_signs = np.where(qr == 0, 1.0, -1.0).astype(np.float32)
        self.l1_signal_per_frame = (qr_signs * l1_signal) / n1

        # L2 overlay state
        self.l2_qr_key: str | None = None
        self.l2_payload: bytes | None = None
        self.l2_signal_per_frame: np.ndarray | None = None
        self.l2_n2: int = 0
        self.l2_frame_count: int = 0

        self.frame_index = 0

    def set_l2_message(
        self,
        qr_key: str,
        payload: bytes,
        n2: int = 20,
        signal_strength: float = 2.0,
    ):
        """Set L2 message to overlay on subsequent frames."""
        self.l2_qr_key = qr_key
        self.l2_payload = payload
        self.l2_n2 = n2
        self.l2_frame_count = 0

        # Compute L2 signal
        qr2 = generate_qr_matrix(qr_key, self.frame_shape[0])
        qr2_signs = np.where(qr2 == 0, 1.0, -1.0).astype(np.float32)

        # Encode payload
        payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        n_cells = self.frame_shape[0] * self.frame_shape[1]

        magnitude_bias = np.zeros(self.frame_shape, dtype=np.float32)
        if len(payload_bits) > 0:
            flat_bias = magnitude_bias.flatten()
            for i in range(n_cells):
                bit_idx = i % len(payload_bits)
                flat_bias[i] = 0.5 if payload_bits[bit_idx] else -0.5
            magnitude_bias = flat_bias.reshape(self.frame_shape)

        l2_target = qr2_signs * (signal_strength + magnitude_bias)
        total_l2_frames = self.n1 * n2
        self.l2_signal_per_frame = l2_target / total_l2_frames

    def next_frame(self) -> np.ndarray:
        """Generate next carrier frame."""
        # L1 base signal
        frame = self.l1_signal_per_frame.copy()

        # L1 noise
        rng = _frame_index_rng(self.l1_qr_key, self.frame_index)
        l1_noise = rng.uniform(-self.l1_noise, self.l1_noise,
                               size=self.frame_shape).astype(np.float32)
        frame += l1_noise

        # L2 overlay if active
        if self.l2_signal_per_frame is not None:
            total_l2_frames = self.n1 * self.l2_n2
            if self.l2_frame_count < total_l2_frames:
                frame += self.l2_signal_per_frame

                # L2 noise
                rng2 = _frame_index_rng(f"l2:{self.l2_qr_key}", self.l2_frame_count)
                l2_noise = rng2.uniform(-0.05, 0.05,
                                        size=self.frame_shape).astype(np.float32)
                frame += l2_noise

                self.l2_frame_count += 1

        self.frame_index += 1
        return frame


class SlidingWindowDecoder:
    """
    Streaming decoder for sliding window L1 with optional L2.
    """

    def __init__(
        self,
        n1: int = 60,
        stride: int = 30,
        l1_noise: float = 0.2,
    ):
        self.n1 = n1
        self.stride = stride
        self.l1_noise = l1_noise
        self._frames: list[np.ndarray] = []
        self._frame_index = 0
        self._last_l1_output: int = -stride  # Track when we last emitted

    def push_frame(self, frame: np.ndarray) -> tuple[str | None, np.ndarray | None]:
        """
        Push a frame. Returns L1 result at stride boundaries.

        Returns:
            (qr_content, accumulated) when L1 window completes, else (None, None)
        """
        self._frames.append(frame)
        self._frame_index += 1

        # Check if we have enough frames and it's a stride boundary
        frames_since_last = self._frame_index - self._last_l1_output

        if len(self._frames) >= self.n1 and frames_since_last >= self.stride:
            # Accumulate last n1 frames
            window = self._frames[-self.n1:]
            accumulated = np.sum(window, axis=0)

            # Extract QR
            qr_matrix = (accumulated <= 0).astype(np.uint8)
            qr_content = scan_qr(qr_matrix)

            self._last_l1_output = self._frame_index

            # Trim old frames we no longer need
            keep_frames = self.n1 + self.stride
            if len(self._frames) > keep_frames:
                self._frames = self._frames[-keep_frames:]

            return qr_content, accumulated

        return None, None

    def reset(self):
        """Reset decoder state."""
        self._frames = []
        self._frame_index = 0
        self._last_l1_output = -self.stride
