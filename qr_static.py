"""
QR Static Stream - Steganographic QR codes hidden in XOR'd noise frames.

Each frame is random static. XOR all N frames together to reveal a QR code.
"""

from __future__ import annotations

import numpy as np
from functools import reduce
from typing import Optional


def generate_qr_matrix(data: str, size: int = None) -> np.ndarray:
    """
    Generate a binary matrix from QR code data.
    Returns a 2D numpy array of 0s and 1s.
    """
    import qrcode

    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
        box_size=1,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Get the QR matrix (list of lists of bools)
    modules = qr.get_matrix()
    matrix = np.array(modules, dtype=np.uint8)

    # Resize if specific size requested
    if size and matrix.shape[0] != size:
        matrix = _resize_matrix(matrix, size)

    return matrix


def _resize_matrix(matrix: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize matrix to target size.

    For QR codes, padding with white (0) preserves scannability better than
    interpolation which can break the alignment patterns.
    """
    from_size = matrix.shape[0]

    if target_size == from_size:
        return matrix

    if target_size > from_size:
        # Pad with zeros (white in QR terms) — centered
        pad_total = target_size - from_size
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(matrix, pad_before, mode='constant', constant_values=0)

    # Shrinking — crop from center (unusual case, avoid if possible)
    crop_total = from_size - target_size
    crop_start = crop_total // 2
    return matrix[crop_start:crop_start + target_size, crop_start:crop_start + target_size]


def generate_noise_frame(shape: tuple[int, int], rng: np.random.Generator = None) -> np.ndarray:
    """Generate a single frame of binary noise."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 2, size=shape, dtype=np.uint8)


def encode(qr_matrix: np.ndarray, n_frames: int, seed: int = None) -> list[np.ndarray]:
    """
    Encode a QR matrix into N frames of noise.

    XOR of all returned frames equals the original QR matrix.

    Args:
        qr_matrix: Binary matrix (0s and 1s) representing the QR code
        n_frames: Number of frames to generate (minimum 2)
        seed: Random seed for reproducibility

    Returns:
        List of N binary matrices (noise frames)
    """
    if n_frames < 2:
        raise ValueError("Need at least 2 frames")

    rng = np.random.default_rng(seed)
    shape = qr_matrix.shape

    # Generate N-1 random frames
    frames = [generate_noise_frame(shape, rng) for _ in range(n_frames - 1)]

    # Compute final frame: QR XOR all previous frames
    # XOR is associative, so: final = QR ^ frame[0] ^ frame[1] ^ ... ^ frame[N-2]
    # Then: frame[0] ^ frame[1] ^ ... ^ frame[N-2] ^ final = QR
    accumulated = reduce(np.bitwise_xor, frames)
    final_frame = np.bitwise_xor(qr_matrix, accumulated)

    frames.append(final_frame)
    return frames


def decode(frames: list[np.ndarray]) -> np.ndarray:
    """
    Decode frames by XORing them together.

    Args:
        frames: List of binary matrices

    Returns:
        Resulting matrix (should be QR code if frames are valid)
    """
    return reduce(np.bitwise_xor, frames)


def scan_qr(matrix: np.ndarray) -> str | None:
    """
    Attempt to scan a QR code from a binary matrix.

    Returns decoded string or None if no QR found.
    """
    import cv2

    # Convert to image (0 -> white 255, 1 -> black 0)
    img = ((1 - matrix) * 255).astype(np.uint8)

    # OpenCV needs larger images to reliably detect QR codes
    min_size = 100
    if img.shape[0] < min_size:
        scale = (min_size // img.shape[0]) + 1
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

    # OpenCV QR detector
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)

    return data if data else None


class StreamEncoder:
    """
    Encodes a stream of QR codes into continuous noise frames.

    Each cycle of N frames XORs to reveal one QR code.
    """

    def __init__(self, n_frames: int, frame_shape: tuple[int, int], seed: int = None):
        self.n_frames = n_frames
        self.frame_shape = frame_shape
        self.rng = np.random.default_rng(seed)
        self._queue: list[str] = []
        self._current_frames: list[np.ndarray] = []
        self._frame_index = 0

    def queue_message(self, data: str):
        """Add a message to the encoding queue."""
        self._queue.append(data)

    def next_frame(self) -> np.ndarray:
        """
        Get the next frame in the stream.

        If no messages queued, returns pure noise.
        """
        # Start new cycle if needed
        if self._frame_index >= len(self._current_frames):
            self._start_new_cycle()

        frame = self._current_frames[self._frame_index]
        self._frame_index += 1
        return frame

    def _start_new_cycle(self):
        """Begin a new N-frame cycle."""
        self._frame_index = 0

        if self._queue:
            # Encode next message
            data = self._queue.pop(0)
            qr = generate_qr_matrix(data, size=self.frame_shape[0])
            self._current_frames = encode(qr, self.n_frames, seed=None)
            # Re-seed frames with our RNG for variety
            for i in range(len(self._current_frames) - 1):
                self._current_frames[i] = generate_noise_frame(self.frame_shape, self.rng)
            # Recompute final frame
            accumulated = reduce(np.bitwise_xor, self._current_frames[:-1])
            self._current_frames[-1] = np.bitwise_xor(qr, accumulated)
        else:
            # Pure noise - no hidden message
            self._current_frames = [
                generate_noise_frame(self.frame_shape, self.rng)
                for _ in range(self.n_frames)
            ]


class StreamDecoder:
    """
    Decodes a stream of noise frames to extract hidden QR codes.
    """

    def __init__(self, n_frames: int):
        self.n_frames = n_frames
        self._buffer: list[np.ndarray] = []
        self._accumulated: np.ndarray | None = None

    def push_frame(self, frame: np.ndarray) -> str | None:
        """
        Push a frame and attempt to decode.

        Returns decoded message if cycle complete and valid QR found,
        otherwise None.
        """
        self._buffer.append(frame)

        if self._accumulated is None:
            self._accumulated = frame.copy()
        else:
            self._accumulated = np.bitwise_xor(self._accumulated, frame)

        # Check if cycle complete
        if len(self._buffer) >= self.n_frames:
            result = scan_qr(self._accumulated)
            self._reset()
            return result

        return None

    def _reset(self):
        """Reset for next cycle."""
        self._buffer = []
        self._accumulated = None

    def peek(self) -> np.ndarray | None:
        """Get current accumulated state without resetting."""
        return self._accumulated.copy() if self._accumulated is not None else None
