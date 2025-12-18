#!/usr/bin/env python3
"""
Demo: Encode a message into noise frames, decode it back.
"""

from qr_static import (
    generate_qr_matrix,
    encode,
    decode,
    scan_qr,
    StreamEncoder,
    StreamDecoder,
)
import numpy as np


def print_matrix(matrix: np.ndarray, width: int = 60):
    """Print a binary matrix as ASCII art."""
    h, w = matrix.shape
    # Downsample if too wide
    step = max(1, w // width)
    for y in range(0, h, step * 2):  # *2 because terminal chars are taller than wide
        row = ""
        for x in range(0, w, step):
            row += "█" if matrix[y, x] else " "
        print(row)


def demo_basic():
    """Basic encode/decode cycle."""
    print("=" * 60)
    print("BASIC ENCODE/DECODE")
    print("=" * 60)

    message = "Hello, steganography!"
    n_frames = 8

    print(f"\nMessage: {message}")
    print(f"Frames: {n_frames}")

    # Generate QR
    qr = generate_qr_matrix(message)
    print(f"\nQR matrix shape: {qr.shape}")
    print("\nOriginal QR:")
    print_matrix(qr)

    # Encode into noise frames
    frames = encode(qr, n_frames, seed=42)

    print("\nNoise frames (first 3):")
    for i, frame in enumerate(frames[:3]):
        print(f"\nFrame {i}:")
        print_matrix(frame)

    # Decode
    recovered = decode(frames)
    print("\nRecovered QR (XOR of all frames):")
    print_matrix(recovered)

    # Verify matrices match
    if np.array_equal(qr, recovered):
        print("\n✓ Matrices match exactly!")
    else:
        diff = np.sum(qr != recovered)
        print(f"\n✗ Mismatch: {diff} pixels differ")

    # Scan QR
    result = scan_qr(recovered)
    if result:
        print(f"✓ Scanned message: {result}")
    else:
        print("✗ Could not scan QR code")


def demo_stream():
    """Streaming encode/decode."""
    print("\n" + "=" * 60)
    print("STREAMING ENCODE/DECODE")
    print("=" * 60)

    n_frames = 5
    messages = ["First message", "https://example.com", "Secret #3"]

    # Determine frame size from largest message QR
    max_qr = max(generate_qr_matrix(msg).shape[0] for msg in messages)
    frame_size = max_qr

    encoder = StreamEncoder(n_frames, (frame_size, frame_size), seed=123)
    decoder = StreamDecoder(n_frames)

    # Queue messages
    for msg in messages:
        encoder.queue_message(msg)

    print(f"\nQueued {len(messages)} messages")
    print(f"Frame size: {frame_size}x{frame_size}")
    print(f"Cycle length: {n_frames} frames")
    print(f"Total frames to process: {len(messages) * n_frames}")

    # Process frames
    decoded = []
    for i in range(len(messages) * n_frames + 5):  # Extra frames past the messages
        frame = encoder.next_frame()
        result = decoder.push_frame(frame)

        if result:
            decoded.append(result)
            print(f"  Frame {i}: Decoded -> {result}")

    print(f"\nDecoded {len(decoded)} messages:")
    for i, msg in enumerate(decoded):
        original = messages[i] if i < len(messages) else "(noise)"
        match = "✓" if i < len(messages) and msg == messages[i] else "?"
        print(f"  {match} {msg}")


def demo_partial():
    """Show progressive XOR accumulation."""
    print("\n" + "=" * 60)
    print("PROGRESSIVE ACCUMULATION")
    print("=" * 60)

    message = "XOR"
    n_frames = 4

    qr = generate_qr_matrix(message)
    frames = encode(qr, n_frames, seed=99)

    print(f"\nMessage: {message}")
    print(f"Frames: {n_frames}")
    print("\nAccumulating XOR frame by frame:\n")

    accumulated = frames[0].copy()
    for i in range(n_frames):
        if i > 0:
            accumulated = np.bitwise_xor(accumulated, frames[i])

        print(f"After {i + 1} frame(s):")
        print_matrix(accumulated)

        # Try to scan
        result = scan_qr(accumulated)
        if result:
            print(f"  -> Scannable! Got: {result}\n")
        else:
            print("  -> Not scannable yet\n")


if __name__ == "__main__":
    demo_basic()
    demo_stream()
    demo_partial()
