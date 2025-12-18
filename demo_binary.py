#!/usr/bin/env python3
"""
Demo: Binary Static Steganography

Each frame is pure binary static (+1/-1 per pixel).
Grayscale emerges from temporal accumulation.
"""

from qr_static_binary import (
    encode_binary,
    accumulate,
    extract_qr,
    scan_qr,
    decode_payload,
    BinaryStreamEncoder,
    BinaryStreamDecoder,
)
import numpy as np


def print_binary_frame(frame: np.ndarray, width: int = 70):
    """Print a binary frame as ASCII."""
    h, w = frame.shape
    step_x = max(1, w // width)
    step_y = max(1, h // (width // 2))

    for y in range(0, h, step_y):
        row = ""
        for x in range(0, w, step_x):
            row += "█" if frame[y, x] > 0 else "░"
        print(row)


def print_accumulated(accumulated: np.ndarray, width: int = 70):
    """Print accumulated values as grayscale ASCII."""
    h, w = accumulated.shape
    step_x = max(1, w // width)
    step_y = max(1, h // (width // 2))

    vmin, vmax = accumulated.min(), accumulated.max()
    if vmax == vmin:
        normalized = np.zeros_like(accumulated, dtype=float)
    else:
        normalized = (accumulated - vmin) / (vmax - vmin)

    chars = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    n_chars = len(chars)

    for y in range(0, h, step_y):
        row = ""
        for x in range(0, w, step_x):
            idx = int(normalized[y, x] * (n_chars - 1))
            row += chars[idx]
        print(row)


def demo_basic():
    """Basic binary static demo."""
    print("=" * 70)
    print("BINARY STATIC - BASIC DEMO")
    print("=" * 70)

    qr_key = "binary-static"
    frame_shape = (64, 64)
    n_frames = 60
    base_bias = 0.8  # Strong bias for clear signal

    print(f"\nConfiguration:")
    print(f"  QR key: {qr_key}")
    print(f"  Frame shape: {frame_shape}")
    print(f"  N frames: {n_frames}")
    print(f"  Base bias: {base_bias} (P(+1) for white modules)")

    # Encode
    print("\nEncoding binary frames...")
    frames = encode_binary(qr_key, frame_shape, n_frames, base_bias)

    print(f"Generated {len(frames)} binary frames")
    print(f"Frame dtype: {frames[0].dtype}")
    print(f"Frame values: {np.unique(frames[0])}")
    print(f"Bits per pixel: 1 (stored as int8 for convenience)")

    # Show sample frames
    print("\nFrame 0 (pure binary static):")
    print_binary_frame(frames[0])

    print("\nFrame 30 (also pure binary static):")
    print_binary_frame(frames[30])

    # Accumulate
    print("\n" + "-" * 70)
    print("ACCUMULATION")
    print("-" * 70)

    accumulated = accumulate(frames)
    print(f"\nAccumulated dtype: {accumulated.dtype}")
    print(f"Value range: [{accumulated.min()}, {accumulated.max()}]")
    print(f"Expected range: [{-n_frames}, {n_frames}]")

    print("\nAccumulated (grayscale emerges!):")
    print_accumulated(accumulated)

    # Extract QR
    print("\n" + "-" * 70)
    print("QR EXTRACTION")
    print("-" * 70)

    qr_matrix = extract_qr(accumulated)
    qr_content = scan_qr(qr_matrix)

    print(f"\nExtracted QR: {qr_content}")
    if qr_content == qr_key:
        print("[OK] QR matches!")
    else:
        print("[FAIL] QR mismatch")


def demo_payload():
    """Demo with hidden payload."""
    print("\n" + "=" * 70)
    print("BINARY STATIC - PAYLOAD DEMO")
    print("=" * 70)

    qr_key = "payload-key"
    payload = b"Secret message!"
    frame_shape = (64, 64)
    n_frames = 60
    base_bias = 0.8

    print(f"\nConfiguration:")
    print(f"  QR key: {qr_key}")
    print(f"  Payload: {payload}")
    print(f"  Frame shape: {frame_shape}")
    print(f"  N frames: {n_frames}")
    print(f"  Base bias: {base_bias}")

    # Encode with payload
    print("\nEncoding with payload...")
    frames = encode_binary(
        qr_key, frame_shape, n_frames, base_bias,
        payload=payload, payload_bias_delta=0.1
    )

    # Accumulate
    accumulated = accumulate(frames)

    # Extract QR
    qr_matrix = extract_qr(accumulated)
    qr_content = scan_qr(qr_matrix)
    print(f"\nExtracted QR: {qr_content}")

    if qr_content != qr_key:
        print("[FAIL] QR mismatch - cannot decode payload")
        return

    print("[OK] QR matches")

    # Decode payload
    decoded_payload = decode_payload(
        accumulated, qr_key, n_frames, len(payload), base_bias
    )
    print(f"Decoded payload: {decoded_payload}")

    if decoded_payload == payload:
        print("[OK] Payload matches exactly!")
    else:
        matches = sum(a == b for a, b in zip(payload, decoded_payload))
        print(f"[PARTIAL] {matches}/{len(payload)} bytes match")


def demo_streaming():
    """Demo streaming encoder/decoder."""
    print("\n" + "=" * 70)
    print("BINARY STATIC - STREAMING DEMO")
    print("=" * 70)

    qr_key = "stream-demo"
    frame_shape = (64, 64)
    n_frames = 60

    encoder = BinaryStreamEncoder(frame_shape, qr_key, n_frames, base_bias=0.8)
    decoder = BinaryStreamDecoder(n_frames)

    print(f"\nStreaming {n_frames * 2} frames (expect 2 outputs)...")

    outputs = []
    for i in range(n_frames * 2):
        frame = encoder.next_frame()
        result, accumulated = decoder.push_frame(frame)

        if result:
            outputs.append((i, result))
            status = "[OK]" if result == qr_key else "[FAIL]"
            print(f"  Frame {i}: {status} QR = {result}")

    print(f"\nTotal outputs: {len(outputs)}")


def demo_efficiency():
    """Show memory efficiency comparison."""
    print("\n" + "=" * 70)
    print("MEMORY EFFICIENCY COMPARISON")
    print("=" * 70)

    frame_shape = (1080, 1920)  # 1080p
    n_frames = 60

    # Binary approach
    binary_frame_size = frame_shape[0] * frame_shape[1] * 1  # 1 byte (int8)
    binary_stream_size = binary_frame_size * n_frames
    binary_accum_size = frame_shape[0] * frame_shape[1] * 2  # int16

    # Float approach (current implementation)
    float_frame_size = frame_shape[0] * frame_shape[1] * 4  # float32
    float_stream_size = float_frame_size * n_frames
    float_accum_size = frame_shape[0] * frame_shape[1] * 4  # float32

    print(f"\nAt 1080p ({frame_shape[0]}x{frame_shape[1]}), N={n_frames}:")
    print()
    print("                    Binary (new)    Float (old)    Savings")
    print("-" * 65)
    print(f"Per frame:          {binary_frame_size/1024/1024:6.2f} MB       {float_frame_size/1024/1024:6.2f} MB       {float_frame_size/binary_frame_size:.0f}x")
    print(f"Stream ({n_frames} frames): {binary_stream_size/1024/1024:6.2f} MB      {float_stream_size/1024/1024:6.2f} MB       {float_stream_size/binary_stream_size:.0f}x")
    print(f"Accumulator:        {binary_accum_size/1024/1024:6.2f} MB       {float_accum_size/1024/1024:6.2f} MB       {float_accum_size/binary_accum_size:.0f}x")

    print("\n" + "-" * 65)
    print("Theoretical minimum (1 bit per pixel):")
    bit_frame_size = frame_shape[0] * frame_shape[1] / 8
    print(f"Per frame:          {bit_frame_size/1024/1024:6.2f} MB       (packed bits)")
    print(f"Stream ({n_frames} frames): {bit_frame_size * n_frames/1024/1024:6.2f} MB")


def demo_visual_comparison():
    """Show that individual frames are indistinguishable."""
    print("\n" + "=" * 70)
    print("VISUAL COMPARISON - SIGNAL VS RANDOM")
    print("=" * 70)

    frame_shape = (64, 64)
    n_frames = 60

    # Frames with embedded QR
    signal_frames = encode_binary("hidden-qr", frame_shape, n_frames, base_bias=0.8)

    # Pure random frames (no signal)
    rng = np.random.default_rng(12345)
    random_frames = [
        np.where(rng.random(frame_shape) < 0.5, 1, -1).astype(np.int8)
        for _ in range(n_frames)
    ]

    print("\nSingle frame with embedded signal:")
    print_binary_frame(signal_frames[0])

    print("\nSingle frame of pure random:")
    print_binary_frame(random_frames[0])

    print("\nStatistics (should be nearly identical):")
    signal_mean = np.mean([f.mean() for f in signal_frames])
    random_mean = np.mean([f.mean() for f in random_frames])
    signal_std = np.mean([f.std() for f in signal_frames])
    random_std = np.mean([f.std() for f in random_frames])

    print(f"  Signal frames - mean: {signal_mean:+.4f}, std: {signal_std:.4f}")
    print(f"  Random frames - mean: {random_mean:+.4f}, std: {random_std:.4f}")

    print("\nAccumulated signal frames (QR emerges):")
    print_accumulated(accumulate(signal_frames))

    print("\nAccumulated random frames (noise):")
    print_accumulated(accumulate(random_frames))


if __name__ == "__main__":
    demo_basic()
    demo_payload()
    demo_streaming()
    demo_efficiency()
    demo_visual_comparison()
