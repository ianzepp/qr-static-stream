#!/usr/bin/env python3
"""
Demo: Sliding Window Steganography

Shows how overlapping L1 windows create smooth carrier video
with no detectable boundaries.
"""

from qr_static_sliding import (
    encode_l1_sliding,
    apply_l2_overlay,
    decode_l1_at_offset,
    decode_l2,
    get_qr_module_size,
    SlidingWindowEncoder,
    SlidingWindowDecoder,
)
import numpy as np


def print_grayscale(matrix: np.ndarray, width: int = 60):
    """Print a float matrix as ASCII grayscale."""
    h, w = matrix.shape
    step = max(1, w // width)

    vmin, vmax = matrix.min(), matrix.max()
    if vmax == vmin:
        normalized = np.zeros_like(matrix)
    else:
        normalized = (matrix - vmin) / (vmax - vmin)

    chars = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    n_chars = len(chars)

    for y in range(0, h, step * 2):
        row = ""
        for x in range(0, w, step):
            if y < h and x < w:
                idx = int(normalized[y, x] * (n_chars - 1))
                row += chars[idx]
        print(row)


def demo_l1_sliding():
    """Demonstrate L1 sliding window - QR readable from any offset."""
    print("=" * 70)
    print("SLIDING WINDOW L1 - DECODE FROM ANY OFFSET")
    print("=" * 70)

    qr_key = "sliding-demo"
    frame_shape = (64, 64)
    n1 = 60
    stride = 30
    total_frames = 150

    print(f"\nConfiguration:")
    print(f"  QR key: {qr_key}")
    print(f"  Frame shape: {frame_shape}")
    print(f"  N1 (window size): {n1}")
    print(f"  Stride: {stride} (50% overlap)")
    print(f"  Total frames: {total_frames}")

    # Check module size
    module_size = get_qr_module_size(frame_shape[0], qr_key)
    print(f"  Pixels per QR module: {module_size}")

    # Encode
    print("\nEncoding L1 frames...")
    frames = encode_l1_sliding(qr_key, frame_shape, total_frames, n1, stride)

    print(f"Generated {len(frames)} frames")

    # Show sample frames
    print("\nFrame 0 (looks like static):")
    print_grayscale(frames[0])

    print("\nFrame 50 (also looks like static):")
    print_grayscale(frames[50])

    # Decode from various offsets
    print("\n" + "-" * 70)
    print("DECODING FROM DIFFERENT OFFSETS")
    print("-" * 70)

    offsets = [0, 15, 30, 45, 60]
    for offset in offsets:
        if offset + n1 <= len(frames):
            accumulated, qr_content = decode_l1_at_offset(
                frames, offset, n1, qr_key
            )
            status = "[OK]" if qr_content == qr_key else "[FAIL]"
            print(f"\nOffset {offset:3d}: {status} QR = {qr_content}")

            if offset == 0:
                print("\nAccumulated at offset 0:")
                print_grayscale(accumulated)


def demo_l1_streaming():
    """Demonstrate streaming encoder/decoder."""
    print("\n" + "=" * 70)
    print("SLIDING WINDOW L1 - STREAMING")
    print("=" * 70)

    qr_key = "stream-key"
    frame_shape = (64, 64)
    n1 = 60
    stride = 30

    encoder = SlidingWindowEncoder(frame_shape, qr_key, n1, stride)
    decoder = SlidingWindowDecoder(n1, stride)

    print(f"\nStreaming {n1 + stride * 3} frames...")
    print(f"Expect L1 output at frames {n1}, {n1+stride}, {n1+stride*2}")

    l1_outputs = []
    for i in range(n1 + stride * 3):
        frame = encoder.next_frame()
        result, accumulated = decoder.push_frame(frame)

        if result:
            l1_outputs.append((i, result))
            status = "[OK]" if result == qr_key else "[FAIL]"
            print(f"  Frame {i}: L1 output - {status} QR = {result}")

    print(f"\nTotal L1 outputs: {len(l1_outputs)}")


def demo_l1_l2_composable():
    """Demonstrate L1 + L2 composition."""
    print("\n" + "=" * 70)
    print("COMPOSABLE L1 + L2")
    print("=" * 70)

    l1_key = "layer-one"
    l2_key = "layer-two"
    payload = b"Hidden in L2!"

    frame_shape = (64, 64)
    n1 = 60
    stride = 30
    n2 = 10
    total_frames = n1 * n2  # 600 frames for one full L2 cycle

    print(f"\nConfiguration:")
    print(f"  L1 key: {l1_key}")
    print(f"  L2 key: {l2_key}")
    print(f"  Payload: {payload}")
    print(f"  N1: {n1}, N2: {n2}")
    print(f"  Total frames: {total_frames}")

    # Step 1: Generate pure L1 stream
    print("\nStep 1: Generate L1 stream...")
    l1_frames = encode_l1_sliding(l1_key, frame_shape, total_frames, n1, stride)

    # Verify L1 works on its own
    _, l1_qr = decode_l1_at_offset(l1_frames, 0, n1, l1_key)
    print(f"  L1 alone: QR = {l1_qr}")

    # Step 2: Apply L2 overlay
    print("\nStep 2: Apply L2 overlay...")
    final_frames = apply_l2_overlay(l1_frames, l2_key, payload, n1, stride, n2)

    # Verify L1 still works after L2 overlay
    _, l1_qr_after = decode_l1_at_offset(final_frames, 0, n1, l1_key)
    print(f"  L1 after L2: QR = {l1_qr_after}")

    if l1_qr_after == l1_key:
        print("  [OK] L2 overlay doesn't disrupt L1")
    else:
        print("  [FAIL] L2 overlay disrupted L1")

    # Step 3: Decode L2
    print("\nStep 3: Decode L2...")
    l2_qr, decoded_payload = decode_l2(
        final_frames, l1_key, n1, n2, len(payload),
        stride=stride
    )

    print(f"  L2 QR: {l2_qr}")
    print(f"  Payload: {decoded_payload}")

    if l2_qr == l2_key:
        print("  [OK] L2 QR matches")
    else:
        print("  [FAIL] L2 QR mismatch")

    if decoded_payload == payload:
        print("  [OK] Payload matches exactly!")
    elif decoded_payload:
        matches = sum(a == b for a, b in zip(payload, decoded_payload))
        print(f"  [PARTIAL] Payload: {matches}/{len(payload)} bytes match")


def demo_boundary_smoothness():
    """Show that there are no detectable boundaries."""
    print("\n" + "=" * 70)
    print("BOUNDARY SMOOTHNESS ANALYSIS")
    print("=" * 70)

    qr_key = "smooth-test"
    frame_shape = (64, 64)
    n1 = 60
    stride = 30
    total_frames = 120

    frames = encode_l1_sliding(qr_key, frame_shape, total_frames, n1, stride)

    # Compare statistics across frames
    print("\nFrame statistics (should be consistent across all frames):")
    print("-" * 50)

    checkpoints = [0, 29, 30, 31, 59, 60, 61, 89, 90]
    for i in checkpoints:
        if i < len(frames):
            f = frames[i]
            marker = " <-- stride boundary" if i % stride == 0 else ""
            print(f"Frame {i:3d}: mean={f.mean():+.4f}, std={f.std():.4f}{marker}")

    # Show that frame 29->30 transition is smooth (no discontinuity)
    print("\n" + "-" * 50)
    print("Frame-to-frame difference (should be consistent):")

    for i in [28, 29, 30, 58, 59, 60]:
        if i + 1 < len(frames):
            diff = np.abs(frames[i+1] - frames[i]).mean()
            marker = " <-- stride boundary" if (i + 1) % stride == 0 else ""
            print(f"Frames {i}->{i+1}: mean |diff| = {diff:.4f}{marker}")


def demo_capacity():
    """Show capacity at different resolutions."""
    print("\n" + "=" * 70)
    print("SLIDING WINDOW - CAPACITY ANALYSIS")
    print("=" * 70)

    scenarios = [
        ("720p", (720, 1280), 60, 30, 20),
        ("1080p", (1080, 1920), 60, 30, 20),
        ("1080p deep", (1080, 1920), 60, 30, 60),
    ]

    print("\nAt 30fps:")
    print("-" * 70)

    for name, shape, n1, stride, n2 in scenarios:
        l1_time = n1 / 30
        l2_time = n1 * n2 / 30

        # Payload capacity
        pixels = shape[0] * shape[1]
        payload_bytes = pixels // 8

        print(f"\n{name}:")
        print(f"  L1 window: {n1} frames ({l1_time:.1f}s), stride: {stride}")
        print(f"  L2 accumulation: {n1 * n2} frames ({l2_time:.1f}s)")
        print(f"  Payload capacity: {payload_bytes:,} bytes per L2 output")


if __name__ == "__main__":
    demo_l1_sliding()
    demo_l1_streaming()
    demo_l1_l2_composable()
    demo_boundary_smoothness()
    demo_capacity()
