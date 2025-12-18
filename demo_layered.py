#!/usr/bin/env python3
"""
Demo: Two-Layer Steganography

Layer 1: Accumulate N₁ carrier frames → QR₁ emerges from sign
Layer 2: Accumulate magnitude deviations across N₂ Layer 1 outputs → QR₂ + payload

Four secrets to unlock:
  1. N₁ (Layer 1 frame count)
  2. N₂ (Layer 2 frame count)
  3. QR₁ content (Layer 1 key)
  4. QR₂ content (Layer 2 key - decodes payload)
"""

from qr_static_layered import (
    encode,
    decode,
    decode_layer1,
    extract_qr_from_accumulated,
    decode_layer2,
    scan_qr,
    TwoLayerEncoder,
    TwoLayerDecoder,
)
import numpy as np


def print_binary(matrix: np.ndarray, width: int = 60):
    """Print a binary matrix."""
    h, w = matrix.shape
    step = max(1, w // width)
    for y in range(0, h, step * 2):
        row = ""
        for x in range(0, w, step):
            if y < h and x < w:
                row += "█" if matrix[y, x] else " "
        print(row)


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


def demo_basic():
    """Basic two-layer encode/decode."""
    print("=" * 70)
    print("TWO-LAYER STEGANOGRAPHY - BASIC DEMO")
    print("=" * 70)

    layer1_key = "layer1-visible"
    layer2_key = "layer2-hidden"
    payload = b"Secret message in the deepest layer!"

    n1 = 10  # Carrier frames per Layer 1 output
    n2 = 5   # Layer 1 outputs per Layer 2 output
    frame_shape = (64, 64)

    print(f"\nConfiguration:")
    print(f"  Layer 1 key: {layer1_key}")
    print(f"  Layer 2 key: {layer2_key}")
    print(f"  Payload: {payload}")
    print(f"  N1 (frames per L1): {n1}")
    print(f"  N2 (L1 outputs per L2): {n2}")
    print(f"  Total frames: {n1 * n2}")
    print(f"  Frame shape: {frame_shape}")

    # Encode
    print("\nEncoding...")
    frames = encode(layer1_key, layer2_key, payload, frame_shape, n1, n2)
    print(f"Generated {len(frames)} carrier frames")

    # Show a carrier frame (looks like static)
    print("\nSample carrier frame (looks like static):")
    print_grayscale(frames[0])

    # Decode
    print("\n" + "-" * 70)
    print("DECODING")
    print("-" * 70)

    decoded_l1, decoded_l2, decoded_payload = decode(
        frames, n1, n2, payload_length=len(payload)
    )

    print(f"\nLayer 1 QR: {decoded_l1}")
    print(f"Layer 2 QR: {decoded_l2}")
    print(f"Payload: {decoded_payload}")

    if decoded_l1 == layer1_key:
        print("\n[OK] Layer 1 key matches!")
    else:
        print(f"\n[FAIL] Layer 1 key mismatch: expected '{layer1_key}'")

    if decoded_l2 == layer2_key:
        print("[OK] Layer 2 key matches!")
    else:
        print(f"[FAIL] Layer 2 key mismatch: expected '{layer2_key}'")

    if decoded_payload == payload:
        print("[OK] Payload matches exactly!")
    else:
        matches = sum(a == b for a, b in zip(payload, decoded_payload or b""))
        print(f"[PARTIAL] Payload: {matches}/{len(payload)} bytes match")


def demo_progressive():
    """Show progressive decoding through layers."""
    print("\n" + "=" * 70)
    print("TWO-LAYER STEGANOGRAPHY - PROGRESSIVE DECODING")
    print("=" * 70)

    layer1_key = "outer-key"
    layer2_key = "inner-key"
    payload = b"Nested secret!"

    # Need sufficient frames for SNR to build up at both layers
    n1 = 15
    n2 = 10
    frame_shape = (64, 64)

    print(f"\nEncoding with N1={n1}, N2={n2} ({n1 * n2} total frames)...")
    frames = encode(layer1_key, layer2_key, payload, frame_shape, n1, n2)

    # Step 1: Accumulate first N1 frames to get Layer 1 output
    print("\n" + "-" * 70)
    print("STEP 1: Accumulate first N1 frames")
    print("-" * 70)

    layer1_outputs = decode_layer1(frames, n1)
    print(f"Got {len(layer1_outputs)} Layer 1 outputs from {len(frames)} carrier frames")

    # Show first Layer 1 output
    print("\nFirst Layer 1 accumulated output:")
    print_grayscale(layer1_outputs[0])

    # Extract QR from Layer 1
    qr1_matrix = extract_qr_from_accumulated(layer1_outputs[0])
    qr1_content = scan_qr(qr1_matrix)
    print(f"\nLayer 1 QR content: {qr1_content}")

    print("\nLayer 1 QR pattern:")
    print_binary(qr1_matrix)

    # Step 2: Decode Layer 2 from magnitude deviations
    print("\n" + "-" * 70)
    print("STEP 2: Extract Layer 2 from magnitude deviations")
    print("-" * 70)

    layer2_accumulated = decode_layer2(
        layer1_outputs, qr1_content, n1,
        layer1_signal=5.0, layer1_noise=0.2,
    )
    print("\nLayer 2 accumulated field:")
    print_grayscale(layer2_accumulated)

    qr2_matrix = extract_qr_from_accumulated(layer2_accumulated)
    qr2_content = scan_qr(qr2_matrix)
    print(f"\nLayer 2 QR content: {qr2_content}")

    print("\nLayer 2 QR pattern:")
    print_binary(qr2_matrix)

    # Step 3: Decode payload
    print("\n" + "-" * 70)
    print("STEP 3: Decode payload using Layer 2 key")
    print("-" * 70)

    if qr2_content:
        from qr_static_layered import decode_layer2_payload
        decoded_payload = decode_layer2_payload(
            layer2_accumulated, qr2_content, n2, len(payload),
            layer2_signal=2.0, layer2_noise=0.1,
        )

        print(f"\nDecoded payload: {decoded_payload}")
        if decoded_payload == payload:
            print("[OK] Payload matches!")
        else:
            matches = sum(a == b for a, b in zip(payload, decoded_payload))
            print(f"[PARTIAL] Payload: {matches}/{len(payload)} bytes match")
    else:
        print("\n[SKIP] Cannot decode payload - Layer 2 QR not readable")


def demo_streaming():
    """Streaming encoder/decoder."""
    print("\n" + "=" * 70)
    print("TWO-LAYER STEGANOGRAPHY - STREAMING")
    print("=" * 70)

    n1 = 12
    n2 = 8
    frame_shape = (64, 64)

    encoder = TwoLayerEncoder(frame_shape, n1, n2)
    decoder = TwoLayerDecoder(n1, n2, expected_payload_length=16)

    # Set message
    layer1_key = "stream-l1"
    layer2_key = "stream-l2"
    payload = b"Streaming test!"

    # Pad payload to expected length
    payload = payload.ljust(16, b'\x00')

    encoder.set_message(layer1_key, layer2_key, payload)

    print(f"\nEncoder has {encoder.total_frames} frames to send")
    print(f"N1={n1}, N2={n2}")

    # Stream frames
    frame_count = 0
    while True:
        frame = encoder.next_frame()
        if frame is None:
            break

        result = decoder.push_frame(frame)
        frame_count += 1

        if result:
            dec_l1, dec_l2, dec_payload = result
            print(f"\nAfter {frame_count} frames:")
            print(f"  Layer 1 key: {dec_l1}")
            print(f"  Layer 2 key: {dec_l2}")
            print(f"  Payload: {dec_payload}")

            if dec_l1 == layer1_key and dec_l2 == layer2_key:
                print("  [OK] Both keys match!")
            if dec_payload and dec_payload.rstrip(b'\x00') == payload.rstrip(b'\x00'):
                print("  [OK] Payload matches!")
            elif dec_payload:
                matches = sum(a == b for a, b in zip(payload, dec_payload))
                print(f"  [PARTIAL] Payload: {matches}/{len(payload)} bytes match")
            break

    print(f"\nTotal frames processed: {frame_count}")


def demo_capacity():
    """Show capacity at different resolutions."""
    print("\n" + "=" * 70)
    print("TWO-LAYER STEGANOGRAPHY - CAPACITY ANALYSIS")
    print("=" * 70)

    scenarios = [
        ("480p", (480, 640), 30, 30),
        ("720p", (720, 1280), 30, 30),
        ("1080p", (1080, 1920), 30, 30),
    ]

    print("\nAssuming 30fps video, N1=N2=30:")
    print("-" * 70)
    print(f"{'Resolution':<12} {'Frame Size':<14} {'L1 Time':<10} {'L2 Time':<10} {'Payload/L2':<12}")
    print("-" * 70)

    for name, shape, n1, n2 in scenarios:
        l1_time = n1 / 30  # seconds per Layer 1 output
        l2_time = n1 * n2 / 30  # seconds per Layer 2 output

        # Payload capacity: 1 bit per pixel per Layer 2 output
        pixels = shape[0] * shape[1]
        payload_bits = pixels
        payload_bytes = payload_bits // 8

        print(f"{name:<12} {shape[0]}x{shape[1]:<8} {l1_time:.1f}s{'':<6} {l2_time:.1f}s{'':<6} {payload_bytes:,} bytes")

    print("\n10-minute video payload capacity:")
    print("-" * 70)

    for name, shape, n1, n2 in scenarios:
        total_seconds = 600
        total_frames = total_seconds * 30
        l2_outputs = total_frames // (n1 * n2)

        pixels = shape[0] * shape[1]
        payload_per_l2 = pixels // 8
        total_payload = l2_outputs * payload_per_l2

        print(f"{name:<12}: {l2_outputs} Layer 2 outputs = {total_payload:,} bytes ({total_payload / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    demo_basic()
    demo_progressive()
    demo_streaming()
    demo_capacity()
