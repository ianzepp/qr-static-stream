#!/usr/bin/env python3
"""
Demo: Signed accumulation with two-layer steganography.

Layer 1: Accumulate N frames → QR emerges from sign
Layer 2: Use QR as key → decode payload from magnitudes
"""

from qr_static_signed import (
    encode,
    accumulate,
    extract_qr,
    decode_payload,
    scan_qr,
    SignedStreamEncoder,
    SignedStreamDecoder,
)
import numpy as np


def print_matrix(matrix: np.ndarray, width: int = 50):
    """Print a binary matrix as ASCII art."""
    h, w = matrix.shape
    step = max(1, w // width)
    for y in range(0, h, step * 2):
        row = ""
        for x in range(0, w, step):
            if y < h and x < w:
                row += "█" if matrix[y, x] else " "
        print(row)


def print_heatmap(matrix: np.ndarray, width: int = 50):
    """Print a signed matrix as ASCII heatmap."""
    h, w = matrix.shape
    step = max(1, w // width)

    # Normalize to 0-9 range
    vmin, vmax = matrix.min(), matrix.max()
    if vmax == vmin:
        normalized = np.zeros_like(matrix)
    else:
        normalized = ((matrix - vmin) / (vmax - vmin) * 9).astype(int)

    chars = " .:-=+*#%@"
    for y in range(0, h, step * 2):
        row = ""
        for x in range(0, w, step):
            if y < h and x < w:
                row += chars[normalized[y, x]]
        print(row)


def demo_basic():
    """Basic signed encode/decode."""
    print("=" * 60)
    print("SIGNED ACCUMULATION - BASIC")
    print("=" * 60)

    qr_seed = "secret-key-123"
    payload = b"Hidden message in the depths!"
    n_frames = 16
    frame_shape = (64, 64)

    print(f"\nQR seed: {qr_seed}")
    print(f"Payload: {payload}")
    print(f"Payload length: {len(payload)} bytes")
    print(f"Frames: {n_frames}")
    print(f"Frame shape: {frame_shape}")

    # Encode
    print("\nEncoding...")
    frames = encode(qr_seed, payload, n_frames, frame_shape)

    print(f"\nGenerated {len(frames)} frames")
    print(f"Frame dtype: {frames[0].dtype}")
    print(f"Frame values: {np.unique(frames[0])}")  # Should be -1 and 1

    # Show a frame (convert -1/+1 to 0/1 for display)
    print("\nSample frame (frame 0, as binary):")
    display = (frames[0] > 0).astype(np.uint8)
    print_matrix(display)

    # Accumulate
    print("\nAccumulating...")
    acc = accumulate(frames)
    print(f"Accumulated range: [{acc.min()}, {acc.max()}]")

    # Show accumulated as heatmap
    print("\nAccumulated (heatmap, light=positive, dark=negative):")
    print_heatmap(acc)

    # Extract QR from sign
    print("\nExtracting QR from signs...")
    qr_matrix = extract_qr(acc)
    print("\nQR code:")
    print_matrix(qr_matrix)

    # Scan QR
    scanned = scan_qr(qr_matrix)
    if scanned:
        print(f"\n✓ QR scanned: {scanned}")
    else:
        print("\n✗ Could not scan QR")
        return

    # Decode payload using QR content as key
    print("\nDecoding payload from magnitudes...")
    decoded_payload = decode_payload(acc, scanned, n_frames, len(payload))

    print(f"\nDecoded payload: {decoded_payload}")

    if decoded_payload == payload:
        print("✓ Payload matches exactly!")
    else:
        # Check how many bytes match
        matches = sum(a == b for a, b in zip(payload, decoded_payload))
        print(f"✗ Payload mismatch: {matches}/{len(payload)} bytes correct")


def demo_streaming():
    """Streaming with multiple messages."""
    print("\n" + "=" * 60)
    print("SIGNED ACCUMULATION - STREAMING")
    print("=" * 60)

    n_frames = 12
    frame_shape = (48, 48)

    messages = [
        ("key-alpha", b"First secret payload"),
        ("key-beta", b"Second hidden message"),
        ("key-gamma", b"Third concealed data!"),
    ]

    # Normalize payload lengths
    max_len = max(len(p) for _, p in messages)
    messages = [(k, p.ljust(max_len, b'\x00')) for k, p in messages]

    print(f"\nFrame shape: {frame_shape}")
    print(f"Frames per cycle: {n_frames}")
    print(f"Messages to send: {len(messages)}")
    for key, payload in messages:
        clean = payload.rstrip(b'\x00')
        print(f"  - QR: '{key}' -> Payload: {clean}")

    encoder = SignedStreamEncoder(n_frames, frame_shape)
    decoder = SignedStreamDecoder(n_frames, expected_payload_length=max_len)

    for key, payload in messages:
        encoder.queue_message(key, payload)

    print(f"\nProcessing {len(messages) * n_frames} frames...")

    decoded = []
    for i in range(len(messages) * n_frames):
        frame = encoder.next_frame()
        result = decoder.push_frame(frame)

        if result:
            qr_content, payload = result
            decoded.append((qr_content, payload))
            clean_payload = payload.rstrip(b'\x00')
            print(f"  Frame {i}: QR='{qr_content}' -> {clean_payload}")

    print(f"\nDecoded {len(decoded)}/{len(messages)} messages")

    # Verify
    all_match = True
    for i, (orig_key, orig_payload) in enumerate(messages):
        if i < len(decoded):
            dec_key, dec_payload = decoded[i]
            key_match = dec_key == orig_key
            payload_match = dec_payload == orig_payload
            if not (key_match and payload_match):
                all_match = False
                print(f"  Message {i}: key={'✓' if key_match else '✗'}, payload={'✓' if payload_match else '✗'}")

    if all_match:
        print("✓ All messages decoded correctly!")


def demo_progressive():
    """Show accumulation converging."""
    print("\n" + "=" * 60)
    print("SIGNED ACCUMULATION - PROGRESSIVE CONVERGENCE")
    print("=" * 60)

    qr_seed = "DEMO"
    payload = b"Hi"
    n_frames = 8
    frame_shape = (37, 37)

    frames = encode(qr_seed, payload, n_frames, frame_shape)

    print(f"\nQR seed: {qr_seed}")
    print(f"Frames: {n_frames}")
    print("\nWatching accumulation converge...\n")

    acc = np.zeros(frame_shape, dtype=np.int32)

    for i, frame in enumerate(frames):
        acc += frame

        # Try to extract QR
        qr_matrix = extract_qr(acc)
        scanned = scan_qr(qr_matrix)

        print(f"After {i + 1} frame(s):")
        print(f"  Accumulation range: [{acc.min():+d}, {acc.max():+d}]")
        print(f"  Mean: {acc.mean():+.2f}")

        if scanned:
            print(f"  QR readable: '{scanned}'")
            print("\n  QR pattern:")
            print_matrix(qr_matrix)
        else:
            print("  QR not yet readable")
            print("\n  Current sign pattern:")
            print_matrix(qr_matrix)

        print()


def demo_capacity():
    """Demonstrate payload capacity."""
    print("\n" + "=" * 60)
    print("SIGNED ACCUMULATION - CAPACITY TEST")
    print("=" * 60)

    frame_shape = (128, 128)
    n_frames = 32
    n_cells = frame_shape[0] * frame_shape[1]

    print(f"\nFrame shape: {frame_shape}")
    print(f"Total cells: {n_cells:,}")
    print(f"Frames: {n_frames}")

    # Test with increasing payload sizes
    test_sizes = [16, 64, 256, 512, 1024]

    for payload_size in test_sizes:
        # Generate random payload
        payload = bytes(np.random.randint(0, 256, payload_size, dtype=np.uint8))
        qr_seed = f"capacity-test-{payload_size}"

        # Encode and decode
        frames = encode(qr_seed, payload, n_frames, frame_shape, signal_strength=5)
        acc = accumulate(frames)

        qr_matrix = extract_qr(acc)
        scanned = scan_qr(qr_matrix)

        if scanned:
            decoded = decode_payload(acc, scanned, n_frames, payload_size)
            errors = sum(a != b for a, b in zip(payload, decoded))
            accuracy = (payload_size - errors) / payload_size * 100
            print(f"  {payload_size:4d} bytes: {accuracy:5.1f}% accuracy ({errors} byte errors)")
        else:
            print(f"  {payload_size:4d} bytes: QR not readable")


if __name__ == "__main__":
    demo_basic()
    demo_streaming()
    demo_progressive()
    demo_capacity()
