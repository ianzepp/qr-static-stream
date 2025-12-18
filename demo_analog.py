#!/usr/bin/env python3
"""
Demo: Analog/continuous grayscale steganography.

Frames are continuous grayscale (floats), not binary.
Signal accumulates linearly, noise grows as sqrt(N).
QR emerges from thresholding the accumulated height field.
"""

from qr_static_analog import (
    encode,
    accumulate,
    extract_qr,
    decode_payload,
    scan_qr,
    AnalogStreamEncoder,
    AnalogStreamDecoder,
    height_field_to_image,
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

    # ASCII grayscale ramp (dark to light)
    chars = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    n_chars = len(chars)

    for y in range(0, h, step * 2):
        row = ""
        for x in range(0, w, step):
            if y < h and x < w:
                idx = int(normalized[y, x] * (n_chars - 1))
                row += chars[idx]
        print(row)


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


def demo_basic():
    """Basic analog encode/decode."""
    print("=" * 70)
    print("ANALOG GRAYSCALE - BASIC ENCODE/DECODE")
    print("=" * 70)

    qr_seed = "analog-key-42"
    payload = b"Hidden in the grayscale depths!"
    n_frames = 16
    frame_shape = (64, 64)

    print(f"\nQR seed: {qr_seed}")
    print(f"Payload: {payload}")
    print(f"Frames: {n_frames}")
    print(f"Frame shape: {frame_shape}")

    # Encode
    print("\nEncoding...")
    frames = encode(qr_seed, payload, n_frames, frame_shape)

    print(f"\nGenerated {len(frames)} frames")
    print(f"Frame dtype: {frames[0].dtype}")
    print(f"Frame value range: [{frames[0].min():.3f}, {frames[0].max():.3f}]")

    # Show a frame as grayscale
    print("\nSample frame (grayscale static):")
    print_grayscale(frames[0])

    # Show another frame - should look different but similar distribution
    print("\nAnother frame (frame 5):")
    print_grayscale(frames[5])

    # Accumulate progressively
    print("\n" + "-" * 70)
    print("PROGRESSIVE ACCUMULATION")
    print("-" * 70)

    acc = np.zeros(frame_shape, dtype=np.float32)
    checkpoints = [1, 4, 8, 12, 16]

    for i, frame in enumerate(frames):
        acc += frame

        if (i + 1) in checkpoints:
            qr_matrix = extract_qr(acc)
            scanned = scan_qr(qr_matrix)

            print(f"\nAfter {i + 1} frames:")
            print(f"  Accumulation range: [{acc.min():.2f}, {acc.max():.2f}]")
            print(f"  QR readable: {scanned if scanned else 'No'}")

            if i + 1 == n_frames:
                print("\n  Final height field:")
                print_grayscale(acc)
                print("\n  Thresholded (QR pattern):")
                print_binary(qr_matrix)

    # Final decode
    print("\n" + "-" * 70)
    print("DECODING")
    print("-" * 70)

    qr_matrix = extract_qr(acc)
    scanned = scan_qr(qr_matrix)

    if scanned:
        print(f"\nQR scanned: {scanned}")

        decoded = decode_payload(acc, scanned, n_frames, len(payload))
        print(f"Payload decoded: {decoded}")

        if decoded == payload:
            print("\n✓ Payload matches exactly!")
        else:
            matches = sum(a == b for a, b in zip(payload, decoded))
            print(f"✗ Partial match: {matches}/{len(payload)} bytes")
    else:
        print("\n✗ Could not scan QR")


def demo_streaming():
    """Streaming with multiple messages."""
    print("\n" + "=" * 70)
    print("ANALOG GRAYSCALE - STREAMING")
    print("=" * 70)

    n_frames = 12
    frame_shape = (48, 48)

    messages = [
        ("key-one", b"First analog message!"),
        ("key-two", b"Second hidden payload"),
        ("key-three", b"Third secret content"),
    ]

    max_len = max(len(p) for _, p in messages)
    messages = [(k, p.ljust(max_len, b'\x00')) for k, p in messages]

    print(f"\nFrame shape: {frame_shape}")
    print(f"Frames per cycle: {n_frames}")
    print(f"Messages: {len(messages)}")

    encoder = AnalogStreamEncoder(n_frames, frame_shape)
    decoder = AnalogStreamDecoder(n_frames, expected_payload_length=max_len)

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
            clean = payload.rstrip(b'\x00')
            print(f"  Frame {i}: QR='{qr_content}' -> {clean}")

    print(f"\nDecoded {len(decoded)}/{len(messages)} messages")

    all_match = True
    for i, (orig_key, orig_payload) in enumerate(messages):
        if i < len(decoded):
            dec_key, dec_payload = decoded[i]
            if dec_key != orig_key or dec_payload != orig_payload:
                all_match = False

    if all_match and len(decoded) == len(messages):
        print("✓ All messages decoded correctly!")


def demo_snr():
    """Demonstrate signal-to-noise ratio improvement."""
    print("\n" + "=" * 70)
    print("ANALOG GRAYSCALE - SIGNAL VS NOISE")
    print("=" * 70)

    qr_seed = "snr-demo"
    payload = b"X"
    frame_shape = (37, 37)

    print("\nDemonstrating how signal accumulates faster than noise...")
    print("With weak signal (1.5) and strong noise (0.5), more frames needed.\n")

    for n_frames in [4, 16, 64]:
        frames = encode(qr_seed, payload, n_frames, frame_shape,
                       noise_amplitude=0.5, signal_strength=1.5)

        acc = accumulate(frames)
        qr_matrix = extract_qr(acc)
        scanned = scan_qr(qr_matrix)

        # Measure actual SNR
        qr_base = extract_qr(np.zeros(frame_shape))  # baseline
        expected_magnitude = 1.5  # signal_strength

        positive_cells = acc[acc > 0]
        negative_cells = acc[acc < 0]

        print(f"N = {n_frames:2d} frames:")
        print(f"  Value range: [{acc.min():+.2f}, {acc.max():+.2f}]")
        print(f"  Mean |value|: {np.abs(acc).mean():.2f}")
        print(f"  Expected signal: ~{expected_magnitude:.1f}")
        print(f"  QR readable: {'Yes - ' + scanned if scanned else 'No'}")
        print()


def demo_visual():
    """Show what the static looks like."""
    print("\n" + "=" * 70)
    print("ANALOG GRAYSCALE - VISUAL COMPARISON")
    print("=" * 70)

    qr_seed = "visual"
    payload = b"See the signal emerge"
    n_frames = 20
    frame_shape = (50, 50)

    frames = encode(qr_seed, payload, n_frames, frame_shape)

    print("\nSingle frame (pure noise appearance):")
    print("-" * 50)
    print_grayscale(frames[0], width=50)

    print("\nAccumulated 5 frames (signal starting to emerge):")
    print("-" * 50)
    acc5 = accumulate(frames[:5])
    print_grayscale(acc5, width=50)

    print("\nAccumulated 10 frames (pattern visible):")
    print("-" * 50)
    acc10 = accumulate(frames[:10])
    print_grayscale(acc10, width=50)

    print("\nAccumulated all 20 frames (clear height field):")
    print("-" * 50)
    acc20 = accumulate(frames)
    print_grayscale(acc20, width=50)

    print("\nThresholded (QR code):")
    print("-" * 50)
    qr = extract_qr(acc20)
    print_binary(qr, width=50)

    scanned = scan_qr(qr)
    print(f"\nScanned: {scanned}")


if __name__ == "__main__":
    demo_basic()
    demo_streaming()
    demo_snr()
    demo_visual()
