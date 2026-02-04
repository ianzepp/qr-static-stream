#!/usr/bin/env python3
"""
Demo: Audio Static Steganography

Embeds QR pattern into audio white noise.
"""

from pathlib import Path
import numpy as np

from qr_static_audio import (
    generate_cover_wav,
    load_cover_wav,
    encode_audio,
    decode_audio,
    save_wav,
    get_audio_stats,
    AudioStreamEncoder,
    AudioStreamDecoder,
    SAMPLE_RATE,
    FRAME_SIZE,
)


COVER_WAV = Path(__file__).parent / "static_cover.wav"
ENCODED_WAV = Path(__file__).parent / "static_encoded.wav"


def ensure_cover_exists():
    """Generate cover WAV if it doesn't exist."""
    if not COVER_WAV.exists():
        print("Generating cover WAV file...")
        generate_cover_wav(COVER_WAV, duration_seconds=30.0, seed=42)
        print()


def demo_basic():
    """Basic audio encoding/decoding demo."""
    print("=" * 70)
    print("AUDIO STATIC - BASIC DEMO")
    print("=" * 70)

    ensure_cover_exists()

    qr_key = "audio-static"
    n_frames = 60
    flip_prob = 0.4

    stats = get_audio_stats(n_frames)
    print(f"\nConfiguration:")
    print(f"  QR key: {qr_key}")
    print(f"  N frames: {n_frames}")
    print(f"  Flip probability: {flip_prob}")
    print(f"  Frame size: {stats['frame_size']} samples ({stats['frame_dim']}x{stats['frame_dim']})")
    print(f"  Samples per decode cycle: {stats['samples_per_cycle']}")
    print(f"  Duration per cycle: {stats['duration_per_cycle_seconds']:.2f}s")

    # Load cover
    print("\nLoading cover WAV...")
    cover_samples, sample_rate = load_cover_wav(COVER_WAV)
    print(f"  Loaded {len(cover_samples)} samples at {sample_rate} Hz")

    # Encode
    print("\nEncoding QR pattern into audio (sign biasing)...")
    encoded_samples = encode_audio(cover_samples, qr_key, n_frames, flip_prob)

    # Save encoded WAV
    save_wav(ENCODED_WAV, encoded_samples, sample_rate)
    print(f"  Saved encoded WAV: {ENCODED_WAV}")

    # Compare samples - show sign flips
    print("\nSample comparison (first 20 samples, showing sign flips):")
    print("  Original   Encoded    Flipped?")
    print("  " + "-" * 40)
    for i in range(20):
        orig = cover_samples[i]
        enc = encoded_samples[i]
        flipped = "FLIP" if np.sign(orig) != np.sign(enc) else ""
        print(f"  {orig:+.4f}    {enc:+.4f}    {flipped}")

    # Decode
    print("\n" + "-" * 70)
    print("DECODING")
    print("-" * 70)

    qr_content, accumulated = decode_audio(encoded_samples, n_frames)
    print(f"\nDecoded QR: {qr_content}")

    if qr_content == qr_key:
        print("[OK] QR matches!")
    else:
        print("[FAIL] QR mismatch")

    # Show accumulated pattern
    print("\nAccumulated pattern (center region):")
    show_accumulated(accumulated)


def demo_streaming():
    """Streaming encoder/decoder demo."""
    print("\n" + "=" * 70)
    print("AUDIO STATIC - STREAMING DEMO")
    print("=" * 70)

    ensure_cover_exists()

    qr_key = "stream-audio"
    n_frames = 60
    flip_prob = 0.4

    # Load cover
    cover_samples, _ = load_cover_wav(COVER_WAV)

    # Create encoder/decoder
    encoder = AudioStreamEncoder(qr_key, flip_prob)
    decoder = AudioStreamDecoder(n_frames)

    stats = get_audio_stats(n_frames)
    samples_per_cycle = stats['samples_per_cycle']

    print(f"\nStreaming {samples_per_cycle * 3} samples (expect 3 decode outputs)...")

    # Stream through
    outputs = []
    for i in range(samples_per_cycle * 3):
        # Encode sample
        encoded = encoder.encode_sample(cover_samples[i % len(cover_samples)])

        # Decode
        result, acc = decoder.push_sample(encoded)
        if result is not None:
            outputs.append((i, result))
            status = "[OK]" if result == qr_key else "[FAIL]"
            print(f"  Sample {i}: {status} QR = {result}")

    print(f"\nTotal outputs: {len(outputs)}")


def demo_signal_vs_noise():
    """Show signal emerging from noise."""
    print("\n" + "=" * 70)
    print("SIGNAL VS NOISE ANALYSIS")
    print("=" * 70)

    ensure_cover_exists()

    qr_key = "signal-test"
    flip_prob = 0.4

    cover_samples, _ = load_cover_wav(COVER_WAV)
    encoded_samples = encode_audio(cover_samples, qr_key, n_frames=60, flip_probability=flip_prob)

    print("\nComparing accumulation at different frame counts:")
    print()

    for n_frames in [10, 20, 40, 60, 80]:
        qr_content, accumulated = decode_audio(encoded_samples, n_frames)

        # Calculate signal strength
        signal_strength = np.abs(accumulated).mean()
        noise_level = accumulated.std()

        status = "[OK]" if qr_content == qr_key else "[FAIL/None]"
        print(f"  N={n_frames:3d}: signal={signal_strength:.2f}, noise={noise_level:.2f}, QR={status}")


def demo_flip_sensitivity():
    """Test different flip probability values."""
    print("\n" + "=" * 70)
    print("FLIP PROBABILITY SENSITIVITY")
    print("=" * 70)

    ensure_cover_exists()

    qr_key = "flip-test"
    n_frames = 60

    cover_samples, _ = load_cover_wav(COVER_WAV)

    print("\nTesting different flip probability values:")
    print()

    for flip_prob in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        encoded = encode_audio(cover_samples, qr_key, n_frames, flip_prob)
        qr_content, accumulated = decode_audio(encoded, n_frames)

        # Calculate SNR
        signal = np.abs(accumulated).mean()
        noise = accumulated.std()
        snr = signal / noise if noise > 0 else float('inf')

        status = "[OK]" if qr_content == qr_key else "[FAIL]"
        print(f"  flip_prob={flip_prob:.1f}: signal={signal:.2f}, SNR={snr:.2f}, {status}")


def demo_audibility():
    """Analyze how audible the encoding is."""
    print("\n" + "=" * 70)
    print("AUDIBILITY ANALYSIS")
    print("=" * 70)

    ensure_cover_exists()

    qr_key = "audible-test"
    flip_prob = 0.4

    cover_samples, _ = load_cover_wav(COVER_WAV)
    encoded_samples = encode_audio(cover_samples, qr_key, n_frames=60, flip_probability=flip_prob)

    # Statistical comparison
    print("\nStatistical comparison:")
    print()
    print("                  Cover       Encoded     Difference")
    print("  " + "-" * 50)

    cover_mean = cover_samples.mean()
    encoded_mean = encoded_samples.mean()
    print(f"  Mean:           {cover_mean:+.6f}   {encoded_mean:+.6f}   {encoded_mean - cover_mean:+.6f}")

    cover_std = cover_samples.std()
    encoded_std = encoded_samples.std()
    print(f"  Std dev:        {cover_std:.6f}   {encoded_std:.6f}   {encoded_std - cover_std:+.6f}")

    cover_rms = np.sqrt(np.mean(cover_samples**2))
    encoded_rms = np.sqrt(np.mean(encoded_samples**2))
    print(f"  RMS:            {cover_rms:.6f}   {encoded_rms:.6f}   {encoded_rms - cover_rms:+.6f}")

    # Sign flip analysis
    sign_flips = np.sum(np.sign(cover_samples) != np.sign(encoded_samples))
    total_samples = len(cover_samples)
    flip_rate = sign_flips / total_samples

    print()
    print(f"  Sign flips: {sign_flips:,} of {total_samples:,} ({flip_rate*100:.1f}%)")
    print(f"  Expected flip rate: ~{flip_prob/2*100:.1f}% (half of wrong-sign samples)")

    # Perceptual notes
    print("\n  Sign flipping preserves magnitude distribution.")
    print("  Audio sounds identical to original white noise.")
    print("  No tonal artifacts introduced.")


def show_accumulated(accumulated: np.ndarray, size: int = 40):
    """Show accumulated pattern as ASCII."""
    h, w = accumulated.shape

    # Normalize
    vmin, vmax = accumulated.min(), accumulated.max()
    if vmax == vmin:
        normalized = np.zeros_like(accumulated)
    else:
        normalized = (accumulated - vmin) / (vmax - vmin)

    # Sample to target size
    step_y = max(1, h // (size // 2))
    step_x = max(1, w // size)

    chars = " .:-=+*#%@"

    for y in range(0, h, step_y):
        row = ""
        for x in range(0, w, step_x):
            idx = int(normalized[y, x] * (len(chars) - 1))
            row += chars[idx]
        print("  " + row)


if __name__ == "__main__":
    demo_basic()
    demo_streaming()
    demo_signal_vs_noise()
    demo_flip_sensitivity()
    demo_audibility()
