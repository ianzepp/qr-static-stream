# QR Static Stream

Steganographic QR codes hidden in accumulated noise frames.

## Core Concept

Each frame looks like random static. Accumulate N frames and a QR code emerges from the noise.

```
Frame 1        Frame 2        Frame 3           Frame N
  ▓░▒█▓          ░▓█▒░          █▒░▓▒             ▒░▓█░
  ▒█░▓▒          █▒▓░█          ░▓█▒▓    ...      ▓█▒░▓
  ░▓█▒░          ▒░▓█▓          ▓▒░█▒             █░▓▒█

                         ↓ accumulate N frames ↓

                           ███████████████
                           ██ ▒▒▒▒▒▒▒ ▒██
                           ██ ██████▒ ▒██
                           ██ ██████▒ ▒██    ← QR code emerges!
                           ██ ██████▒ ▒██
                           ██ ▒▒▒▒▒▒▒ ▒██
                           ███████████████
```

### Layered Keys

**N is only the first key.** Wrong N yields garbage — but knowing N is just the beginning.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer   │  Key              │  Unlocks                                 │
├──────────┼───────────────────┼──────────────────────────────────────────┤
│  0       │  N (frame count)  │  Ability to accumulate correctly         │
│  1       │  QR content       │  The visible message OR a decryption key │
│  2       │  Magnitude data   │  Payload hidden in "how far" from zero   │
│  3       │  L2 QR content    │  Key to decode deeper payload stream     │
└─────────────────────────────────────────────────────────────────────────┘
```

The QR code that emerges can be:
- **Public data** — a message, URL, or identifier visible to anyone who knows N
- **A decryption key** — unlocks the magnitude data (how deep each peak/valley goes)
- **A nested key** — points to another layer of steganography in the L2 stream

Each layer requires the previous layer's key. An attacker who doesn't know N sees noise. One who knows N but not the QR meaning sees a QR code but can't decode the magnitude payload. The deeper you go, the more keys you need.

---

## Encoding Approaches

Four progressively sophisticated approaches are provided:

| Approach | File | Key Features |
|----------|------|--------------|
| Binary XOR | `qr_static.py` | Simplest, XOR-based |
| Analog Grayscale | `qr_static_analog.py` | Payload hidden in magnitude |
| Two-Layer Recursive | `qr_static_layered.py` | Nested steganography |
| Sliding Window | `qr_static_sliding.py` | Smooth carrier, no boundaries |

---

## 1. Binary XOR

The simplest approach. Each frame is binary (0 or 1). XOR all frames to recover the QR.

```
                    ENCODING                              DECODING

  QR Target     Random Frames      Final Frame
  ┌───────┐     ┌───────┐         ┌───────┐
  │█░█░█░█│     │░█░█░░█│         │░░█░░█░│
  │░█████░│  +  │█░█░█░░│   →     │█░░░█░░│        XOR all N frames
  │█░█░█░█│     │░░█░░█░│         │█░░░█░█│             ↓
  └───────┘     └───────┘         └───────┘        ┌───────┐
                Frame 1..N-1       Frame N         │█░█░█░█│
                (random)          (computed)       │░█████░│  ← QR!
                                                   │█░█░█░█│
  Frame N = QR ⊕ Frame₁ ⊕ ... ⊕ Frame_{N-1}        └───────┘
```

### How It Works

```python
# Encoding
frames[0..N-2] = random binary noise
frames[N-1] = QR ⊕ frames[0] ⊕ frames[1] ⊕ ... ⊕ frames[N-2]

# Decoding
QR = frames[0] ⊕ frames[1] ⊕ ... ⊕ frames[N-1]
```

### Usage

```python
from qr_static import generate_qr_matrix, encode, decode, scan_qr

qr = generate_qr_matrix("Hello, world!")
frames = encode(qr, n_frames=8, seed=42)

recovered = decode(frames)
message = scan_qr(recovered)  # "Hello, world!"
```

### Properties
- Binary frames (harsh visual appearance)
- Any frame order works (XOR is commutative)
- No hidden payload capacity

---

## 2. Analog Grayscale

Frames contain continuous grayscale values. Signal accumulates linearly while noise grows as √N, improving SNR.

```
                         SIGNAL ACCUMULATION

    Frame 1          Frame 2          Frame N         Accumulated
   ┌────────┐       ┌────────┐       ┌────────┐       ┌────────┐
   │+.1 -.1 │       │+.1 -.1 │       │+.1 -.1 │       │+N  -N  │
   │-.1 +.1 │   +   │-.1 +.1 │  ...  │-.1 +.1 │   =   │-N  +N  │
   │+.1 -.1 │       │+.1 -.1 │       │+.1 -.1 │       │+N  -N  │
   └────────┘       └────────┘       └────────┘       └────────┘
    + noise          + noise          + noise         signal >> noise

                    Signal: grows as N
                    Noise:  grows as √N
                    SNR:    improves as √N
```

### Hidden Payload in Magnitude

The QR emerges from the **sign** of accumulated values. But the **magnitude** (how far from zero) can encode additional data:

```
                    DUAL-CHANNEL ENCODING

                      Accumulated Height Field

        Sign reveals QR:              Magnitude hides payload:

          + + - - + +                   2.1  2.3  1.8  1.9  2.2  2.0
          + - - - - +                   2.0  1.7  2.1  1.8  1.9  2.2
          - - + + - -        →          1.9  2.0  2.4  2.1  1.8  1.7
          - - + + - -                   2.1  1.8  2.0  2.3  2.0  1.9
          + - - - - +                   2.2  1.9  1.7  2.0  1.8  2.1
          + + - - + +                   2.0  2.1  1.9  1.8  2.2  2.0

        Threshold at 0 → QR            Decode with QR key → payload
```

### Usage

```python
from qr_static_analog import encode, accumulate, extract_qr, scan_qr, decode_payload

qr_seed = "secret-key-123"
payload = b"Hidden message in the depths!"

frames = encode(qr_seed, payload, n_frames=16, frame_shape=(64, 64))
height_field = accumulate(frames)

qr_matrix = extract_qr(height_field)
scanned_key = scan_qr(qr_matrix)  # "secret-key-123"

hidden = decode_payload(height_field, scanned_key, n_frames=16, payload_length=len(payload))
# b"Hidden message in the depths!"
```

### Properties
- Grayscale frames (natural TV static appearance)
- QR in sign, payload in magnitude
- ~3KB QR capacity + large payload in magnitude

---

## 3. Two-Layer Recursive

A video within a video. Layer 1 outputs become the "frames" for Layer 2.

```
                         RECURSIVE STRUCTURE

    Carrier Frames (N₁ = 30)              Layer 1 Output
    ┌──┬──┬──┬──┬──┬──┬──┬──┐            ┌────────────┐
    │▓▒│░▓│█▒│▒░│▓█│░▒│▓░│...│  accumulate  │            │
    └──┴──┴──┴──┴──┴──┴──┴──┘      →     │   QR₁      │
         30 noise frames                  │            │
                                          └────────────┘

    Layer 1 Outputs (N₂ = 30)             Layer 2 Output
    ┌────┬────┬────┬────┬────┐           ┌────────────┐
    │QR₁ │QR₁ │QR₁ │QR₁ │... │  accumulate  │            │
    │out₁│out₂│out₃│out₄│    │      →     │ QR₂ + data │
    └────┴────┴────┴────┴────┘           │            │
         30 L1 outputs                    └────────────┘

    Total: 30 × 30 = 900 carrier frames per L2 output
```

### Four Secrets Required

```
┌─────────────────────────────────────────────────────────────┐
│  Secret    │  Purpose                                       │
├────────────┼────────────────────────────────────────────────┤
│  N₁        │  Carrier frames per L1 output                  │
│  N₂        │  L1 outputs per L2 output                      │
│  QR₁       │  Validates L1 structure                        │
│  QR₂       │  Key to decode L2 payload                      │
└─────────────────────────────────────────────────────────────┘
```

### Usage

```python
from qr_static_layered import encode, decode

layer1_key = "visible-qr"
layer2_key = "hidden-qr"
payload = b"Deep secret message!"
n1, n2 = 30, 30

# Encode: generates n1 * n2 = 900 carrier frames
frames = encode(layer1_key, layer2_key, payload, (64, 64), n1, n2)

# Decode: requires knowing n1, n2, and payload length
l1_qr, l2_qr, decoded_payload = decode(frames, n1, n2, len(payload))
```

### Capacity at 1080p (30fps)

| Metric | Value |
|--------|-------|
| L1 emerges | Every 1 second |
| L2 emerges | Every 30 seconds |
| Payload per L2 | ~250 KB |
| 10-min video | ~5 MB hidden |

---

## 4. Sliding Window

The most sophisticated approach. Overlapping windows create smooth carrier video with no detectable boundaries.

```
                       SLIDING WINDOW (50% overlap)

    Frame index:  0         30         60         90        120
                  │          │          │          │          │
    Window A:     ├──────────────────────┤
                  │◄───── N=60 frames ──►│

    Window B:                ├──────────────────────┤
                             │◄───── N=60 frames ──►│

    Window C:                           ├──────────────────────┤
                                        │◄───── N=60 frames ──►│

    ──────────────────────────────────────────────────────────────────►
                                                                   time

    Window A: frames 0-59      ─┬─ 30 frames overlap (50%)
    Window B: frames 30-89     ─┘─┬─ 30 frames overlap (50%)
    Window C: frames 60-119       ─┘

    Each window decodes to the same QR.
    Decoder can lock on at ANY frame — no fixed boundaries to detect.
```

### Why Sliding Windows?

Fixed boundaries create detectable patterns:

```
    FIXED WINDOWS (detectable)          SLIDING WINDOWS (smooth)

    │←── N ──→│←── N ──→│               ════════════════════════
    ▓▓▓▓▓▓▓▓▓▓│░░░░░░░░░░│              Continuous, uniform carrier
              ↑                         No statistical discontinuity
         Boundary creates               Decoder locks on anywhere
         statistical artifact
```

### Composable L1 + L2

L1 and L2 signals are additive. Generate L1 first, overlay L2 independently:

```
                        SIGNAL COMPOSITION

    L1 carrier    +    L2 overlay    =    Final frame
    ┌──────────┐      ┌──────────┐       ┌──────────┐
    │  QR₁/N₁  │      │ QR₂/N₁N₂ │       │  Combined │
    │  signal  │  +   │  signal  │   =   │   signal  │
    │  + noise │      │  + noise │       │  + noise  │
    └──────────┘      └──────────┘       └──────────┘

    L1: Provides smooth carrier, QR₁ readable from any N₁ consecutive frames
    L2: Tiny per-frame shift, accumulates over N₁×N₂ frames to reveal QR₂
```

### Usage

```python
from qr_static_sliding import (
    encode_l1_sliding,
    apply_l2_overlay,
    decode_l1_at_offset,
    decode_l2,
)

# L1 only
l1_key = "carrier-qr"
frames = encode_l1_sliding(l1_key, (64, 64), total_frames=300, n1=60, stride=30)

# Decode from ANY offset
accumulated, qr = decode_l1_at_offset(frames, start=47, n1=60, qr_key=l1_key)
# qr == "carrier-qr" (works from any starting position!)

# With L2 overlay
l2_key = "hidden-qr"
payload = b"Secret payload"
frames_with_l2 = apply_l2_overlay(frames, l2_key, payload, n1=60, stride=30, n2=10)

# L1 still works
_, l1_qr = decode_l1_at_offset(frames_with_l2, 0, 60, l1_key)

# L2 decodes from accumulated L1 outputs
l2_qr, decoded_payload = decode_l2(frames_with_l2, l1_key, n1=60, n2=10, payload_length=len(payload))
```

### Streaming API

```python
from qr_static_sliding import SlidingWindowEncoder, SlidingWindowDecoder

encoder = SlidingWindowEncoder((64, 64), "stream-key", n1=60, stride=30)
decoder = SlidingWindowDecoder(n1=60, stride=30)

for i in range(150):
    frame = encoder.next_frame()
    qr_result, accumulated = decoder.push_frame(frame)
    if qr_result:
        print(f"Frame {i}: QR decoded = {qr_result}")
```

### Properties
- No detectable window boundaries
- Decode from any frame offset
- 50% overlap means each frame contributes to 2 windows
- L1 + L2 independently composable

---

## Signal vs Noise Theory

All analog approaches rely on signal accumulating faster than noise:

```
Frames:        1        4       16       64      256
              ───      ───      ───      ───      ───
Signal sum:    S       4S      16S      64S     256S    (linear)
Noise sum:     σ       2σ       4σ       8σ      16σ    (√N)
              ───      ───      ───      ───      ───
SNR:           1        2        4        8       16    (√N improvement)


             │
     Signal  │                                    ****
       or    │                               *****
     Noise   │                          *****
             │                     *****
             │               ******
             │         ******                  ←── Signal (linear)
             │    *****
             │ ***
             │**     ════════════════════════ ←── Noise (√N)
             └────────────────────────────────────────
                              N frames
```

More frames = cleaner QR emergence = more reliable decoding.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and OpenCV.

## Demos

```bash
python demo.py           # Binary XOR
python demo_analog.py    # Analog grayscale
python demo_layered.py   # Two-layer recursive
python demo_sliding.py   # Sliding window
```

## Properties

- Individual frames indistinguishable from random noise
- N is a key — wrong N yields garbage
- QR error correction provides resilience
- Analog versions: magnitude carries hidden payload
- Sliding window: no detectable boundaries in carrier

## License

MIT
