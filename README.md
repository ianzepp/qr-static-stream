# QR Static Stream

Steganographic QR codes hidden in accumulated noise frames.

## Concept

Each frame looks like random static. Accumulate N frames and a QR code emerges from the noise.

Two algorithms are provided:

### Binary XOR (`qr_static.py`)

```
Frame 0:  ░▒█░▒░█▒░   (binary noise)
Frame 1:  █░▒█░▒░█▒   (binary noise)
...
Frame N:  ░█▒░█▒█░▒   (binary noise)
─────────────────────
XOR all:  ███████     (QR code!)
          █ ███ █
          ███████
```

### Analog Grayscale (`qr_static_analog.py`)

```
Frame 0:  ▓░▒▓░▒█▒░   (grayscale noise + faint signal)
Frame 1:  ░▓▒░▓▒░▓▒   (grayscale noise + faint signal)
...
Frame N:  ▒░▓▒░▓▒░▓   (grayscale noise + faint signal)
─────────────────────
Sum all:  ████████    ← peaks (positive)
             ░░░░     ← valleys (negative)
          ████████
```

The analog version models a **gravity well**: each frame applies continuous pressure (positive or negative) at each point. After N frames, the accumulated "height field" reveals the QR pattern through the difference between peaks and valleys.

## Three-Layer Steganography (Analog)

```
Layer 0: Know N         → can accumulate frames
Layer 1: Accumulate     → QR emerges from sign of height field
Layer 2: Scan QR        → get decoding key
Layer 3: Apply key      → decode payload from magnitude (depth)
```

The QR code capacity is ~3KB. But the magnitude data (how deep each peak/valley is) can store much more. The QR acts as a header containing the key to unlock the larger payload hidden in the depth information.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and OpenCV.

## Usage

### Binary XOR

```python
from qr_static import generate_qr_matrix, encode, decode, scan_qr

qr = generate_qr_matrix("Hello, world!")
frames = encode(qr, n_frames=8, seed=42)

# Each frame is binary noise
# XOR them all...
recovered = decode(frames)
message = scan_qr(recovered)  # "Hello, world!"
```

### Analog Grayscale

```python
from qr_static_analog import encode, accumulate, extract_qr, scan_qr, decode_payload

# QR seed becomes the visible QR code AND the key for payload decoding
qr_seed = "secret-key-123"
payload = b"Hidden message in the depths!"

# Encode into grayscale noise frames
frames = encode(qr_seed, payload, n_frames=16, frame_shape=(64, 64))

# Each frame looks like grayscale static
# Sum them all...
height_field = accumulate(frames)

# Threshold to get QR
qr_matrix = extract_qr(height_field)
scanned_key = scan_qr(qr_matrix)  # "secret-key-123"

# Use the key to decode the hidden payload from magnitudes
hidden = decode_payload(height_field, scanned_key, n_frames=16, payload_length=len(payload))
# b"Hidden message in the depths!"
```

### Streaming

```python
from qr_static_analog import AnalogStreamEncoder, AnalogStreamDecoder

encoder = AnalogStreamEncoder(n_frames=12, frame_shape=(48, 48))
encoder.queue_message("key-alpha", b"First hidden payload")
encoder.queue_message("key-beta", b"Second hidden payload")

decoder = AnalogStreamDecoder(n_frames=12, expected_payload_length=20)

for i in range(24):
    frame = encoder.next_frame()
    result = decoder.push_frame(frame)
    if result:
        qr_content, payload = result
        print(f"QR: {qr_content}, Payload: {payload}")
```

## Demos

```bash
python demo.py          # Binary XOR demo
python demo_analog.py   # Analog grayscale demo
```

## How It Works

### Binary XOR

1. Generate target QR code as binary matrix
2. Generate N-1 frames of random binary noise
3. Compute frame N as: `QR XOR frame[0] XOR ... XOR frame[N-1]`
4. XOR of all N frames equals original QR

### Analog Grayscale

1. Decide target sum at each pixel (sign = QR bit, magnitude = payload)
2. Each frame contributes `target/N + noise`
3. Signal accumulates as N, noise grows as √N
4. After N frames: SNR improved by √N, QR readable from sign
5. Payload decoded from magnitude using QR content as key

### Signal vs Noise

```
Frames:     1      4      16     64
Signal:     S      S      S      S     (constant per frame)
Noise σ:    σ      σ      σ      σ     (constant per frame)
──────────────────────────────────────
Sum signal: S      4S     16S    64S   (linear growth)
Sum noise:  σ      2σ     4σ     8σ    (√N growth)
SNR:        1      2      4      8     (√N improvement)
```

More frames = cleaner QR emergence.

## Properties

- Individual frames are indistinguishable from random noise
- N is a key — wrong N yields garbage
- QR error correction provides resilience
- Analog version: magnitude carries hidden payload (steganography²)
- Analog version: grayscale looks more like real TV static

## License

MIT
