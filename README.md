# QR Static Stream

Steganographic QR codes hidden in XOR'd noise frames.

## Concept

Each frame looks like random TV static. XOR all N frames together and a QR code appears.

```
Frame 0:  ░▒█░▒░█▒░   (noise)
Frame 1:  █░▒█░▒░█▒   (noise)
Frame 2:  ▒░█░▒█▒░█   (noise)
...
Frame N:  ░█▒░█▒█░▒   (noise)
─────────────────────
XOR all:  ███████     (QR code!)
          █ ███ █
          ███████
```

N is effectively a key — wrong N yields garbage.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

## Usage

### Basic encode/decode

```python
from qr_static import generate_qr_matrix, encode, decode, scan_qr

# Create QR matrix from data
qr = generate_qr_matrix("Hello, world!")

# Encode into 8 frames of noise
frames = encode(qr, n_frames=8, seed=42)

# Each frame looks like random noise
# But XOR them all together...
recovered = decode(frames)

# ...and you get the QR back
message = scan_qr(recovered)  # "Hello, world!"
```

### Streaming

```python
from qr_static import StreamEncoder, StreamDecoder

# Encoder generates continuous noise frames
encoder = StreamEncoder(n_frames=10, frame_shape=(64, 64))
encoder.queue_message("First message")
encoder.queue_message("Second message")

# Get frames one at a time
frame = encoder.next_frame()  # noise
frame = encoder.next_frame()  # noise
# ... after 10 frames, XOR reveals "First message"
# ... next 10 frames reveal "Second message"

# Decoder accumulates and scans
decoder = StreamDecoder(n_frames=10)
for i in range(20):
    frame = encoder.next_frame()
    result = decoder.push_frame(frame)
    if result:
        print(f"Decoded: {result}")
```

## Demo

```bash
python demo.py
```

Runs three demos:
1. **Basic** — encode message, XOR frames, scan QR
2. **Streaming** — multiple messages through encoder/decoder
3. **Progressive** — visualize noise converging to QR frame by frame

## How it works

1. Generate target QR code as binary matrix
2. Generate N-1 frames of random binary noise
3. Compute frame N as: `QR XOR frame[0] XOR frame[1] XOR ... XOR frame[N-1]`
4. Result: N frames of apparent noise, but `XOR(all frames) = QR`

Properties:
- Individual frames are indistinguishable from random noise
- QR error correction provides resilience to minor corruption
- Different messages can be embedded in successive N-frame cycles
- Cycle length N acts as a simple key

## License

MIT
