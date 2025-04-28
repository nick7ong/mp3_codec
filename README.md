# MUE610 Project 2:
### Psychoacoustic Ear Model II utilized in MPEG-1 layer III (MP3) as described in ISO/IEC 11172-3

## Project Structure
```text
mp3_codec/
├── ear_model.py      # contains the layer 3 ear model and visualizations.
├── quantizer.py      # uses the ear_model to simulate audio compression.
├── utils.py          # contains psychoacoustic utility and helper functions.
├── audio/            # audio directory for test samples
```

### How To Run
```bash
# For psychoacoustic model simulation
python ear_model.py audio/<input>.wav --verbose

# For simulated perceptual audio compression/quantization
python quantizer.py audio/<input>.wav --output audio/<output>.wav --smr 0.0 --visualize
```

## Environment Setup
Setup instructions using `Python v3.11`, `.venv` and `requirements.txt`.

### 1. Clone the Repository
```bash
git clone git@github.com:nick7ong/mp3_codec.git
cd mp3_codec
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Activate venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
