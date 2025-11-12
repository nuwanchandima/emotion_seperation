# Installation Guide for UV Package Manager

This project supports installation with **uv** (ultra-fast Python package installer) or traditional pip.

## Option 1: Install with UV (Recommended for Linux/macOS)

### Step 1: Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Create Virtual Environment
```bash
# Create venv with Python 3.11
uv venv --python 3.11

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
# Install from pyproject.toml (recommended)
uv pip sync requirements.txt

# Or install project in editable mode
uv pip install -e .
```

### Troubleshooting UV Installation

#### Issue: "No solution found" for retinaface-pytorch
**Solution**: The error occurs because `retinaface-pytorch>=0.0.10` doesn't exist. This has been fixed in the latest `requirements.txt` (changed to `>=0.0.7`).

If you still see this error:
```bash
# Use the --frozen flag to skip resolution
uv pip install -r requirements.txt --frozen

# Or install without strict resolution
uv pip install -r requirements.txt --no-deps
pip install -r requirements.txt  # Then use pip for dependencies
```

#### Issue: Python version conflicts (3.13 markers)
**Solution**: The project now specifies `requires-python = ">=3.8,<3.13"` in `pyproject.toml`.

```bash
# Ensure you're using Python 3.8-3.12
python --version  # Should show 3.8.x - 3.12.x

# Create venv with specific version
uv venv --python 3.11
```

#### Issue: Platform-specific packages fail on macOS/Darwin
**Solution**: Some packages (like `retinaface-pytorch`) may have limited platform support.

Alternative face detection approaches:
```bash
# Option 1: Skip retinaface-pytorch, use MediaPipe instead
uv pip install -r requirements.txt --no-deps
uv pip install mediapipe>=0.10.0

# Option 2: Use YOLOv8 for face detection
uv pip install ultralytics>=8.0.0
```

---

## Option 2: Install with PIP (Traditional Method)

### Step 1: Create Virtual Environment
```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Option 3: Install with Conda

```bash
# Create conda environment
conda create -n emotion_sep python=3.11 -y
conda activate emotion_sep

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

---

## Verify Installation

After installation with any method, run the setup check:

```bash
python setup_check.py
```

**Expected output:**
```
✓ Python version: 3.11.0
✓ FFmpeg: 4.4.2
✓ torch: 2.1.0 (CUDA available)
✓ opencv-python: 4.8.1
✓ librosa: 0.10.1
✓ All dependencies installed!
```

---

## Post-Installation Steps

### 1. Configure Hugging Face Token (for pyannote)
```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_token_here"

# Or add to config.yaml
diarization:
  hf_token: "your_token_here"
```

### 2. Accept pyannote model terms
Visit these pages and accept terms:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### 3. Test the pipeline
```bash
# Run tests
python test_pipeline.py

# Process a sample video
python src/pipeline.py sample.mp4
```

---

## Dependency Notes

### Core Requirements
- **Python**: 3.8 - 3.12 (NOT 3.13+ due to package compatibility)
- **FFmpeg**: 4.0+ (required for video processing)
- **CUDA**: 11.7+ (optional, for GPU acceleration)

### Optional Dependencies

#### For GPU Acceleration
```bash
# NVIDIA CUDA Toolkit
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Check CUDA version
nvcc --version
```

#### For Advanced Face Detection
```bash
# MediaPipe (alternative to RetinaFace)
pip install mediapipe>=0.10.0

# YOLOv8 (alternative face detector)
pip install ultralytics>=8.0.0
```

#### For Development
```bash
uv pip install -e ".[dev]"
```

---

## Common Installation Issues

### 1. `retinaface-pytorch` fails to install
**Symptoms**: `No matching distribution found for retinaface-pytorch>=0.0.10`

**Solutions**:
```bash
# A) Use lower version (already in requirements.txt)
pip install retinaface-pytorch==0.0.7

# B) Use alternative face detector
pip install mediapipe>=0.10.0
# Then set in config.yaml: face_detection.detector_type: "mediapipe"
```

### 2. `insightface` fails on macOS
**Symptoms**: `error: command 'clang' failed`

**Solutions**:
```bash
# Install system dependencies
brew install cmake openblas

# Install with specific flags
CFLAGS="-I/usr/local/opt/openblas/include" pip install insightface
```

### 3. `pyannote.audio` model download fails
**Symptoms**: `403 Forbidden` or authentication error

**Solutions**:
```bash
# Set Hugging Face token
export HF_TOKEN="your_token_here"

# Accept model terms on Hugging Face website
# Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
```

### 4. `torch` CUDA not available
**Symptoms**: `torch.cuda.is_available()` returns `False`

**Solutions**:
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 5. `lap` package fails to install
**Symptoms**: `error: Microsoft Visual C++ 14.0 is required` (Windows)

**Solutions**:
```bash
# Windows: Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or use pre-built wheel
pip install lap --prefer-binary
```

---

## Platform-Specific Instructions

### Ubuntu/Debian Linux
```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-dev python3-pip python3-venv
sudo apt install -y ffmpeg libsm6 libxext6 libxrender-dev
sudo apt install -y build-essential cmake

# Install project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### macOS
```bash
# Install Homebrew dependencies
brew install python@3.11 ffmpeg cmake

# Install project
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows
```powershell
# Install FFmpeg (using Chocolatey)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
# Add to PATH: C:\ffmpeg\bin

# Install project
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Performance Optimization

### Install with CUDA Support
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install with CPU-only (smaller download)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Minimal Installation (Core Features Only)

If you want to install only essential packages:

```bash
# Core processing
pip install torch torchvision torchaudio
pip install opencv-python ffmpeg-python
pip install librosa soundfile
pip install numpy scipy scikit-learn
pip install pyyaml tqdm

# Face detection (choose one)
pip install mediapipe  # Easiest
# OR
pip install facenet-pytorch  # Better accuracy

# Speaker diarization
pip install pyannote.audio pyannote.core

# Emotion detection
pip install transformers
```

---

## Next Steps

After successful installation:

1. **Configure**: Edit `config.yaml` with your settings
2. **Test**: Run `python setup_check.py`
3. **Quick Start**: Follow `QUICKSTART.md`
4. **Process Video**: `python src/pipeline.py your_video.mp4`

For more help, see:
- `README.md` - Full documentation
- `TROUBLESHOOTING.md` - Detailed problem solving
- `QUICKSTART.md` - 5-minute tutorial
