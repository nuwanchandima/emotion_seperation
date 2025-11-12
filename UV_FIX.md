# UV Installation Quick Fix

## Your Error
```
× No solution found when resolving dependencies for split
╰─▶ Because only retinaface-pytorch<=0.0.8 is available and your project 
    depends on retinaface-pytorch>=0.0.10, we can conclude that your 
    project's requirements are unsatisfiable.
```

## ✅ Solution (FIXED)

The issue has been **fixed** in the latest files:
- `requirements.txt` now uses `retinaface-pytorch>=0.0.7` (compatible)
- `pyproject.toml` added with `requires-python = ">=3.8,<3.13"`

## 🚀 Installation Commands (Linux/Server)

### Method 1: UV with sync (Recommended)
```bash
cd /var/www/spera_AI/6_emotion_seperation/emotion_seperation

# Pull latest fixed requirements.txt
git pull origin main

# Create venv with Python 3.11
uv venv --python 3.11

# Activate
source .venv/bin/activate

# Install dependencies
uv pip sync requirements.txt
```

### Method 2: Traditional pip (Most Reliable)
```bash
cd /var/www/spera_AI/6_emotion_seperation/emotion_seperation

# Create venv
python3.11 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Method 3: UV with frozen (If sync fails)
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt --frozen
```

## 🔍 Verify Installation

```bash
# Check Python version (should be 3.8-3.12)
python --version

# Run setup check
python setup_check.py

# You should see:
# ✓ Python version: 3.11.x
# ✓ FFmpeg: x.x.x
# ✓ torch: x.x.x
# ✓ All dependencies installed!
```

## 📦 What Changed

### requirements.txt
```diff
- retinaface-pytorch>=0.0.10  # ❌ This version doesn't exist
+ retinaface-pytorch>=0.0.7   # ✅ Maximum available version
```

### pyproject.toml (NEW)
```toml
[project]
name = "emotion-seperation"
requires-python = ">=3.8,<3.13"  # ✅ Prevents Python 3.13 issues
dependencies = [
    "retinaface-pytorch>=0.0.7",
    # ... all other deps
]
```

## 🐛 If You Still Have Issues

### Issue: "retinaface-pytorch not found"
```bash
# Option A: Use MediaPipe instead
pip install mediapipe>=0.10.0

# Then in config.yaml, set:
# face_detection:
#   detector_type: "mediapipe"
```

### Issue: "CUDA not available"
```bash
# Install PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "pyannote.audio fails"
```bash
# 1. Get HuggingFace token
# Visit: https://huggingface.co/settings/tokens

# 2. Set environment variable
export HF_TOKEN="your_token_here"

# 3. Accept model terms
# Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
```

## 📚 Full Documentation

- **Installation Guide**: [`INSTALL.md`](INSTALL.md) - Complete installation instructions
- **Quick Start**: [`QUICKSTART.md`](QUICKSTART.md) - 5-minute tutorial
- **Troubleshooting**: [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) - Detailed problem solving

## ✅ Next Steps After Installation

```bash
# 1. Configure HuggingFace token
export HF_TOKEN="your_token_here"

# 2. Test installation
python setup_check.py

# 3. Run pipeline test
python test_pipeline.py

# 4. Process a video
python src/pipeline.py sample.mp4
```

---

**TL;DR**: Pull the latest code with fixed `requirements.txt`, use Python 3.11, and install with `pip install -r requirements.txt` for most reliable results.
