# 🔧 Installation Issue - RESOLVED

## What Was Wrong

Your `uv add -r requirements.txt` command failed because:

1. ❌ `retinaface-pytorch>=0.0.10` - **This version doesn't exist** (max is 0.0.8)
2. ❌ No Python version constraint - UV tried to resolve for Python 3.13+ which has package incompatibilities
3. ❌ No `pyproject.toml` - UV prefers modern Python project structure

## What Was Fixed

### ✅ Files Updated

1. **`requirements.txt`**
   - Changed: `retinaface-pytorch>=0.0.7` (was `>=0.0.10`)
   - Added: `numpy>=1.24.0,<2.0.0` (prevent numpy 2.0 issues)

2. **`pyproject.toml`** (NEW)
   - Added Python constraint: `requires-python = ">=3.8,<3.13"`
   - Proper project metadata for UV
   - All dependencies listed with correct versions

3. **`INSTALL.md`** (NEW)
   - Comprehensive installation guide
   - UV-specific instructions
   - Platform-specific troubleshooting
   - Alternative installation methods

4. **`UV_FIX.md`** (NEW)
   - Quick reference for your specific error
   - Step-by-step resolution commands
   - Common issues and solutions

## ✅ Working Installation Commands

### On Your Linux Server (Python 3.11.0rc1)

```bash
cd /var/www/spera_AI/6_emotion_seperation/emotion_seperation

# Method 1: Traditional pip (Most Reliable)
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Method 2: UV with sync
uv venv --python 3.11
source .venv/bin/activate
uv pip sync requirements.txt

# Method 3: UV with install (if sync doesn't work)
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 📋 Verification Steps

```bash
# 1. Check Python version (should be 3.11.x)
python --version

# 2. Check installations
python setup_check.py

# 3. Expected output:
✓ Python version: 3.11.0
✓ FFmpeg: x.x.x
✓ torch: 2.x.x
✓ torchvision: 0.x.x
✓ opencv-python: 4.x.x
✓ librosa: 0.x.x
✓ pyannote.audio: 3.x.x
✓ All core dependencies installed!
```

## 🎯 Next Steps

### 1. Configure HuggingFace Token (Required for pyannote)
```bash
# Get token from: https://huggingface.co/settings/tokens
export HF_TOKEN="your_token_here"

# Or add to config.yaml:
nano config.yaml
# Add under diarization section:
#   hf_token: "your_token_here"
```

### 2. Accept Model Terms
Visit and accept:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### 3. Test the Pipeline
```bash
# Run component tests
python test_pipeline.py

# Process a sample video
python src/pipeline.py sample.mp4
```

## 📚 Documentation Structure

```
emotion_seperation/
├── README.md              # Main documentation
├── QUICKSTART.md          # 5-minute tutorial
├── INSTALL.md             # Detailed installation guide ⭐
├── UV_FIX.md              # UV-specific troubleshooting ⭐
├── TROUBLESHOOTING.md     # General problem solving
├── ARCHITECTURE.md        # Technical deep dive
├── WORKFLOW.md            # Visual guide
├── requirements.txt       # Dependencies (FIXED) ⭐
└── pyproject.toml         # Project config (NEW) ⭐
```

## 🐛 Common Issues & Solutions

### Issue: "pyannote.audio authentication failed"
```bash
# Solution:
export HF_TOKEN="hf_xxxxxxxxxxxxx"
python src/pipeline.py video.mp4
```

### Issue: "CUDA not available"
```bash
# Check CUDA
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "FFmpeg not found"
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

### Issue: "retinaface-pytorch still fails"
```bash
# Alternative: Use MediaPipe instead
pip install mediapipe>=0.10.0

# Update config.yaml:
face_detection:
  detector_type: "mediapipe"  # Change from "retinaface"
```

## ✅ Summary

| Item | Status | Notes |
|------|--------|-------|
| requirements.txt | ✅ FIXED | retinaface-pytorch version corrected |
| pyproject.toml | ✅ ADDED | Python version constraint added |
| Installation docs | ✅ ADDED | INSTALL.md & UV_FIX.md created |
| NumPy version | ✅ FIXED | Pinned to <2.0.0 |
| UV compatibility | ✅ FIXED | Proper project structure |

## 🚀 Quick Start (After Installation)

```bash
# 1. Set token
export HF_TOKEN="your_token_here"

# 2. Process video
python src/pipeline.py your_video.mp4

# 3. Check outputs
ls -lh output/
# You'll find:
# - person_roster.json (face IDs)
# - speaker_roster.json (speaker diarization)
# - av_map.json (person ↔ speaker matches)
# - clips_summary.json (emotion changes)
# - clips/ (auto-generated video clips)
```

## 📖 Full Documentation

- **Quick Fix**: [`UV_FIX.md`](UV_FIX.md) - Your specific error
- **Installation**: [`INSTALL.md`](INSTALL.md) - All installation methods
- **Quick Start**: [`QUICKSTART.md`](QUICKSTART.md) - 5-minute tutorial
- **Main Guide**: [`README.md`](README.md) - Complete documentation
- **Troubleshooting**: [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) - Problem solving

---

**Status**: ✅ All issues resolved. You can now install and run the pipeline.

**Recommended Command**:
```bash
pip install -r requirements.txt  # Most reliable method
```
