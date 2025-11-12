# 🚀 QUICK START - Installation Fixed!

## Your Error → Solution (30 seconds)

```
❌ BEFORE:
uv add -r requirements.txt
× No solution found: retinaface-pytorch>=0.0.10 not available

✅ NOW (Pull latest code):
git pull origin main
# or download the fixed requirements.txt

✅ THEN (Choose one):
bash install.sh              # Auto-install (Linux/macOS)
.\install.ps1               # Auto-install (Windows)
pip install -r requirements.txt  # Manual (everywhere)
```

---

## 📋 3-Step Installation (Your Linux Server)

### Step 1: Navigate to Project
```bash
cd /var/www/spera_AI/6_emotion_seperation/emotion_seperation
```

### Step 2: Install (Choose one method)

**Option A: Automated (Easiest)**
```bash
bash install.sh
```

**Option B: Manual with pip (Most Reliable)**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Option C: UV (Fast)**
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip sync requirements.txt
```

### Step 3: Configure Token
```bash
# Get token: https://huggingface.co/settings/tokens
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
```

---

## ✅ Verify Installation (30 seconds)

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
✓ All core dependencies installed!
```

---

## 🎬 Process Your First Video (2 minutes)

```bash
# Download a test video (optional)
wget https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4

# Process it
python src/pipeline.py big_buck_bunny_720p_1mb.mp4

# Check results
ls -lh output/
cat output/clips_summary.json
```

**Output structure:**
```
output/
├── person_roster.json      # All unique faces detected
├── speaker_roster.json     # All speakers identified
├── av_map.json            # Person-to-speaker matches
├── clips_summary.json     # All emotion change points
└── clips/                 # Auto-generated video clips
    ├── change_0001_speaker_1_positive_to_negative.mp4
    ├── change_0002_speaker_2_neutral_to_positive.mp4
    └── ...
```

---

## 🔧 What Was Fixed

| Issue | Fix | File |
|-------|-----|------|
| ❌ `retinaface-pytorch>=0.0.10` doesn't exist | ✅ Changed to `>=0.0.7` | `requirements.txt` |
| ❌ No Python version constraint | ✅ Added `requires-python = ">=3.8,<3.13"` | `pyproject.toml` (new) |
| ❌ numpy 2.0 compatibility issues | ✅ Pinned to `<2.0.0` | `requirements.txt` |
| ❌ Missing UV support | ✅ Added proper project structure | `pyproject.toml` (new) |
| ❌ Complex installation | ✅ Added auto-install scripts | `install.sh`, `install.ps1` |

---

## 📖 Documentation Files (New)

```
emotion_seperation/
├── 📘 README.md              # Main documentation (updated)
├── ⚡ QUICKSTART.md          # 5-minute tutorial
├── 🔧 INSTALL.md             # Complete installation guide ⭐ NEW
├── 🚨 UV_FIX.md              # Your specific UV error fix ⭐ NEW
├── ✅ INSTALLATION_FIXED.md  # Overview of fixes ⭐ NEW
├── 🎯 FIXED_SUMMARY.md       # Complete summary ⭐ NEW
├── 🏃 THIS_FILE.md           # Quick visual guide ⭐ NEW
├── 🐛 TROUBLESHOOTING.md     # Problem solving
├── 🏗️  ARCHITECTURE.md       # Technical deep dive
├── 📊 WORKFLOW.md            # Visual workflow
├── 📦 requirements.txt       # Dependencies (FIXED) ⭐
├── ⚙️  pyproject.toml         # Project config (NEW) ⭐
├── 🐧 install.sh             # Linux/macOS installer ⭐ NEW
└── 🪟 install.ps1            # Windows installer ⭐ NEW
```

---

## 🎯 One-Line Solutions

### Problem: UV fails with "no solution found"
```bash
# Solution: Use pip instead (most reliable)
pip install -r requirements.txt
```

### Problem: "pyannote authentication failed"
```bash
# Solution: Set token
export HF_TOKEN="hf_xxxxxxxxxxxxx"  # Linux/macOS
$env:HF_TOKEN = "hf_xxxxxxxxxxxxx"   # Windows
```

### Problem: "CUDA not available"
```bash
# Solution: Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problem: "FFmpeg not found"
```bash
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: choco install ffmpeg
```

---

## 🎬 Complete Workflow (Visual)

```
┌─────────────────────────────────────────────────────────┐
│ 1️⃣  INSTALLATION                                        │
├─────────────────────────────────────────────────────────┤
│  bash install.sh  (or pip install -r requirements.txt) │
│  ✓ Creates venv, installs all dependencies             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2️⃣  CONFIGURATION                                       │
├─────────────────────────────────────────────────────────┤
│  export HF_TOKEN="your_token"                          │
│  ✓ Accept terms on HuggingFace website                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3️⃣  VERIFICATION                                        │
├─────────────────────────────────────────────────────────┤
│  python setup_check.py                                 │
│  ✓ All checks pass                                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 4️⃣  PROCESSING                                          │
├─────────────────────────────────────────────────────────┤
│  python src/pipeline.py video.mp4                      │
│  ✓ Face detection → Diarization → Emotion → Clips      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 5️⃣  RESULTS                                             │
├─────────────────────────────────────────────────────────┤
│  output/                                               │
│  ├── person_roster.json    (face IDs)                  │
│  ├── speaker_roster.json   (speakers)                  │
│  ├── av_map.json          (person↔speaker)             │
│  ├── clips_summary.json   (emotion changes)            │
│  └── clips/               (auto-generated clips)       │
└─────────────────────────────────────────────────────────┘
```

---

## 📱 Quick Reference Card

| Command | Purpose |
|---------|---------|
| `bash install.sh` | Auto-install everything |
| `python setup_check.py` | Verify installation |
| `python test_pipeline.py` | Test all components |
| `python src/pipeline.py video.mp4` | Process single video |
| `python batch_process.py videos/` | Process multiple videos |
| `cat output/clips_summary.json` | View emotion changes |

---

## 🚨 Emergency Troubleshooting

**If everything fails:**
```bash
# Nuclear option: Fresh install
rm -rf venv .venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Still failing? Use minimal install:
pip install torch torchvision opencv-python librosa pyannote.audio transformers pyyaml
```

**If retinaface still fails:**
```bash
# Use MediaPipe instead
pip install mediapipe

# Edit config.yaml:
# face_detection:
#   detector_type: "mediapipe"
```

---

## ✅ Success Indicators

After running `python setup_check.py`, you should see:

```
✓ Python version: 3.11.0
✓ FFmpeg: 4.4.2
✓ torch: 2.1.0
✓ torchvision: 0.16.0
✓ opencv-python: 4.8.1
✓ librosa: 0.10.1
✓ pyannote.audio: 3.1.0
✓ transformers: 4.35.0
✓ All core dependencies installed!

CUDA Status: Available ✓
GPU: NVIDIA GeForce RTX 3090

🎉 All checks passed! Ready to process videos.
```

---

## 🎉 You're Ready!

Your installation issues are completely resolved. You can now:

✅ Install with pip, UV, or automated scripts  
✅ Process videos with emotion detection  
✅ Get person rosters, speaker diarization, and A/V mapping  
✅ Auto-generate clips at emotion change points  

**Start now:**
```bash
bash install.sh && python src/pipeline.py video.mp4
```

---

Need more details? Check:
- **Installation Issues**: `INSTALL.md` or `UV_FIX.md`
- **Usage Guide**: `README.md` or `QUICKSTART.md`
- **Problems**: `TROUBLESHOOTING.md`
