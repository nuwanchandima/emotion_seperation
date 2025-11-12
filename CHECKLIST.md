# ✅ Installation & Setup Checklist

## 📋 Pre-Installation Checklist

- [ ] Python 3.8 - 3.12 installed (`python --version`)
- [ ] FFmpeg installed (`ffmpeg -version`)
- [ ] Git installed (if cloning repository)
- [ ] CUDA toolkit (optional, for GPU acceleration)
- [ ] HuggingFace account created (for pyannote token)

---

## 🔧 Installation Checklist

### Option A: Automated Installation
- [ ] Downloaded `install.sh` (Linux/macOS) or `install.ps1` (Windows)
- [ ] Made script executable: `chmod +x install.sh` (Linux/macOS only)
- [ ] Ran installation script: `bash install.sh` or `.\install.ps1`
- [ ] Installation completed without errors
- [ ] Virtual environment activated

### Option B: Manual Installation
- [ ] Created virtual environment: `python -m venv venv`
- [ ] Activated virtual environment
  - [ ] Linux/macOS: `source venv/bin/activate`
  - [ ] Windows: `.\venv\Scripts\Activate.ps1`
- [ ] Upgraded pip: `pip install --upgrade pip`
- [ ] Installed requirements: `pip install -r requirements.txt`
- [ ] All packages installed successfully

### Option C: UV Installation
- [ ] UV installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] Created venv: `uv venv --python 3.11`
- [ ] Activated venv
- [ ] Installed deps: `uv pip sync requirements.txt`
- [ ] All packages installed successfully

---

## 🔑 Configuration Checklist

### HuggingFace Token Setup
- [ ] Visited https://huggingface.co/settings/tokens
- [ ] Created new token (read access)
- [ ] Copied token
- [ ] Set environment variable:
  - [ ] Linux/macOS: `export HF_TOKEN="your_token"`
  - [ ] Windows: `$env:HF_TOKEN = "your_token"`
- [ ] OR added to `config.yaml` under `diarization.hf_token`

### Model Terms Acceptance
- [ ] Visited https://huggingface.co/pyannote/speaker-diarization-3.1
- [ ] Clicked "Agree and access repository"
- [ ] Visited https://huggingface.co/pyannote/segmentation-3.0
- [ ] Clicked "Agree and access repository"

### Configuration File
- [ ] Reviewed `config.yaml`
- [ ] Adjusted paths if needed
- [ ] Set `face_detection.detector_type` (retinaface/mediapipe/opencv)
- [ ] Set `diarization.use_auth_token: true`
- [ ] Adjusted performance settings (GPU/CPU)

---

## ✅ Verification Checklist

### Basic Checks
- [ ] Ran: `python setup_check.py`
- [ ] All dependency checks passed (✓)
- [ ] FFmpeg detected
- [ ] Python version correct
- [ ] torch installed
- [ ] opencv installed
- [ ] librosa installed
- [ ] pyannote.audio installed

### GPU Checks (Optional)
- [ ] Ran: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Result is `True` (if you have NVIDIA GPU)
- [ ] Ran: `nvidia-smi` (shows GPU info)
- [ ] CUDA version compatible with PyTorch

### Component Tests
- [ ] Ran: `python test_pipeline.py`
- [ ] All tests passed:
  - [ ] Import test
  - [ ] FFmpeg test
  - [ ] Config test
  - [ ] Media extraction test
  - [ ] Face detection test
  - [ ] Utility functions test
  - [ ] Emotion features test (if models available)
  - [ ] Change detection test

---

## 🎬 First Run Checklist

### Prepare Test Video
- [ ] Have a test video file ready (MP4, AVI, MOV, etc.)
- [ ] Video contains visible faces
- [ ] Video has audio track
- [ ] Video is at least 30 seconds long (for meaningful results)
- [ ] Video path is accessible

### Run Pipeline
- [ ] Activated virtual environment
- [ ] Set HF_TOKEN environment variable
- [ ] Ran: `python src/pipeline.py path/to/video.mp4`
- [ ] Pipeline started without errors
- [ ] Watched progress through stages:
  - [ ] Stage 1: Media extraction
  - [ ] Stage 2: Face detection
  - [ ] Stage 3: Face tracking
  - [ ] Stage 4: Face clustering
  - [ ] Stage 5: Speaker diarization
  - [ ] Stage 6: Active speaker detection
  - [ ] Stage 7: A/V matching
  - [ ] Stage 8: Emotion recognition
  - [ ] Stage 9: Change detection
  - [ ] Stage 10: Clip extraction
- [ ] Pipeline completed successfully

### Check Outputs
- [ ] `output/` directory created
- [ ] `person_roster.json` exists and contains face IDs
- [ ] `speaker_roster.json` exists and contains speakers
- [ ] `av_map.json` exists and contains person-speaker matches
- [ ] `clips_summary.json` exists and contains emotion changes
- [ ] `clips/` directory exists
- [ ] Video clips generated (if emotion changes detected)
- [ ] `pipeline.log` contains detailed logs

---

## 📊 Output Validation Checklist

### person_roster.json
- [ ] File exists and is valid JSON
- [ ] Contains `persons` array
- [ ] Each person has:
  - [ ] `person_id` (e.g., "person_1")
  - [ ] `appearances` (list of time segments)
  - [ ] `total_frames` count
  - [ ] `representative_frame` path

### speaker_roster.json
- [ ] File exists and is valid JSON
- [ ] Contains `speakers` array
- [ ] Each speaker has:
  - [ ] `speaker_id` (e.g., "speaker_1")
  - [ ] `segments` (list of time segments)
  - [ ] `total_duration`
- [ ] Overlaps detected (if multiple speakers)

### av_map.json
- [ ] File exists and is valid JSON
- [ ] Contains `matches` array
- [ ] Each match has:
  - [ ] `person_id`
  - [ ] `speaker_id`
  - [ ] `confidence` score (0-1)
  - [ ] `method` (temporal/asd/combined)
- [ ] Off-screen speakers identified (if any)

### clips_summary.json
- [ ] File exists and is valid JSON
- [ ] Contains `changes` array
- [ ] Each change has:
  - [ ] `change_id`
  - [ ] `timestamp`
  - [ ] `speaker_id`
  - [ ] `from_emotion` and `to_emotion`
  - [ ] `confidence` score
  - [ ] `clip_path`

### Video Clips
- [ ] Clips directory exists: `output/clips/`
- [ ] Clip files exist (*.mp4)
- [ ] Clips are playable
- [ ] Clips are ~10 seconds duration
- [ ] Clips centered around emotion change point

---

## 🐛 Troubleshooting Checklist

### Installation Issues
- [ ] Checked Python version (3.8-3.12, NOT 3.13+)
- [ ] Checked FFmpeg installation
- [ ] Reviewed error messages in terminal
- [ ] Checked `INSTALL.md` for specific issues
- [ ] Tried alternative installation method

### Runtime Issues
- [ ] HF_TOKEN is set correctly
- [ ] pyannote model terms accepted
- [ ] Video file path is correct
- [ ] Video file is not corrupted
- [ ] Sufficient disk space for outputs
- [ ] Checked `pipeline.log` for errors

### Performance Issues
- [ ] CUDA available (for GPU acceleration)
- [ ] Adjusted batch sizes in `config.yaml`
- [ ] Reduced video resolution if needed
- [ ] Closed other GPU-intensive applications
- [ ] Checked system resources (RAM, disk)

### Output Quality Issues
- [ ] Face detection threshold adjusted
- [ ] Speaker diarization min_speakers/max_speakers set
- [ ] Emotion change detection penalty adjusted
- [ ] Video has sufficient quality (resolution, lighting)
- [ ] Audio quality is adequate

---

## 📚 Documentation Review Checklist

- [ ] Read `README.md` (main guide)
- [ ] Read `QUICKSTART.md` (quick tutorial)
- [ ] Read `INSTALL.md` (installation details)
- [ ] Reviewed `config.yaml` comments
- [ ] Bookmarked `TROUBLESHOOTING.md`
- [ ] Reviewed `ARCHITECTURE.md` (understanding internals)
- [ ] Checked `WORKFLOW.md` (visual guide)

---

## 🚀 Production Readiness Checklist

### Code Quality
- [ ] All tests passing: `python test_pipeline.py`
- [ ] No errors in logs
- [ ] Processed multiple videos successfully
- [ ] Outputs validated manually

### Configuration
- [ ] `config.yaml` tuned for your use case
- [ ] Model choices finalized
- [ ] Performance settings optimized
- [ ] Logging level set appropriately

### Deployment
- [ ] Environment variables documented
- [ ] Dependencies locked (consider `pip freeze > requirements.lock`)
- [ ] Resource requirements documented (RAM, GPU, disk)
- [ ] Batch processing tested: `python batch_process.py videos/`

### Documentation
- [ ] Usage documented for your team
- [ ] Example videos and outputs prepared
- [ ] Known limitations documented
- [ ] Troubleshooting steps added

### Monitoring
- [ ] Log rotation configured
- [ ] Disk space monitoring in place
- [ ] Error alerting configured (optional)
- [ ] Performance metrics tracked (optional)

---

## 🎉 Final Checklist

- [ ] **Installation complete** - All dependencies installed
- [ ] **Configuration complete** - HF token set, config.yaml tuned
- [ ] **Verification complete** - All checks passed
- [ ] **First run successful** - Test video processed
- [ ] **Outputs validated** - All JSON files and clips correct
- [ ] **Documentation reviewed** - Know where to find help
- [ ] **Ready for production** - System stable and optimized

---

## 📞 Getting Help

If any checkbox above fails:

1. **Check logs**: Review `pipeline.log` for errors
2. **Search docs**: Check `TROUBLESHOOTING.md` for your issue
3. **Review installation**: See `INSTALL.md` or `UV_FIX.md`
4. **Check configuration**: Review `config.yaml` settings
5. **Test components**: Run `python test_pipeline.py`

---

## 🔄 Maintenance Checklist (Periodic)

### Weekly
- [ ] Check for package updates: `pip list --outdated`
- [ ] Review logs for errors
- [ ] Clean up old output directories

### Monthly
- [ ] Update dependencies: `pip install -r requirements.txt --upgrade`
- [ ] Test with new model versions
- [ ] Review and optimize config.yaml

### As Needed
- [ ] Update documentation for changes
- [ ] Add new test cases
- [ ] Tune parameters based on results
- [ ] Backup important configurations

---

**Print this checklist and check off items as you complete them!**
