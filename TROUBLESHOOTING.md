# Troubleshooting Guide

Common issues and their solutions for the A/V Emotion Detection Pipeline.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Output Quality Issues](#output-quality-issues)
- [Model-Specific Issues](#model-specific-issues)

---

## Installation Issues

### Issue: "pip install fails with dependency conflicts"

**Symptoms:**
```
ERROR: Cannot install package-a and package-b because these package versions have conflicting dependencies
```

**Solutions:**
1. Create a fresh virtual environment:
   ```bash
   python -m venv venv_fresh
   venv_fresh\Scripts\activate  # Windows
   source venv_fresh/bin/activate  # Linux/Mac
   ```

2. Install with --upgrade:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. Install problematic packages separately:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

### Issue: "FFmpeg not found"

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

**Solutions:**

**Windows:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add to PATH:
   - Search "Environment Variables" in Start Menu
   - Edit "Path" variable
   - Add `C:\ffmpeg\bin`
4. Restart terminal and verify: `ffmpeg -version`

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

### Issue: "CUDA/GPU not detected"

**Symptoms:**
```python
torch.cuda.is_available()  # Returns False
```

**Solutions:**
1. Check NVIDIA driver:
   ```bash
   nvidia-smi
   ```

2. Install correct PyTorch version:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CPU only
   pip install torch torchvision torchaudio
   ```

3. Update config for CPU:
   ```yaml
   # config.yaml
   performance:
     use_gpu: false
   ```

---

## Runtime Errors

### Issue: "pyannote.audio fails to load model"

**Symptoms:**
```
OSError: Can't load tokenizer for 'pyannote/speaker-diarization'
```

**Solutions:**
1. Accept model terms:
   - Visit: https://huggingface.co/pyannote/speaker-diarization
   - Click "Agree and access repository"

2. Get HuggingFace token:
   - Go to: https://huggingface.co/settings/tokens
   - Create new token
   - Set environment variable:
     ```bash
     # Windows PowerShell
     $env:HF_TOKEN="hf_..."
     
     # Linux/Mac
     export HF_TOKEN="hf_..."
     ```

3. Alternative: Use fallback VAD
   ```yaml
   # config.yaml - comment out model
   diarization:
     # model: "pyannote/speaker-diarization-3.1"
   ```

### Issue: "Out of memory error"

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size:
   ```yaml
   # config.yaml
   performance:
     batch_size: 8  # Down from 32
   ```

2. Process fewer frames:
   ```yaml
   video:
     target_fps: 5  # Down from 10
   ```

3. Enable mixed precision:
   ```yaml
   performance:
     mixed_precision: true
   ```

4. Clear CUDA cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Issue: "No audio stream found"

**Symptoms:**
```
ValueError: No audio stream found
```

**Solutions:**
1. Check video has audio:
   ```bash
   ffprobe -i video.mp4 -show_streams
   ```

2. Add audio to video:
   ```bash
   ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac output.mp4
   ```

3. Extract audio separately:
   ```bash
   ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav
   ```

### Issue: "Permission denied when writing outputs"

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'outputs/...'
```

**Solutions:**
1. Run with administrator/sudo (not recommended)
2. Change output directory:
   ```bash
   python src/pipeline.py video.mp4 --output ~/my_outputs/
   ```
3. Check folder permissions:
   ```bash
   # Linux/Mac
   chmod -R 755 outputs/
   ```

---

## Performance Issues

### Issue: "Pipeline is very slow"

**Optimization Checklist:**

1. **Use GPU**:
   ```yaml
   performance:
     use_gpu: true
     mixed_precision: true
   ```

2. **Reduce frame rate**:
   ```yaml
   video:
     target_fps: 5  # Instead of 10-15
   ```

3. **Use faster models**:
   ```yaml
   face_detection:
     model: "opencv"  # Instead of retinaface
   
   emotion:
     model: "features"  # Instead of transformer
   ```

4. **Fast clip extraction**:
   ```yaml
   clips:
     codec: "copy"  # Instead of libx264
   ```

5. **Disable caching if low on disk**:
   ```yaml
   performance:
     cache_embeddings: false
   ```

### Issue: "High memory usage"

**Solutions:**
1. Process video in chunks
2. Reduce batch size
3. Clear intermediate results:
   ```python
   import gc
   gc.collect()
   ```

---

## Output Quality Issues

### Issue: "No faces detected"

**Diagnostics:**
```python
# Test face detection manually
from src.faces_track_cluster import FaceDetector
from src.utils import load_config
import cv2

config = load_config()
detector = FaceDetector(config, None)

frame = cv2.imread('test_frame.jpg')
detections = detector.detect(frame)
print(f"Found {len(detections)} faces")
```

**Solutions:**
1. Lower confidence threshold:
   ```yaml
   face_detection:
     confidence_threshold: 0.7  # Down from 0.9
     min_face_size: 30  # Down from 40
   ```

2. Try different detector:
   ```yaml
   face_detection:
     model: "opencv"  # More permissive
   ```

3. Check video quality (resolution, lighting)

### Issue: "Incorrect person clustering"

**Symptoms:**
- Same person gets multiple IDs
- Different people share one ID

**Solutions:**
1. Adjust clustering threshold:
   ```yaml
   face_clustering:
     distance_threshold: 0.5  # Higher = fewer clusters
     # OR
     distance_threshold: 0.3  # Lower = more clusters
   ```

2. Increase minimum track length:
   ```yaml
   face_clustering:
     min_track_length: 15  # Up from 10
   ```

3. Use better embeddings:
   ```yaml
   face_recognition:
     model: "arcface"  # Instead of facenet
   ```

### Issue: "No emotion changes detected"

**Solutions:**
1. Lower change detection penalty:
   ```yaml
   change_detection:
     penalty: 3  # Down from 10 (more sensitive)
     min_size: 2  # Down from 4
   ```

2. Check audio quality:
   ```bash
   ffprobe -i data/audio.wav
   ```

3. Verify speaker segments exist:
   ```python
   from src.utils import load_json
   diar = load_json('outputs/diarization.json')
   print(diar['speakers'])
   ```

### Issue: "Poor A/V matching"

**Solutions:**
1. Adjust matching thresholds:
   ```yaml
   av_matching:
     overlap_threshold: 0.3  # Down from 0.5
     confidence_threshold: 0.5  # Down from 0.6
   ```

2. Increase ASD window size:
   ```yaml
   active_speaker:
     window_size: 0.5  # Up from 0.25
   ```

---

## Model-Specific Issues

### Issue: "RetinaFace installation fails"

**Solution:** Use alternative detector
```yaml
face_detection:
  model: "opencv"  # Fallback to OpenCV
```

### Issue: "transformers model download fails"

**Solutions:**
1. Manual download:
   ```python
   from transformers import pipeline
   model = pipeline("audio-classification", 
                    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
   ```

2. Use offline mode:
   ```python
   export TRANSFORMERS_OFFLINE=1
   ```

3. Use feature-based approach:
   ```yaml
   emotion:
     model: "features"  # Doesn't need downloads
   ```

### Issue: "ruptures import error"

**Solutions:**
1. Install manually:
   ```bash
   pip install ruptures
   ```

2. Check numpy compatibility:
   ```bash
   pip install numpy==1.24.0
   pip install ruptures
   ```

---

## Debugging Tips

### Enable verbose logging:
```yaml
# config.yaml
logging:
  level: "DEBUG"  # Instead of INFO
  save_logs: true
```

### Check logs:
```bash
# Windows
type outputs\pipeline.log | more

# Linux/Mac
tail -f outputs/pipeline.log
```

### Test individual stages:
```bash
# Stage by stage
python src/extract_media.py video.mp4
python src/faces_track_cluster.py video.mp4
python src/diarize.py data/audio.wav
# ... etc
```

### Verify FFmpeg works:
```bash
ffmpeg -i video.mp4 -t 5 test_clip.mp4
```

### Check GPU usage:
```bash
# While pipeline is running
nvidia-smi -l 1  # Update every second
```

### Profile performance:
```python
import cProfile
import pstats

cProfile.run('run_pipeline("video.mp4")', 'output.prof')
stats = pstats.Stats('output.prof')
stats.sort_stats('cumulative')
stats.print_stats(20)
```

---

## Getting More Help

1. **Check logs**: Always check `outputs/pipeline.log` first
2. **Verify setup**: Run `python setup_check.py`
3. **Test with sample**: Try with a short (30s) test video
4. **Isolate the issue**: Run pipeline stages one by one
5. **Check versions**:
   ```bash
   python --version
   pip list | grep torch
   ffmpeg -version
   ```

---

## Common Warnings (Safe to Ignore)

✓ These warnings are normal:

- `UserWarning: torch.meshgrid` - PyTorch version compatibility
- `FutureWarning: ` - Library deprecation notices
- `DeprecationWarning: ` - Same as above
- `⚠ CUDA not available - will use CPU` - If you don't have GPU

---

## Still Having Issues?

Create a detailed bug report with:
1. Full error message and stack trace
2. Contents of `outputs/pipeline.log`
3. System info: OS, Python version, CUDA version
4. Video metadata: `ffprobe -i video.mp4`
5. Steps to reproduce

Run diagnostics:
```bash
python setup_check.py > diagnostics.txt
```
