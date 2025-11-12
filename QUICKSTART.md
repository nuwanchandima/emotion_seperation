# Quick Start Guide

Welcome to the A/V Emotion Detection Pipeline! This guide will get you up and running in 5 minutes.

## Prerequisites Check

Before starting, ensure you have:

1. **Python 3.8+** installed
   ```bash
   python --version
   ```

2. **FFmpeg** installed and in PATH
   ```bash
   ffmpeg -version
   ```

3. **GPU** (optional but recommended)
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Installation

### Step 1: Clone/Download

You already have the project! Navigate to it:
```bash
cd "c:\Users\Nuwan\OneDrive\Desktop\ML\Spera ML\Task31_emotion_seperation"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Important for pyannote.audio**:
1. Visit: https://huggingface.co/pyannote/speaker-diarization
2. Accept the conditions
3. Get your HuggingFace token from: https://huggingface.co/settings/tokens
4. Set it as environment variable:
   ```bash
   # Windows PowerShell
   $env:HF_TOKEN="your_token_here"
   
   # Linux/Mac
   export HF_TOKEN="your_token_here"
   ```

## Quick Test

### Option 1: Use a Sample Video

```bash
# Download a sample video (replace with your own)
# For testing, use any short video file you have

# Run the pipeline
python src/pipeline.py path/to/your/sample.mp4
```

### Option 2: Test Individual Components

```bash
# Test media extraction
python src/extract_media.py sample.mp4

# Test face detection (after media extraction)
python src/faces_track_cluster.py sample.mp4

# Test diarization (after media extraction)
python src/diarize.py data/audio.wav
```

## Expected Output

After running the full pipeline, you should see:

```
outputs/
├── tracks_faces.json       # Face tracking results
├── diarization.json        # Speaker segments
├── diarization.rttm        # RTTM format for evaluation
├── active_speaker.json     # Lip-audio sync scores
├── av_map.json            # Person ↔ Speaker mapping
├── emotion_changes.json   # Detected emotion changes
├── clips_summary.json     # Clip metadata
├── clips_manifest.md      # Human-readable clip list
├── pipeline.log          # Detailed logs
└── clips/                # Video clips
    ├── person_1_change_000_neutral_to_happy_t14.1s.mp4
    ├── person_1_change_001_happy_to_sad_t28.5s.mp4
    └── ...
```

## Configuration

Edit `config.yaml` to customize:

### For Faster Processing (Lower Quality)
```yaml
video:
  target_fps: 5  # Process fewer frames

face_detection:
  model: "opencv"  # Faster but less accurate

clips:
  codec: "copy"  # Fast clip extraction
```

### For Higher Accuracy (Slower)
```yaml
video:
  target_fps: 15  # More frames

face_detection:
  model: "retinaface"  # Better detection
  confidence_threshold: 0.95

clips:
  codec: "libx264"  # Precise timestamps
```

## Common Issues & Solutions

### Issue: "FFmpeg not found"
**Solution**: Install FFmpeg
- Windows: Download from https://ffmpeg.org/download.html
- Linux: `sudo apt install ffmpeg`
- Mac: `brew install ffmpeg`

### Issue: "CUDA out of memory"
**Solution**: 
```yaml
# In config.yaml
performance:
  batch_size: 8  # Reduce from 32
  mixed_precision: true
```

### Issue: "pyannote model not found"
**Solution**: Set HuggingFace token (see Installation Step 3)

### Issue: "No faces detected"
**Solution**: 
- Check video quality
- Lower confidence threshold in config
- Try different face detection model

### Issue: "Emotion changes not detected"
**Solution**:
```yaml
# In config.yaml
change_detection:
  penalty: 5  # Lower = more sensitive (default: 10)
```

## Next Steps

1. **Analyze Your Video**
   ```bash
   python src/pipeline.py your_video.mp4 --output results/video1/
   ```

2. **Review Results**
   - Check `outputs/clips_manifest.md` for a summary
   - View clips in `outputs/clips/`
   - Examine JSON files for detailed data

3. **Tune Configuration**
   - Adjust thresholds based on your use case
   - Try different models for better accuracy
   - Optimize for speed vs quality

4. **Integrate Into Your Workflow**
   - Use JSON outputs in your own analysis
   - Build dashboards or visualizations
   - Automate batch processing

## Example: Batch Processing

Create a script to process multiple videos:

```python
import os
from src.pipeline import run_pipeline

videos = [
    "video1.mp4",
    "video2.mp4",
    "video3.mp4"
]

for video in videos:
    print(f"Processing {video}...")
    output_dir = f"outputs/{os.path.splitext(video)[0]}"
    
    # Create temp config with custom output
    config = {
        'paths': {
            'output_dir': output_dir
        }
    }
    
    try:
        run_pipeline(video, 'config.yaml')
        print(f"✓ {video} complete")
    except Exception as e:
        print(f"✗ {video} failed: {e}")
```

## Getting Help

1. Check `outputs/pipeline.log` for detailed error messages
2. Review the README.md for full documentation
3. Try running individual stages to isolate issues
4. Verify your video has both audio and video streams:
   ```bash
   ffprobe -i your_video.mp4
   ```

## Performance Benchmarks

Typical processing times (on NVIDIA RTX 3080):

| Video Length | Target FPS | Processing Time |
|--------------|-----------|-----------------|
| 1 minute     | 10        | ~2 minutes      |
| 5 minutes    | 10        | ~8 minutes      |
| 30 minutes   | 5         | ~30 minutes     |
| 1 hour       | 5         | ~1 hour         |

*Note: First run is slower due to model downloads*

## Support

For issues, questions, or contributions:
- Check the full README.md
- Review logs in `outputs/pipeline.log`
- Ensure all dependencies are installed
- Verify FFmpeg is working: `ffmpeg -version`

---

**You're all set! 🚀 Start analyzing your videos!**
