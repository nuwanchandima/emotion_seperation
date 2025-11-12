# Visual Workflow Guide

## 🎬 Complete Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  START: You have a video file (MP4, AVI, MOV, etc.)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Run Setup Check                                        │
│  $ python setup_check.py                                        │
│                                                                 │
│  Verifies:                                                      │
│  • Python 3.8+      ✓                                          │
│  • FFmpeg           ✓                                          │
│  • Dependencies     ✓                                          │
│  • Directory setup  ✓                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Run Tests (Optional but Recommended)                  │
│  $ python test_pipeline.py                                      │
│                                                                 │
│  Tests all components:                                          │
│  • Imports          ✓                                          │
│  • Configuration    ✓                                          │
│  • Face detection   ✓                                          │
│  • Utilities        ✓                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Run Pipeline                                           │
│  $ python src/pipeline.py your_video.mp4                        │
│                                                                 │
│  Or with options:                                               │
│  $ python src/pipeline.py video.mp4 --output results/          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                   ┌─────────┴─────────┐
                   │  Pipeline Stages  │
                   └─────────┬─────────┘
                             │
        ╔════════════════════╧════════════════════╗
        ║                                         ║
        ▼                                         ▼
┌───────────────┐                         ┌───────────────┐
│ Video Track   │                         │ Audio Track   │
└───────┬───────┘                         └───────┬───────┘
        │                                         │
        │ [1] Extract Media                       │
        │     • FFmpeg audio extraction           │
        │     • Frame generator setup             │
        │                                         │
        │     ✓ data/audio.wav                    │
        │                                         │
        ├─────────────────────────────────────────┤
        │                                         │
        │ [2] Face Detection → Tracking           │
        │     • RetinaFace/YOLO/OpenCV            │
        │     • BYTETrack multi-object tracking   │
        │     • FaceNet embeddings                │
        │     • Agglomerative clustering          │
        │                                         │
        │     ✓ outputs/tracks_faces.json         │
        │                                         │
        │                           [3] Speaker Diarization
        │                               • pyannote.audio
        │                               • VAD + clustering
        │                               • Overlap detection
        │                                         │
        │                               ✓ outputs/diarization.json
        │                                         │
        ├─────────────────────────────────────────┤
        │                                         │
        │ [4] Active Speaker Detection            │
        │     • Lip motion extraction             │
        │     • Audio energy computation          │
        │     • Cross-correlation sync            │
        │                                         │
        │     ✓ outputs/active_speaker.json       │
        │                                         │
        ├─────────────────────────────────────────┤
        │                                         │
        │ [5] A/V Matching (Hungarian Algorithm)  │
        │     • Build cost matrix                 │
        │       - Temporal overlap (40%)          │
        │       - ASD sync score (60%)            │
        │     • Linear assignment                 │
        │     • Off-screen detection              │
        │                                         │
        │     ✓ outputs/av_map.json               │
        │                                         │
        │                           [6] Emotion Change Detection
        │                               • Speech Emotion Recognition
        │                               • MFCC + pitch + energy
        │                               • wav2vec2 (optional)
        │                               • PELT change-point
        │                                         │
        │                               ✓ outputs/emotion_changes.json
        │                                         │
        ├─────────────────────────────────────────┤
        │                                         │
        │ [7] Clip Extraction                     │
        │     For each emotion change:            │
        │     • t_start = change_time - 0.5s      │
        │     • duration = 1.0s                   │
        │     • FFmpeg clip extraction            │
        │                                         │
        │     ✓ outputs/clips/*.mp4               │
        │     ✓ outputs/clips_manifest.md         │
        │                                         │
        ▼                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  COMPLETE! Pipeline finished successfully                       │
│                                                                 │
│  Generated outputs:                                             │
│  • Person IDs (person_1, person_2, ...)                        │
│  • Speaker IDs (SPEAKER_00, SPEAKER_01, ...)                   │
│  • A/V mappings with confidence scores                          │
│  • Emotion change timestamps                                    │
│  • Video clips at each change point                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Review Results                                         │
│                                                                 │
│  1. Open clips_manifest.md                                      │
│     • Human-readable summary                                    │
│     • All clips organized by person/speaker                     │
│                                                                 │
│  2. Watch clips in outputs/clips/                              │
│     • Each clip shows emotion transition                        │
│     • Named with emotion labels                                │
│                                                                 │
│  3. Examine JSON files                                          │
│     • tracks_faces.json - person tracks                        │
│     • av_map.json - person↔speaker links                       │
│     • emotion_changes.json - all changes                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Analyze & Use Results                                 │
│                                                                 │
│  Option A: Manual Review                                        │
│  • Watch clips in video player                                 │
│  • Read clips_manifest.md                                      │
│                                                                 │
│  Option B: Programmatic Analysis                               │
│  • Load JSON files in Python                                   │
│  • Build visualizations                                        │
│  • Export to CSV/database                                      │
│  • See example_usage.py for patterns                           │
│                                                                 │
│  Option C: Further Processing                                   │
│  • Feed clips to another model                                 │
│  • Combine with other data sources                             │
│  • Build dashboards or reports                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Quick Reference Commands

### Initial Setup
```bash
# Create environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (for diarization)
$env:HF_TOKEN="your_token"  # Windows PowerShell
export HF_TOKEN="your_token"  # Linux/Mac

# Verify installation
python setup_check.py
python test_pipeline.py
```

### Running the Pipeline
```bash
# Basic usage
python src\pipeline.py video.mp4

# With custom output directory
python src\pipeline.py video.mp4 --output my_results\

# With custom config
python src\pipeline.py video.mp4 --config my_config.yaml
```

### Examining Results
```bash
# View clips manifest (human-readable)
type outputs\clips_manifest.md  # Windows
cat outputs/clips_manifest.md   # Linux/Mac

# List all clips
dir outputs\clips\  # Windows
ls outputs/clips/   # Linux/Mac

# Check logs
type outputs\pipeline.log  # Windows
tail -f outputs/pipeline.log  # Linux/Mac (live)

# Open in Python
python
>>> from src.utils import load_json
>>> results = load_json('outputs/emotion_changes.json')
>>> print(results)
```

### Running Individual Stages
```bash
# Stage 1: Extract media
python src\extract_media.py video.mp4

# Stage 2: Face tracking
python src\faces_track_cluster.py video.mp4

# Stage 3: Diarization
python src\diarize.py data\audio.wav

# Stage 4: Active speaker
python src\active_speaker.py video.mp4 data\audio.wav

# Stage 5: A/V matching
python src\av_match.py

# Stage 6: Emotion detection
python src\emotion_change.py data\audio.wav

# Stage 7: Clip extraction
python src\export_clips.py video.mp4
```

## 📊 Output Files Reference

### JSON Files (Machine-Readable)
| File | Contains | Size |
|------|----------|------|
| `tracks_faces.json` | Person IDs, face tracks, segments | ~100KB |
| `diarization.json` | Speaker IDs, speech segments | ~50KB |
| `active_speaker.json` | Lip-audio sync scores | ~200KB |
| `av_map.json` | Person↔Speaker mappings | ~10KB |
| `emotion_changes.json` | Emotion shifts with timestamps | ~30KB |
| `clips_summary.json` | Clip metadata | ~50KB |

### Other Files
| File | Purpose | Size |
|------|---------|------|
| `diarization.rttm` | Standard RTTM format for evaluation | ~20KB |
| `clips_manifest.md` | Human-readable clip index | ~20KB |
| `pipeline.log` | Detailed execution logs | ~1MB |
| `clips/*.mp4` | Video clips | ~1-5MB each |

## 🔧 Configuration Quick Tweaks

### Make it Faster
```yaml
video:
  target_fps: 5  # ← Change from 10
face_detection:
  model: "opencv"  # ← Change from "retinaface"
clips:
  codec: "copy"  # ← Keep as "copy"
```

### Make it More Accurate
```yaml
video:
  target_fps: 15  # ← Change from 10
face_detection:
  confidence_threshold: 0.95  # ← Change from 0.9
emotion:
  model: "wav2vec2-ser"  # ← Keep transformer model
change_detection:
  penalty: 5  # ← Change from 10 (more sensitive)
```

### Reduce Memory Usage
```yaml
performance:
  batch_size: 8  # ← Change from 32
  cache_embeddings: false  # ← Change from true
video:
  target_fps: 5  # ← Lower frame rate
```

## 🎓 Understanding the Outputs

### Person vs Speaker
```
PERSON_1 (visual)  ←→  SPEAKER_00 (audio)
• Detected by face     • Detected by voice
• Tracked across       • Segmented by
  frames                 speech activity
• May appear/          • May be on or
  disappear              off screen
```

### Emotion Change Format
```json
{
  "t": 14.1,                    // Timestamp in seconds
  "from": {
    "label": "neutral",         // Discrete emotion
    "valence": 0.1,            // Pleasure (-1 to 1)
    "arousal": 0.0             // Energy (-1 to 1)
  },
  "to": {
    "label": "happy",
    "valence": 0.6,
    "arousal": 0.5
  }
}
```

### Clip Naming Convention
```
person_1_change_000_neutral_to_happy_t14.1s.mp4
│       │ │      │   │       │  │     │  │   │
│       │ │      │   │       │  │     │  │   └─ Extension
│       │ │      │   │       │  │     │  └───── Timestamp
│       │ │      │   │       │  │     └────────Time indicator
│       │ │      │   │       │  └──────────────To emotion
│       │ │      │   │       └─────────────────Transition
│       │ │      │   └─────────────────────────From emotion
│       │ │      └─────────────────────────────Change index
│       │ └────────────────────────────────────"change" literal
│       └──────────────────────────────────────Person/Speaker ID
└──────────────────────────────────────────────Entity type
```

## 🚨 Troubleshooting Decision Tree

```
Pipeline fails?
│
├─ During setup?
│  ├─ "FFmpeg not found" → Install FFmpeg, add to PATH
│  ├─ "ImportError" → pip install -r requirements.txt
│  └─ "Permission denied" → Run as admin or change output dir
│
├─ During face detection?
│  ├─ "No faces found" → Lower confidence_threshold
│  ├─ "CUDA out of memory" → Reduce batch_size
│  └─ "Too slow" → Use opencv model, reduce target_fps
│
├─ During diarization?
│  ├─ "No audio stream" → Check video has audio (ffprobe)
│  ├─ "pyannote fails" → Set HF_TOKEN environment variable
│  └─ "No speakers found" → Check audio quality
│
└─ During emotion detection?
   ├─ "No changes found" → Lower penalty in config
   ├─ "Too many changes" → Increase penalty
   └─ "Model download fails" → Use "features" model
```

## ✅ Success Checklist

Before considering it "working":
- [ ] `setup_check.py` all green
- [ ] `test_pipeline.py` passes
- [ ] Pipeline runs without errors
- [ ] `outputs/clips/` contains video files
- [ ] `outputs/clips_manifest.md` readable
- [ ] Clips play correctly in video player
- [ ] Emotion labels make sense for content
- [ ] Logs show reasonable processing time

## 🎉 You're Done!

Pipeline is working when you see:
```
✓ [7/7] Exporting emotion change clips (100% complete, 180s elapsed)
Pipeline complete! Total time: 180.3s (3.0 minutes)

Key results:
  - Persons detected: 2
  - Speakers detected: 2
  - A/V matches: 2
  - Emotion changes: 8
  - Clips exported: 8
```

Now go analyze some videos! 🎬✨
