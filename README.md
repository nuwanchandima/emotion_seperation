# A/V Emotion Detection Pipeline

A production-ready system for **face tracking, speaker diarization, audio-visual mapping, and emotion change detection** in video content. Perfect for analyzing films, interviews, meetings, and any multi-speaker video content.

> **🚨 INSTALLATION ISSUE FIXED**: If you encountered `uv add -r requirements.txt` errors, see [`QUICK_FIX.md`](QUICK_FIX.md) for immediate solution!

## 🎯 What You Get

1. **Person Roster**: Unique face IDs (`person_1`, `person_2`, ...) with persistent tracking across the video
2. **Speaker Roster**: Speaker diarization with overlapping speech detection (`speaker_1`, `speaker_2`, ...)
3. **A/V Mapping**: Best match between visible persons and speakers with confidence scores
4. **Emotion Changes**: Timestamps where vocal emotion shifts, plus auto-generated video clips

## 📚 Documentation Quick Links

| Need | Read This | Time |
|------|-----------|------|
| 🚨 **UV/pip error fix** | [`QUICK_FIX.md`](QUICK_FIX.md) | 30 sec |
| ✅ **Complete checklist** | [`CHECKLIST.md`](CHECKLIST.md) | 5 min |
| 🔧 **Installation guide** | [`INSTALL.md`](INSTALL.md) | 10 min |
| ⚡ **Quick tutorial** | [`QUICKSTART.md`](QUICKSTART.md) | 5 min |
| 🐛 **Troubleshooting** | [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | As needed |
| 🏗️ **Architecture** | [`ARCHITECTURE.md`](ARCHITECTURE.md) | 15 min |
| 📊 **Workflow** | [`WORKFLOW.md`](WORKFLOW.md) | 5 min |

## 🏗️ Architecture

```
Input Video → [Extract Media] → Audio + Frames
                                     ↓
                          [Face Detection & Tracking]
                                     ↓
                          [Face Embedding & Clustering] → Person IDs
                                     ↓
Audio → [Speaker Diarization] → Speaker IDs
                                     ↓
Audio + Faces → [Active Speaker Detection] → Lip-Audio Sync Scores
                                     ↓
                          [A/V Matching (Hungarian)] → Person ↔ Speaker Links
                                     ↓
Audio → [Speech Emotion Recognition] → Emotion Time Series
                                     ↓
                          [Change Point Detection] → Emotion Shifts
                                     ↓
                          [Clip Extraction] → Video Clips
```

## 📋 Prerequisites

- **Python 3.8 - 3.12** (NOT 3.13+ due to package compatibility)
- **FFmpeg** (must be in PATH)
- **CUDA-capable GPU** (recommended for speed, but CPU works)

## 🚀 Quick Start

### 1. Install Dependencies

**Choose your installation method:**

#### Option A: UV (Fast, recommended for Linux/macOS)
```bash
uv venv --python 3.11
source .venv/bin/activate  # Linux/macOS
uv pip sync requirements.txt
```

#### Option B: PIP (Traditional, works everywhere)
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS: source venv/bin/activate
                          # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

📖 **Having issues?** See detailed instructions: [`INSTALL.md`](INSTALL.md)

**Note**: For pyannote.audio, you need to accept model terms and provide a HuggingFace token:
```bash
# 1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1
# 2. Accept conditions and get your token from https://huggingface.co/settings/tokens
export HF_TOKEN=your_token_here
```

### 2. Run the Pipeline

```bash
# Full pipeline on a video
python src/pipeline.py path/to/your/video.mp4

# With custom config
python src/pipeline.py video.mp4 --config config.yaml

# Custom output directory
python src/pipeline.py video.mp4 --output results/my_analysis/
```

### 3. Check Results

All outputs go to `outputs/`:
- `tracks_faces.json` - Face tracks and person IDs
- `diarization.json` + `.rttm` - Speaker timeline
- `av_map.json` - Person ↔ Speaker mappings
- `emotion_changes.json` - Emotion change timestamps
- `clips/` - Video clips around each change point
- `clips_manifest.md` - Human-readable clip index

## 📦 Folder Structure

```
Task31_emotion_seperation/
├── config.yaml              # Configuration (models, thresholds, etc.)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Extracted media (audio.wav, etc.)
├── outputs/                # Pipeline outputs
│   ├── tracks_faces.json
│   ├── diarization.json
│   ├── av_map.json
│   ├── emotion_changes.json
│   ├── clips_summary.json
│   ├── clips_manifest.md
│   └── clips/             # Emotion change video clips
├── models/                # Downloaded model checkpoints (auto-created)
└── src/
    ├── pipeline.py         # Main orchestrator
    ├── extract_media.py    # Audio/video extraction
    ├── faces_track_cluster.py  # Face detection → tracking → clustering
    ├── diarize.py          # Speaker diarization
    ├── active_speaker.py   # Lip-audio sync
    ├── av_match.py         # Hungarian matching
    ├── emotion_change.py   # SER + change-point detection
    ├── export_clips.py     # Clip extraction
    └── utils.py            # Shared utilities
```

## 🔧 Configuration

Edit `config.yaml` to customize:

- **Face Detection**: Model choice (RetinaFace, YOLOv8, MediaPipe), confidence thresholds
- **Face Tracking**: BYTETrack or DeepSORT settings
- **Face Clustering**: Agglomerative or DBSCAN with distance thresholds
- **Diarization**: pyannote.audio model, overlap detection
- **Active Speaker**: Window sizes, sync thresholds
- **Emotion Recognition**: Model (wav2vec2, ECAPA-TDNN), discrete vs continuous
- **Change Detection**: PELT, kernel CPD, penalty values
- **Clip Export**: Padding before/after, codec settings

## 🎬 Example Outputs

### Face Tracking (`tracks_faces.json`)
```json
{
  "persons": [
    {
      "person_id": "person_1",
      "track_ids": [3, 17],
      "segments": [
        {"t0": 12.04, "t1": 28.20, "frames": 162}
      ],
      "embedding_mean": [0.12, -0.45, ...]
    }
  ]
}
```

### A/V Mapping (`av_map.json`)
```json
{
  "av_links": [
    {
      "person_id": "person_1",
      "speaker_id": "SPEAKER_00",
      "confidence": 0.87,
      "notes": "on-screen; strong lip-audio sync"
    },
    {
      "person_id": null,
      "speaker_id": "SPEAKER_02",
      "confidence": 0.74,
      "notes": "off-screen speaker"
    }
  ]
}
```

### Emotion Changes (`emotion_changes.json`)
```json
{
  "emotion_changes": {
    "SPEAKER_00": [
      {
        "t": 14.1,
        "from": {"label": "neutral", "valence": 0.1, "arousal": 0.0},
        "to": {"label": "happy", "valence": 0.6, "arousal": 0.5},
        "reason": "vocal emotion change"
      }
    ]
  }
}
```

## 🎯 How It Handles Edge Cases

1. **Multiple people talking at once**: Diarization produces overlapping segments; ASD scores each visible face; Hungarian matching assigns optimally
2. **Visible person not talking**: Low ASD score → no match → that speaker maps to background/off-screen
3. **Off-screen speakers**: Diarized segments with no high ASD scores → marked as `off_screen_speaker`
4. **Identity switches**: Embeddings + temporal smoothing re-associate tracks across scenes

## 🧠 Model Choices

| Component | Default Model | Alternatives |
|-----------|---------------|--------------|
| Face Detection | RetinaFace | YOLOv8-face, MediaPipe, OpenCV cascade |
| Face Tracking | BYTETrack | DeepSORT |
| Face Embedding | FaceNet | ArcFace, InsightFace |
| Diarization | pyannote/speaker-diarization-3.1 | SpeechBrain diarization |
| Emotion Recognition | wav2vec2-SER | ECAPA-TDNN, acoustic features |
| Change Detection | PELT (ruptures) | Kernel CPD, BottomUp |

## 📊 Performance Tips

- **Speed vs Accuracy**: Lower `target_fps` in config (e.g., 5-10 FPS for film analysis)
- **GPU Usage**: Set `use_gpu: true` and `mixed_precision: true` in config
- **Caching**: Enable `cache_embeddings: true` to speed up re-runs
- **Clip Codec**: Use `codec: copy` for fast extraction (slight timestamp imprecision) or `libx264` for precise cuts

## 🛠️ Running Individual Stages

You can run pipeline stages independently:

```bash
# 1. Extract audio
python src/extract_media.py video.mp4

# 2. Face detection/tracking
python src/faces_track_cluster.py video.mp4

# 3. Diarization
python src/diarize.py data/audio.wav

# 4. Active speaker detection
python src/active_speaker.py video.mp4 data/audio.wav

# 5. A/V matching
python src/av_match.py

# 6. Emotion change detection
python src/emotion_change.py data/audio.wav

# 7. Export clips
python src/export_clips.py video.mp4
```

## 🧪 Testing

```bash
# Test with a short sample video
python src/pipeline.py sample_video.mp4 --output test_output/

# Check logs
cat outputs/pipeline.log

# Review clips
ls outputs/clips/
```

## 📝 Output Format Details

All JSON outputs use consistent timestamp units (seconds as floats). RTTM format is compatible with standard diarization evaluation tools (pyannote-metrics, NIST tools).

## 🔍 Troubleshooting

**Issue**: `pyannote.audio` fails to load model
- **Solution**: Accept HuggingFace model terms, provide token via `HF_TOKEN` env var

**Issue**: Face detection is slow
- **Solution**: Reduce `target_fps`, use OpenCV cascade instead of RetinaFace

**Issue**: No audio stream found
- **Solution**: Check video has audio track with `ffprobe -i video.mp4`

**Issue**: Emotion changes not detected
- **Solution**: Lower `penalty` in change_detection config; check audio quality

**Issue**: Clips have wrong timestamps
- **Solution**: Use `codec: libx264` instead of `copy` for precise cuts

## 📚 Citation & References

This pipeline integrates techniques from:
- **BYTETrack**: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
- **pyannote.audio**: Bredin et al., "pyannote.audio 2.1: speaker diarization"
- **SyncNet**: Chung & Zisserman, "Out of time: automated lip sync in the wild"
- **ruptures**: Truong et al., "Selective review of offline change point detection methods"

## 🤝 Contributing

Improvements welcome! Key areas:
- Additional emotion models (facial expression, multimodal fusion)
- Better active speaker models (TalkNet, AVA-ActiveSpeaker)
- Scene boundary detection
- Speaker identification (match to known voices)

## 📄 License

MIT License - see LICENSE file for details

## 🎉 Acknowledgments

Built with ❤️ using PyTorch, OpenCV, librosa, pyannote.audio, and ruptures.

---

**Happy analyzing! 🎬🔍✨**

For questions or issues, check the logs in `outputs/pipeline.log` or open an issue on GitHub.
