# Project Overview & Architecture

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT VIDEO FILE                           │
│                      (e.g., interview.mp4)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: MEDIA EXTRACTION                        │
│  ┌─────────────────┐                    ┌─────────────────┐        │
│  │  Extract Audio  │                    │ Frame Generator │        │
│  │  (FFmpeg WAV)   │                    │  (OpenCV cap)   │        │
│  └────────┬────────┘                    └────────┬────────┘        │
│           │                                      │                  │
│           ▼                                      ▼                  │
│      audio.wav                           frame iterator             │
└────────────┬────────────────────────────────────┬────────────────────┘
             │                                     │
    ┌────────┴────────┐                   ┌────────┴────────┐
    │                 │                   │                 │
    ▼                 ▼                   ▼                 ▼
┌────────────┐  ┌────────────┐    ┌────────────┐  ┌────────────┐
│  STAGE 3   │  │  STAGE 6   │    │  STAGE 2   │  │  STAGE 4   │
│ Diarization│  │  Emotion   │    │Face Detect │  │ Active Spk │
│            │  │   Change   │    │   Track    │  │  Detection │
│  pyannote  │  │   SER +    │    │  Cluster   │  │            │
│   .audio   │  │  ruptures  │    │            │  │  Lip-Audio │
└─────┬──────┘  └─────┬──────┘    └─────┬──────┘  │    Sync    │
      │               │                  │         └──────┬─────┘
      │               │                  │                │
      ▼               ▼                  ▼                ▼
  Speaker IDs    Emotion Changes    Person IDs      ASD Scores
  + segments     + timestamps       + tracks        + confidence
      │               │                  │                │
      └───────┬───────┴──────────────────┴────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  STAGE 5: A/V MATCHING                              │
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  Cost Matrix: Person vs Speaker                       │        │
│  │  • Temporal overlap: 40%                              │        │
│  │  • ASD sync score: 60%                                │        │
│  └────────────────────┬───────────────────────────────────┘        │
│                       │                                            │
│                       ▼                                            │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  Hungarian Algorithm (scipy.optimize)                 │        │
│  └────────────────────┬───────────────────────────────────┘        │
│                       │                                            │
│                       ▼                                            │
│          Person ↔ Speaker Links + Confidence                      │
│          (on-screen + off-screen speakers)                         │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  STAGE 7: CLIP EXTRACTION                           │
│                                                                     │
│  For each emotion change timestamp:                                │
│  1. Calculate: start = t - padding_before                          │
│  2. Calculate: duration = padding_before + padding_after           │
│  3. FFmpeg extract: -ss <start> -t <duration>                      │
│                                                                     │
│  Output: person_X_change_NNN_emotion1_to_emotion2_tXX.Xs.mp4      │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUTS DIRECTORY                            │
│                                                                     │
│  JSON Files:                      Other:                           │
│  • tracks_faces.json              • diarization.rttm               │
│  • diarization.json               • clips_manifest.md              │
│  • active_speaker.json            • pipeline.log                   │
│  • av_map.json                    • emotion_timeline.csv           │
│  • emotion_changes.json                                            │
│  • clips_summary.json                                              │
│                                                                     │
│  clips/ directory:                                                 │
│  • person_1_change_000_neutral_to_happy_t14.1s.mp4                │
│  • person_1_change_001_happy_to_sad_t28.5s.mp4                    │
│  • speaker_2_change_000_angry_to_neutral_t45.3s.mp4               │
│  • ... (all emotion change clips)                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 🎬 Stage 1: Media Extraction
**Purpose:** Prepare audio and video for processing
- **Input:** Video file (MP4, AVI, MOV, etc.)
- **Output:** `data/audio.wav` (16kHz mono), frame generator
- **Key Operations:**
  - FFmpeg audio extraction
  - Frame reading with optional downsampling (target_fps)
  - Video metadata collection

### 👤 Stage 2: Face Detection → Tracking → Clustering
**Purpose:** Identify unique persons in video
- **Input:** Video frames
- **Output:** `outputs/tracks_faces.json`
- **Key Operations:**
  1. **Detection:** RetinaFace/YOLOv8/OpenCV finds faces per frame
  2. **Tracking:** BYTETrack assigns persistent track IDs across frames
  3. **Embedding:** FaceNet/ArcFace extracts 512-d face vectors
  4. **Clustering:** Agglomerative clustering groups tracks → person IDs

**Models Used:**
- Face Detector: RetinaFace (GPU) or OpenCV Cascade (CPU fallback)
- Face Embedder: FaceNet (facenet-pytorch)
- Tracker: BYTETrack (IoU-based multi-object tracking)

### 🔊 Stage 3: Speaker Diarization
**Purpose:** Identify who speaks when
- **Input:** `data/audio.wav`
- **Output:** `outputs/diarization.json`, `outputs/diarization.rttm`
- **Key Operations:**
  - Voice Activity Detection (VAD)
  - Speaker embedding extraction
  - Clustering into speaker IDs
  - Overlap detection (handles simultaneous speech)

**Models Used:**
- Primary: pyannote.audio 3.1 (state-of-the-art diarization)
- Fallback: Energy-based VAD with single speaker

### 🎤 Stage 4: Active Speaker Detection
**Purpose:** Determine which visible face is speaking
- **Input:** Video frames + audio segments + face tracks
- **Output:** `outputs/active_speaker.json`
- **Key Operations:**
  1. Extract lip region motion (frame differences in mouth area)
  2. Compute audio energy in sliding windows
  3. Cross-correlation between lip motion and audio
  4. Sync score per person per segment

**Algorithm:**
- Lip motion: frame difference in bottom 1/3 of face
- Audio energy: RMS in 0.25s windows
- Sync score: normalized cross-correlation

### 🔗 Stage 5: Audio-Visual Matching
**Purpose:** Link persons (faces) to speakers (voices)
- **Input:** Face tracks + diarization + ASD scores
- **Output:** `outputs/av_map.json`
- **Key Operations:**
  1. Build cost matrix (persons × speakers):
     - Temporal overlap: 40% weight
     - ASD sync score: 60% weight
  2. Hungarian algorithm (maximize cost)
  3. Assign matches with confidence > threshold
  4. Mark remaining speakers as "off-screen"

**Algorithm:** Linear assignment with scipy.optimize.linear_sum_assignment

### 😊 Stage 6: Emotion Change Detection
**Purpose:** Find timestamps where vocal emotion shifts
- **Input:** `data/audio.wav` + speaker segments
- **Output:** `outputs/emotion_changes.json`
- **Key Operations:**
  1. **SER (Speech Emotion Recognition):**
     - Extract MFCCs, pitch, energy, spectral features
     - Predict valence/arousal or discrete emotions
     - Sliding window (1.5s window, 0.25s hop)
  2. **Change Point Detection:**
     - PELT algorithm (ruptures library)
     - Detect abrupt changes in valence/arousal time series
     - Merge nearby changes (< 0.5s apart)

**Models Used:**
- Primary: wav2vec2-based emotion classifier (transformers)
- Fallback: Acoustic features (MFCCs, pitch, energy)

### 🎥 Stage 7: Clip Extraction
**Purpose:** Generate video clips around emotion changes
- **Input:** Original video + emotion change timestamps
- **Output:** `outputs/clips/*.mp4`, `outputs/clips_manifest.md`
- **Key Operations:**
  - For each change at time t:
    - start = t - padding_before (default 0.5s)
    - duration = padding_before + padding_after (default 1.0s)
    - FFmpeg extract with copy codec (fast) or re-encode (precise)
  - Generate descriptive filenames with emotion labels
  - Create markdown manifest for browsing

## File Format Specifications

### tracks_faces.json
```json
{
  "persons": [
    {
      "person_id": "person_0",
      "track_ids": [3, 17, 24],
      "segments": [
        {"t0": 12.04, "t1": 28.20, "frames": 162},
        {"t0": 35.10, "t1": 45.80, "frames": 107}
      ],
      "embedding_mean": [0.12, -0.45, 0.33, ...] // 512-d vector
    }
  ],
  "total_tracks": 42,
  "total_persons": 3
}
```

### diarization.json
```json
{
  "speakers": {
    "SPEAKER_00": [
      {"start": 0.5, "end": 3.8, "duration": 3.3},
      {"start": 5.2, "end": 8.7, "duration": 3.5}
    ],
    "SPEAKER_01": [
      {"start": 4.1, "end": 5.0, "duration": 0.9}
    ]
  },
  "num_speakers": 2,
  "total_speech_time": 7.7
}
```

### av_map.json
```json
{
  "av_links": [
    {
      "person_id": "person_0",
      "speaker_id": "SPEAKER_00",
      "confidence": 0.87,
      "notes": "on-screen; strong lip-audio sync"
    },
    {
      "person_id": null,
      "speaker_id": "SPEAKER_01",
      "confidence": 0.74,
      "notes": "off-screen speaker"
    }
  ],
  "total_matches": 1,
  "off_screen_speakers": 1
}
```

### emotion_changes.json
```json
{
  "emotion_changes": {
    "SPEAKER_00": [
      {
        "t": 14.1,
        "from": {
          "label": "neutral",
          "valence": 0.1,
          "arousal": 0.0
        },
        "to": {
          "label": "happy",
          "valence": 0.6,
          "arousal": 0.5
        },
        "reason": "vocal emotion change"
      }
    ]
  },
  "total_changes": 12
}
```

## Configuration Hierarchy

```
config.yaml
├── paths/              # Directory settings
├── video/              # Frame processing
├── audio/              # Audio settings
├── face_detection/     # RetinaFace/YOLO/OpenCV
├── face_tracking/      # BYTETrack/DeepSORT
├── face_recognition/   # ArcFace/FaceNet
├── face_clustering/    # Agglomerative/DBSCAN
├── diarization/        # pyannote settings
├── active_speaker/     # ASD windows & thresholds
├── av_matching/        # Hungarian matching
├── emotion/            # SER model & emotions
├── change_detection/   # PELT parameters
├── clips/              # Clip extraction settings
├── performance/        # GPU/batch/cache
└── logging/            # Log level & output
```

## Performance Characteristics

### Processing Time (RTX 3080, 10 FPS)
- 1-minute video: ~2 minutes
- 10-minute video: ~15 minutes
- 1-hour video: ~60 minutes

**Bottlenecks:**
1. Face detection (30-40% of time)
2. Diarization (20-30%)
3. Emotion recognition (15-25%)
4. A/V matching (5-10%)

### Memory Usage
- Peak RAM: ~4-8 GB (CPU)
- Peak VRAM: ~2-4 GB (GPU)
- Storage: ~2-5 MB per minute of video (outputs)

### Scalability
- **Parallel processing:** Can batch process videos independently
- **Streaming:** Not supported (requires full video for face clustering)
- **Incremental:** Can cache embeddings for re-runs with different parameters

## Error Handling Strategy

1. **Graceful Degradation:**
   - If GPU unavailable → use CPU
   - If pyannote fails → use VAD fallback
   - If transformer fails → use acoustic features

2. **Validation:**
   - Check video/audio streams before processing
   - Verify output file existence after each stage
   - Log warnings for low-quality results

3. **Recovery:**
   - Each stage can run independently
   - Outputs saved incrementally
   - Can resume from any stage if previous outputs exist

## Extension Points

Want to add features? Here's where to start:

1. **New face detector:** Modify `FaceDetector` class in `faces_track_cluster.py`
2. **New emotion model:** Modify `SpeechEmotionRecognizer` in `emotion_change.py`
3. **Facial expression:** Add module between ASD and emotion detection
4. **Scene detection:** Add stage before face clustering
5. **Speaker identification:** Add stage after diarization
6. **Multi-modal fusion:** Combine audio+visual emotions in Stage 6

## Dependencies Graph

```
Pipeline Core
├── extract_media.py (FFmpeg, OpenCV)
├── faces_track_cluster.py
│   ├── RetinaFace / OpenCV
│   ├── BYTETrack (scipy)
│   └── FaceNet (facenet-pytorch)
├── diarize.py
│   └── pyannote.audio (HuggingFace)
├── active_speaker.py
│   ├── OpenCV (face crops)
│   └── librosa (audio energy)
├── av_match.py
│   └── scipy (linear_sum_assignment)
├── emotion_change.py
│   ├── librosa (features)
│   ├── transformers (wav2vec2)
│   └── ruptures (change detection)
└── export_clips.py
    └── FFmpeg (clip extraction)

Utilities
└── utils.py (yaml, json, logging)
```

---

**For more details, see:**
- README.md - User guide
- QUICKSTART.md - Getting started
- TROUBLESHOOTING.md - Common issues
- example_usage.py - Code examples
