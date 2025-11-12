"""
Emotion/tone change detection module
Detects changes in vocal emotion using speech emotion recognition and change-point detection
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import librosa


class SpeechEmotionRecognizer:
    """Speech Emotion Recognition using acoustic features"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.emotion_config = config['emotion']
        self.logger = logger or logging.getLogger(__name__)
        
        self.window_size = self.emotion_config['window_size']
        self.hop_size = self.emotion_config['hop_size']
        self.emotions = self.emotion_config['emotions']
        self.continuous = self.emotion_config['continuous']
        
        self._load_model()
        
    def _load_model(self):
        """Load emotion recognition model"""
        self.logger.info("Loading emotion recognition model...")
        
        try:
            # Try to load transformer-based model
            from transformers import pipeline
            
            try:
                self.model = pipeline(
                    "audio-classification",
                    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                )
                self.model_type = 'transformer'
                self.logger.info("Loaded wav2vec2 emotion model")
            except Exception as e:
                self.logger.warning(f"Failed to load transformer model: {e}")
                self.model = None
                self.model_type = 'features'
                self.logger.info("Using acoustic features for emotion detection")
                
        except ImportError:
            self.model = None
            self.model_type = 'features'
            self.logger.info("Using acoustic features for emotion detection")
            
    def extract_acoustic_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract acoustic features for emotion recognition
        
        Returns:
            Feature vector
        """
        features = []
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend([
            np.mean(mfccs),
            np.std(mfccs),
            np.min(mfccs),
            np.max(mfccs)
        ])
        
        # Pitch (F0)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
                
        if pitch_values:
            features.extend([
                np.mean(pitch_values),
                np.std(pitch_values),
                np.min(pitch_values),
                np.max(pitch_values)
            ])
        else:
            features.extend([0, 0, 0, 0])
            
        # Energy
        energy = librosa.feature.rms(y=audio)[0]
        features.extend([
            np.mean(energy),
            np.std(energy)
        ])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.extend([
            np.mean(zcr),
            np.std(zcr)
        ])
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff)
        ])
        
        return np.array(features)
        
    def predict_emotion(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Predict emotion from audio segment
        
        Returns:
            Dictionary with emotion label and/or valence-arousal
        """
        if len(audio) == 0:
            return {'valence': 0.0, 'arousal': 0.0, 'label': 'neutral'}
            
        if self.model_type == 'transformer' and self.model is not None:
            try:
                # Run model
                result = self.model(audio, sampling_rate=sr)
                
                # Get top prediction
                top = max(result, key=lambda x: x['score'])
                
                # Map to valence-arousal (simplified)
                emotion_map = {
                    'happy': (0.7, 0.6),
                    'sad': (-0.6, -0.3),
                    'angry': (-0.5, 0.8),
                    'fear': (-0.4, 0.7),
                    'neutral': (0.0, 0.0),
                    'disgust': (-0.6, 0.4),
                    'surprise': (0.3, 0.7)
                }
                
                label = top['label'].lower()
                valence, arousal = emotion_map.get(label, (0.0, 0.0))
                
                return {
                    'label': label,
                    'confidence': top['score'],
                    'valence': valence,
                    'arousal': arousal
                }
                
            except Exception as e:
                self.logger.warning(f"Model prediction failed: {e}")
                
        # Fallback: use acoustic features to estimate valence/arousal
        features = self.extract_acoustic_features(audio, sr)
        
        # Simple heuristics (not accurate, but better than nothing)
        # High pitch + high energy → high arousal
        # Low energy + low pitch → low valence
        
        pitch_mean = features[4]  # Mean pitch
        energy_mean = features[8]  # Mean energy
        
        # Normalize (assuming typical ranges)
        arousal = np.clip((energy_mean - 0.02) / 0.05, 0, 1) * 2 - 1
        valence = np.clip((pitch_mean - 100) / 100, 0, 1) * 2 - 1
        
        # Simple emotion classification
        if valence > 0.3 and arousal > 0.3:
            label = 'happy'
        elif valence < -0.3 and arousal < -0.3:
            label = 'sad'
        elif valence < -0.3 and arousal > 0.3:
            label = 'angry'
        else:
            label = 'neutral'
            
        return {
            'label': label,
            'confidence': 0.5,
            'valence': float(valence),
            'arousal': float(arousal),
            'features': features.tolist()
        }
        
    def process_speaker_segments(
        self,
        audio_path: str,
        speaker_segments: List[Dict[str, float]]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Process speaker segments and extract emotion time series
        
        Returns:
            (timestamps, emotion_predictions)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        timestamps = []
        predictions = []
        
        # Process each segment
        for segment in speaker_segments:
            start_time = segment['start']
            end_time = segment['end']
            
            # Extract audio segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Window-based processing
            window_samples = int(self.window_size * sr)
            hop_samples = int(self.hop_size * sr)
            
            for i in range(0, len(segment_audio) - window_samples, hop_samples):
                window = segment_audio[i:i + window_samples]
                timestamp = start_time + (i / sr)
                
                # Predict emotion
                emotion = self.predict_emotion(window, sr)
                
                timestamps.append(timestamp)
                predictions.append(emotion)
                
        return timestamps, predictions


class ChangePointDetector:
    """Detect emotion/tone change points using ruptures"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.cpd_config = config['change_detection']
        self.logger = logger or logging.getLogger(__name__)
        
        self.method = self.cpd_config['method']
        self.model = self.cpd_config['model']
        self.penalty = self.cpd_config['penalty']
        self.min_size = self.cpd_config['min_size']
        self.merge_threshold = self.cpd_config['merge_threshold']
        
    def detect_changes(
        self,
        timestamps: List[float],
        values: np.ndarray
    ) -> List[float]:
        """
        Detect change points in time series
        
        Args:
            timestamps: Time stamps for each value
            values: Value array (can be multi-dimensional)
            
        Returns:
            List of change point timestamps
        """
        if len(timestamps) < self.min_size * 2:
            self.logger.warning("Not enough data for change point detection")
            return []
            
        try:
            import ruptures as rpt
            
            # Ensure 2D array
            if len(values.shape) == 1:
                values = values.reshape(-1, 1)
                
            # Detect change points
            if self.method == 'pelt':
                algo = rpt.Pelt(model=self.model, min_size=self.min_size)
            elif self.method == 'kernelcpd':
                algo = rpt.KernelCPD(kernel="rbf", min_size=self.min_size)
            elif self.method == 'bottomup':
                algo = rpt.BottomUp(model=self.model, min_size=self.min_size)
            else:
                algo = rpt.Pelt(model="rbf", min_size=self.min_size)
                
            algo.fit(values)
            change_indices = algo.predict(pen=self.penalty)
            
            # Convert indices to timestamps
            change_times = [timestamps[i-1] for i in change_indices[:-1]]  # Exclude last point
            
            # Merge nearby changes
            merged_changes = self._merge_nearby_changes(change_times)
            
            self.logger.info(
                f"Detected {len(change_indices)-1} change points, "
                f"merged to {len(merged_changes)}"
            )
            
            return merged_changes
            
        except ImportError:
            self.logger.error("ruptures not installed. Install with: pip install ruptures")
            return self._simple_change_detection(timestamps, values)
            
    def _merge_nearby_changes(self, change_times: List[float]) -> List[float]:
        """Merge change points that are very close together"""
        if len(change_times) <= 1:
            return change_times
            
        merged = [change_times[0]]
        
        for t in change_times[1:]:
            if t - merged[-1] >= self.merge_threshold:
                merged.append(t)
                
        return merged
        
    def _simple_change_detection(
        self,
        timestamps: List[float],
        values: np.ndarray
    ) -> List[float]:
        """Simple threshold-based change detection fallback"""
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)
            
        # Compute differences
        diffs = np.diff(values, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        
        # Threshold
        threshold = np.mean(norms) + 2 * np.std(norms)
        
        change_indices = np.where(norms > threshold)[0]
        change_times = [timestamps[i] for i in change_indices]
        
        return self._merge_nearby_changes(change_times)


def process_all_speakers(
    audio_path: str,
    diarization_data: Dict[str, Any],
    av_map: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Process all speakers and detect emotion changes
    
    Returns:
        Dictionary with emotion changes per speaker
    """
    recognizer = SpeechEmotionRecognizer(config, logger)
    detector = ChangePointDetector(config, logger)
    
    emotion_changes = {}
    
    # Process each speaker
    for speaker, segments in diarization_data['speakers'].items():
        logger.info(f"Processing {speaker}...")
        
        # Get emotion time series
        timestamps, predictions = recognizer.process_speaker_segments(
            audio_path, segments
        )
        
        if len(predictions) == 0:
            continue
            
        # Extract valence-arousal for change detection
        valence_arousal = np.array([
            [p['valence'], p['arousal']] for p in predictions
        ])
        
        # Detect change points
        change_times = detector.detect_changes(timestamps, valence_arousal)
        
        # Build change point details
        speaker_changes = []
        
        for i, change_time in enumerate(change_times):
            # Find nearest prediction indices before and after change
            idx = np.searchsorted(timestamps, change_time)
            
            if idx > 0 and idx < len(predictions):
                before = predictions[idx - 1]
                after = predictions[idx]
                
                change_info = {
                    't': change_time,
                    'from': {
                        'label': before['label'],
                        'valence': before['valence'],
                        'arousal': before['arousal']
                    },
                    'to': {
                        'label': after['label'],
                        'valence': after['valence'],
                        'arousal': after['arousal']
                    },
                    'reason': 'vocal emotion change'
                }
                
                speaker_changes.append(change_info)
                
        emotion_changes[speaker] = speaker_changes
        
        logger.info(f"{speaker}: {len(speaker_changes)} emotion changes detected")
        
    result = {
        'emotion_changes': emotion_changes,
        'total_changes': sum(len(changes) for changes in emotion_changes.values())
    }
    
    return result


def main():
    """Run emotion change detection"""
    import sys
    from utils import load_config, setup_logging, save_json, load_json
    
    config = load_config()
    logger = setup_logging(config)
    
    if len(sys.argv) < 2:
        logger.error("Usage: python emotion_change.py <audio_path>")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    
    # Load required data
    output_dir = config['paths']['output_dir']
    diarization_path = os.path.join(output_dir, 'diarization.json')
    av_map_path = os.path.join(output_dir, 'av_map.json')
    
    if not os.path.exists(diarization_path):
        logger.error("Diarization results not found. Run diarize.py first")
        sys.exit(1)
        
    if not os.path.exists(av_map_path):
        logger.error("A/V mapping not found. Run av_match.py first")
        sys.exit(1)
        
    diarization_data = load_json(diarization_path)
    av_map = load_json(av_map_path)
    
    # Process all speakers
    result = process_all_speakers(audio_path, diarization_data, av_map, config, logger)
    
    # Save results
    output_path = os.path.join(output_dir, 'emotion_changes.json')
    save_json(result, output_path)
    
    logger.info(f"Emotion change detection complete: {output_path}")
    logger.info(f"Total changes detected: {result['total_changes']}")


if __name__ == '__main__':
    main()
