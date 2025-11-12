"""
Active Speaker Detection - determine which visible person is speaking
Uses lip-audio synchronization
"""

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import cv2


class ActiveSpeakerDetector:
    """Detect which visible face is actively speaking"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.asd_config = config['active_speaker']
        self.logger = logger or logging.getLogger(__name__)
        
        self.window_size = self.asd_config['window_size']
        self.hop_size = self.asd_config['hop_size']
        self.sync_threshold = self.asd_config['sync_threshold']
        
    def compute_lip_motion(self, face_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Compute lip motion energy from face sequence
        
        Args:
            face_sequence: List of face crops (grayscale or BGR)
            
        Returns:
            Motion energy time series
        """
        if len(face_sequence) < 2:
            return np.array([0.0])
            
        motion = []
        
        for i in range(1, len(face_sequence)):
            # Simple frame difference in mouth region (bottom third of face)
            prev = face_sequence[i-1]
            curr = face_sequence[i]
            
            if len(prev.shape) == 3:
                prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            if len(curr.shape) == 3:
                curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                
            # Focus on mouth region (bottom third)
            h = prev.shape[0]
            mouth_region_prev = prev[2*h//3:, :]
            mouth_region_curr = curr[2*h//3:, :]
            
            # Frame difference
            diff = cv2.absdiff(mouth_region_prev, mouth_region_curr)
            energy = np.mean(diff)
            motion.append(energy)
            
        return np.array(motion)
        
    def compute_audio_energy(self, audio_segment: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Compute audio energy in windows
        
        Args:
            audio_segment: Audio samples
            sr: Sample rate
            
        Returns:
            Energy time series
        """
        # RMS energy in windows
        window_samples = int(self.window_size * sr)
        hop_samples = int(self.hop_size * sr)
        
        energies = []
        
        for start in range(0, len(audio_segment) - window_samples, hop_samples):
            window = audio_segment[start:start + window_samples]
            energy = np.sqrt(np.mean(window ** 2))
            energies.append(energy)
            
        return np.array(energies)
        
    def compute_sync_score(
        self, 
        lip_motion: np.ndarray, 
        audio_energy: np.ndarray
    ) -> float:
        """
        Compute lip-audio synchronization score using cross-correlation
        
        Args:
            lip_motion: Lip motion energy time series
            audio_energy: Audio energy time series
            
        Returns:
            Sync score (0-1, higher = better sync)
        """
        # Normalize
        lip_motion = (lip_motion - np.mean(lip_motion)) / (np.std(lip_motion) + 1e-7)
        audio_energy = (audio_energy - np.mean(audio_energy)) / (np.std(audio_energy) + 1e-7)
        
        # Make same length
        min_len = min(len(lip_motion), len(audio_energy))
        lip_motion = lip_motion[:min_len]
        audio_energy = audio_energy[:min_len]
        
        if min_len < 2:
            return 0.0
            
        # Cross-correlation
        correlation = np.correlate(lip_motion, audio_energy, mode='valid')
        max_corr = np.max(np.abs(correlation))
        
        # Normalize to 0-1
        score = max_corr / (min_len + 1e-7)
        score = np.clip(score, 0, 1)
        
        return float(score)
        
    def score_person_segments(
        self,
        video_path: str,
        audio_path: str,
        persons_data: Dict[str, Any],
        time_window: float = 5.0
    ) -> Dict[str, Any]:
        """
        Score each person's face tracks for active speaking
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            persons_data: Face tracking results
            time_window: Time window for scoring (seconds)
            
        Returns:
            Active speaker scores per person and time segment
        """
        import librosa
        from extract_media import MediaExtractor
        
        self.logger.info("Computing active speaker scores...")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Initialize extractor for frame access
        extractor = MediaExtractor(self.config, self.logger)
        cap = extractor.get_video_reader(video_path)
        fps = extractor.fps
        
        # Storage for scores
        person_scores = {}
        
        # Process each person
        for person_data in persons_data['persons']:
            person_id = person_data['person_id']
            self.logger.info(f"Processing {person_id}...")
            
            person_scores[person_id] = []
            
            # Process each segment
            for segment in person_data['segments']:
                t0, t1 = segment['t0'], segment['t1']
                
                # Extract face sequence
                face_sequence = []
                
                for t in np.arange(t0, t1, 1.0 / fps):
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Need bbox info - simplified here
                        # In real implementation, use track history
                        h, w = frame.shape[:2]
                        face_crop = frame[h//4:3*h//4, w//4:3*w//4]
                        face_crop = cv2.resize(face_crop, (64, 64))
                        face_sequence.append(face_crop)
                        
                if len(face_sequence) < 2:
                    continue
                    
                # Compute lip motion
                lip_motion = self.compute_lip_motion(face_sequence)
                
                # Get corresponding audio
                start_sample = int(t0 * sr)
                end_sample = int(t1 * sr)
                audio_seg = audio[start_sample:end_sample]
                
                if len(audio_seg) == 0:
                    continue
                    
                # Compute audio energy
                audio_energy = self.compute_audio_energy(audio_seg, sr)
                
                # Compute sync score
                sync_score = self.compute_sync_score(lip_motion, audio_energy)
                
                person_scores[person_id].append({
                    'start': t0,
                    'end': t1,
                    'sync_score': sync_score,
                    'is_speaking': sync_score >= self.sync_threshold
                })
                
        cap.release()
        
        result = {
            'person_scores': person_scores,
            'sync_threshold': self.sync_threshold
        }
        
        return result


def main():
    """Test active speaker detection"""
    import sys
    from utils import load_config, setup_logging, save_json, load_json
    
    config = load_config()
    logger = setup_logging(config)
    
    if len(sys.argv) < 3:
        logger.error("Usage: python active_speaker.py <video_path> <audio_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    
    # Load face tracking results
    faces_path = os.path.join(config['paths']['output_dir'], 'tracks_faces.json')
    
    if not os.path.exists(faces_path):
        logger.error(f"Face tracking results not found: {faces_path}")
        logger.error("Run faces_track_cluster.py first")
        sys.exit(1)
        
    persons_data = load_json(faces_path)
    
    # Run ASD
    detector = ActiveSpeakerDetector(config, logger)
    result = detector.score_person_segments(video_path, audio_path, persons_data)
    
    # Save results
    output_path = os.path.join(config['paths']['output_dir'], 'active_speaker.json')
    save_json(result, output_path)
    
    logger.info(f"Active speaker detection complete: {output_path}")
    
    # Show summary
    for person_id, scores in result['person_scores'].items():
        speaking_segs = [s for s in scores if s['is_speaking']]
        total_speaking = sum(s['end'] - s['start'] for s in speaking_segs)
        avg_score = np.mean([s['sync_score'] for s in scores]) if scores else 0
        
        logger.info(
            f"{person_id}: {len(speaking_segs)}/{len(scores)} speaking segments, "
            f"{total_speaking:.1f}s, avg_score={avg_score:.3f}"
        )


if __name__ == '__main__':
    main()
