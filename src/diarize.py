"""
Speaker diarization module using pyannote.audio
Identifies who spoke when, including overlapping speech
"""

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from collections import defaultdict


class SpeakerDiarizer:
    """Speaker diarization using pyannote.audio"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.diar_config = config['diarization']
        self.logger = logger or logging.getLogger(__name__)
        
        self._load_pipeline()
        
    def _load_pipeline(self):
        """Load pyannote diarization pipeline"""
        self.logger.info("Loading pyannote.audio diarization pipeline...")
        
        try:
            from pyannote.audio import Pipeline
            
            # Load pre-trained pipeline
            model_name = self.diar_config['model']
            
            try:
                self.pipeline = Pipeline.from_pretrained(model_name)
                self.logger.info(f"Loaded pipeline: {model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load {model_name}: {e}")
                self.logger.info("Attempting to use default pipeline...")
                self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
                
        except ImportError:
            self.logger.error(
                "pyannote.audio not installed. "
                "Install with: pip install pyannote.audio"
            )
            self.pipeline = None
            
    def diarize(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file (WAV format)
            
        Returns:
            Dictionary with speaker segments and timeline
        """
        if self.pipeline is None:
            self.logger.error("Diarization pipeline not available")
            return self._fallback_diarization(audio_path)
            
        self.logger.info(f"Running diarization on {audio_path}")
        
        try:
            # Run diarization
            diarization = self.pipeline(audio_path)
            
            # Parse results
            speakers = defaultdict(list)
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers[speaker].append({
                    'start': turn.start,
                    'end': turn.end,
                    'duration': turn.end - turn.start
                })
                
            # Sort segments by start time
            for speaker in speakers:
                speakers[speaker].sort(key=lambda x: x['start'])
                
            result = {
                'speakers': dict(speakers),
                'num_speakers': len(speakers),
                'total_speech_time': sum(
                    seg['duration'] 
                    for segs in speakers.values() 
                    for seg in segs
                )
            }
            
            self.logger.info(
                f"Found {result['num_speakers']} speakers, "
                f"{result['total_speech_time']:.1f}s total speech"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Diarization failed: {e}")
            return self._fallback_diarization(audio_path)
            
    def _fallback_diarization(self, audio_path: str) -> Dict[str, Any]:
        """Simple VAD-based fallback when pyannote is not available"""
        self.logger.warning("Using fallback VAD-based diarization")
        
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Simple energy-based VAD
            frame_length = int(0.025 * sr)  # 25ms
            hop_length = int(0.010 * sr)    # 10ms
            
            energy = librosa.feature.rms(
                y=y, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            # Threshold
            threshold = np.mean(energy) * 0.5
            speech_frames = energy > threshold
            
            # Convert to segments
            segments = []
            in_speech = False
            start_time = 0
            
            for i, is_speech in enumerate(speech_frames):
                time = i * hop_length / sr
                
                if is_speech and not in_speech:
                    start_time = time
                    in_speech = True
                elif not is_speech and in_speech:
                    segments.append({
                        'start': start_time,
                        'end': time,
                        'duration': time - start_time
                    })
                    in_speech = False
                    
            # Assign all to single speaker
            result = {
                'speakers': {'SPEAKER_00': segments},
                'num_speakers': 1,
                'total_speech_time': sum(s['duration'] for s in segments)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fallback diarization failed: {e}")
            return {
                'speakers': {},
                'num_speakers': 0,
                'total_speech_time': 0.0
            }
            
    def export_rttm(self, diarization_result: Dict[str, Any], output_path: str):
        """
        Export diarization to RTTM format
        
        RTTM format: SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for speaker, segments in diarization_result['speakers'].items():
                for seg in segments:
                    line = (
                        f"SPEAKER file 1 {seg['start']:.3f} {seg['duration']:.3f} "
                        f"<NA> <NA> {speaker} <NA> <NA>\n"
                    )
                    f.write(line)
                    
        self.logger.info(f"Exported RTTM to {output_path}")
        
    def get_speaker_at_time(
        self, 
        diarization_result: Dict[str, Any], 
        timestamp: float
    ) -> Optional[str]:
        """
        Get speaker ID at specific timestamp
        
        Args:
            diarization_result: Output from diarize()
            timestamp: Time in seconds
            
        Returns:
            Speaker ID or None if no speaker at that time
        """
        for speaker, segments in diarization_result['speakers'].items():
            for seg in segments:
                if seg['start'] <= timestamp <= seg['end']:
                    return speaker
        return None
        
    def get_overlapping_speakers(
        self, 
        diarization_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find time periods where multiple speakers overlap
        
        Returns:
            List of overlap periods with participating speakers
        """
        overlaps = []
        
        speakers = list(diarization_result['speakers'].keys())
        
        for i, speaker1 in enumerate(speakers):
            for speaker2 in speakers[i+1:]:
                segs1 = diarization_result['speakers'][speaker1]
                segs2 = diarization_result['speakers'][speaker2]
                
                # Check for overlaps
                for seg1 in segs1:
                    for seg2 in segs2:
                        start = max(seg1['start'], seg2['start'])
                        end = min(seg1['end'], seg2['end'])
                        
                        if start < end:
                            overlaps.append({
                                'start': start,
                                'end': end,
                                'duration': end - start,
                                'speakers': [speaker1, speaker2]
                            })
                            
        overlaps.sort(key=lambda x: x['start'])
        return overlaps


def main():
    """Run speaker diarization"""
    import sys
    from utils import load_config, setup_logging, save_json
    
    config = load_config()
    logger = setup_logging(config)
    
    if len(sys.argv) < 2:
        logger.error("Usage: python diarize.py <audio_path>")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    
    # Output paths
    output_json = os.path.join(config['paths']['output_dir'], 'diarization.json')
    output_rttm = os.path.join(config['paths']['output_dir'], 'diarization.rttm')
    
    # Run diarization
    diarizer = SpeakerDiarizer(config, logger)
    result = diarizer.diarize(audio_path)
    
    # Export results
    save_json(result, output_json)
    diarizer.export_rttm(result, output_rttm)
    
    logger.info(f"Diarization complete!")
    logger.info(f"JSON: {output_json}")
    logger.info(f"RTTM: {output_rttm}")
    
    # Show summary
    for speaker, segments in result['speakers'].items():
        total_time = sum(s['duration'] for s in segments)
        logger.info(f"{speaker}: {len(segments)} segments, {total_time:.1f}s total")
        
    # Check for overlaps
    overlaps = diarizer.get_overlapping_speakers(result)
    if overlaps:
        total_overlap = sum(o['duration'] for o in overlaps)
        logger.info(f"Found {len(overlaps)} overlapping speech periods, {total_overlap:.1f}s total")


if __name__ == '__main__':
    main()
