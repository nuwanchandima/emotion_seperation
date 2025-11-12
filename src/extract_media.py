"""
Media extraction module - Extract audio and video frames from input video
"""

import os
import subprocess
import logging
from typing import Dict, Any, Optional
import cv2
import numpy as np
from pathlib import Path


class MediaExtractor:
    """Extract and preprocess audio and video from input file"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.data_dir = config['paths']['data_dir']
        self.audio_config = config['audio']
        self.video_config = config['video']
        
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video as mono WAV at 16kHz
        
        Args:
            video_path: Path to input video file
            output_path: Optional output path for audio file
            
        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            output_path = os.path.join(self.data_dir, 'audio.wav')
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.logger.info(f"Extracting audio from {video_path}")
        
        cmd = [
            'ffmpeg',
            '-y',  # overwrite
            '-i', video_path,
            '-ac', str(self.audio_config['channels']),
            '-ar', str(self.audio_config['sample_rate']),
            '-vn',  # no video
            output_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info(f"Audio extracted successfully to {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to extract audio: {e.stderr}")
            raise
            
    def get_video_reader(self, video_path: str) -> cv2.VideoCapture:
        """
        Get OpenCV video reader with optional frame rate adjustment
        
        Args:
            video_path: Path to video file
            
        Returns:
            OpenCV VideoCapture object
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
            
        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps
        
        self.logger.info(
            f"Video: {self.width}x{self.height}, "
            f"{self.fps:.2f} FPS, "
            f"{self.duration:.2f}s, "
            f"{self.frame_count} frames"
        )
        
        return cap
        
    def frame_generator(self, video_path: str, target_fps: Optional[float] = None):
        """
        Generate video frames with optional downsampling
        
        Args:
            video_path: Path to video file
            target_fps: Target FPS for processing (None = use original)
            
        Yields:
            Tuple of (frame_number, timestamp, frame_image)
        """
        cap = self.get_video_reader(video_path)
        
        if target_fps is None:
            target_fps = self.video_config.get('target_fps', self.fps)
            
        # Calculate frame skip
        if target_fps >= self.fps:
            frame_skip = 1
        else:
            frame_skip = int(self.fps / target_fps)
            
        self.logger.info(f"Processing at {target_fps} FPS (every {frame_skip} frames)")
        
        frame_idx = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame at target rate
                if frame_idx % frame_skip == 0:
                    timestamp = frame_idx / self.fps
                    
                    # Optional resize
                    target_width = self.video_config.get('frame_width')
                    if target_width and target_width != self.width:
                        scale = target_width / self.width
                        target_height = int(self.height * scale)
                        frame = cv2.resize(frame, (target_width, target_height))
                        
                    yield frame_idx, timestamp, frame
                    processed_count += 1
                    
                frame_idx += 1
                
        finally:
            cap.release()
            self.logger.info(f"Processed {processed_count} frames")
            
    def extract_frame_at_time(self, video_path: str, timestamp: float) -> np.ndarray:
        """
        Extract a single frame at specific timestamp
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds
            
        Returns:
            Frame image as numpy array
        """
        cap = cv2.VideoCapture(video_path)
        
        # Seek to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to extract frame at {timestamp}s")
            
        return frame
        
    def validate_input(self, video_path: str) -> Dict[str, Any]:
        """
        Validate input video and return metadata
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        # Get video info using ffprobe
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise ValueError(f"Invalid video file: {video_path}")
            
        import json
        info = json.loads(result.stdout)
        
        # Check for video and audio streams
        has_video = any(s['codec_type'] == 'video' for s in info['streams'])
        has_audio = any(s['codec_type'] == 'audio' for s in info['streams'])
        
        if not has_video:
            raise ValueError("No video stream found")
        if not has_audio:
            self.logger.warning("No audio stream found - diarization will fail")
            
        return {
            'path': video_path,
            'duration': float(info['format']['duration']),
            'has_video': has_video,
            'has_audio': has_audio,
            'format': info['format']['format_name']
        }


def main():
    """Test media extraction"""
    import sys
    from utils import load_config, setup_logging
    
    config = load_config()
    logger = setup_logging(config)
    
    if len(sys.argv) < 2:
        logger.error("Usage: python extract_media.py <video_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    extractor = MediaExtractor(config, logger)
    
    # Validate
    info = extractor.validate_input(video_path)
    logger.info(f"Video info: {info}")
    
    # Extract audio
    audio_path = extractor.extract_audio(video_path)
    logger.info(f"Audio saved to: {audio_path}")
    
    # Test frame generator
    logger.info("Testing frame extraction...")
    count = 0
    for frame_idx, timestamp, frame in extractor.frame_generator(video_path):
        count += 1
        if count <= 3:
            logger.info(f"Frame {frame_idx} at {timestamp:.2f}s: {frame.shape}")
        if count >= 10:
            break
            
    logger.info("Media extraction test complete!")


if __name__ == '__main__':
    main()
