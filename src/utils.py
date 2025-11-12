"""
Utility functions for the A/V Emotion Detection Pipeline
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Create logger
    logger = logging.getLogger('av_pipeline')
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_config.get('save_logs', True):
        log_file = log_config.get('log_file', 'outputs/pipeline.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_timestamp(seconds: float, format: str = 'hms') -> str:
    """Format seconds to human-readable timestamp"""
    if format == 'hms':
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{seconds:.3f}s"


def time_overlap(seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
    """Calculate overlap between two time segments"""
    start = max(seg1[0], seg2[0])
    end = min(seg1[1], seg2[1])
    overlap = max(0, end - start)
    return overlap


def iou_overlap(seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
    """Calculate IoU (Intersection over Union) for time segments"""
    intersection = time_overlap(seg1, seg2)
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - intersection
    return intersection / union if union > 0 else 0.0


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def bbox_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU between two bounding boxes
    Format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def ensure_dir(path: str):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video metadata using ffprobe"""
    import subprocess
    import json
    
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    
    # Extract video stream info
    video_stream = next(
        (s for s in info['streams'] if s['codec_type'] == 'video'),
        None
    )
    
    if video_stream:
        return {
            'duration': float(info['format']['duration']),
            'width': video_stream['width'],
            'height': video_stream['height'],
            'fps': eval(video_stream['r_frame_rate']),
            'codec': video_stream['codec_name']
        }
    return {}


class ProgressTracker:
    """Simple progress tracking for pipeline stages"""
    
    def __init__(self, total_stages: int, logger: logging.Logger):
        self.total_stages = total_stages
        self.current_stage = 0
        self.logger = logger
        self.start_time = datetime.now()
        
    def next_stage(self, stage_name: str):
        """Move to next stage"""
        self.current_stage += 1
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress = (self.current_stage / self.total_stages) * 100
        
        self.logger.info(
            f"[{self.current_stage}/{self.total_stages}] {stage_name} "
            f"({progress:.1f}% complete, {elapsed:.1f}s elapsed)"
        )
        
    def complete(self):
        """Mark pipeline as complete"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"Pipeline complete! Total time: {elapsed:.1f}s "
            f"({elapsed/60:.1f} minutes)"
        )
