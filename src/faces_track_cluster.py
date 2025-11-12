"""
Face detection, tracking, and clustering module
Detects faces, tracks them across frames, extracts embeddings, and clusters into person IDs
"""

import os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import cv2
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import pickle


class FaceDetector:
    """Face detection using RetinaFace or other models"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config['face_detection']
        self.logger = logger
        self.model_name = self.config['model']
        self.confidence_threshold = self.config['confidence_threshold']
        
        self._load_model()
        
    def _load_model(self):
        """Load face detection model"""
        self.logger.info(f"Loading face detector: {self.model_name}")
        
        if self.model_name == 'retinaface':
            try:
                from retinaface import RetinaFace
                self.model = RetinaFace
                self.detector_type = 'retinaface'
            except ImportError:
                self.logger.warning("RetinaFace not available, using OpenCV cascade")
                self.model = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.detector_type = 'opencv'
        else:
            # Fallback to OpenCV
            self.model = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.detector_type = 'opencv'
            
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in frame
        
        Returns:
            List of detections with bbox and confidence
            Format: [{'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
        """
        if self.detector_type == 'retinaface':
            try:
                detections = self.model.detect_faces(frame)
                results = []
                
                for key, face_info in detections.items():
                    bbox = face_info['facial_area']  # [x, y, w, h]
                    confidence = face_info['score']
                    
                    if confidence >= self.confidence_threshold:
                        results.append({
                            'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],  # x1,y1,x2,y2
                            'confidence': confidence,
                            'landmarks': face_info.get('landmarks', {})
                        })
                return results
            except:
                pass
                
        # OpenCV fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(self.config['min_face_size'], self.config['min_face_size'])
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 1.0  # OpenCV doesn't give confidence
            })
            
        return results


class ByteTrack:
    """Simple BYTETrack-inspired tracker"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config['face_tracking']
        self.logger = logger
        
        self.max_age = self.config['max_age']
        self.min_hits = self.config['min_hits']
        self.iou_threshold = self.config['iou_threshold']
        
        self.tracks = []
        self.next_id = 0
        
    def update(self, detections: List[Dict[str, Any]], frame_idx: int, timestamp: float):
        """
        Update tracks with new detections
        
        Args:
            detections: List of detection dicts with 'bbox' and 'confidence'
            frame_idx: Current frame number
            timestamp: Current timestamp in seconds
            
        Returns:
            List of active tracks with track_id, bbox, and confidence
        """
        # Predict next positions (simple: use last position)
        for track in self.tracks:
            track['age'] += 1
            
        # Match detections to tracks using IoU
        if len(detections) > 0 and len(self.tracks) > 0:
            matches, unmatched_dets, unmatched_tracks = self._match(detections)
            
            # Update matched tracks
            for det_idx, track_idx in matches:
                track = self.tracks[track_idx]
                track['bbox'] = detections[det_idx]['bbox']
                track['confidence'] = detections[det_idx]['confidence']
                track['age'] = 0
                track['hits'] += 1
                track['history'].append({
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'bbox': detections[det_idx]['bbox']
                })
                
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self._create_track(detections[det_idx], frame_idx, timestamp)
                
            # Remove old tracks
            unmatched_track_indices = set(unmatched_tracks)
        else:
            # No detections or no tracks
            if len(detections) > 0:
                for det in detections:
                    self._create_track(det, frame_idx, timestamp)
                    
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
        
        # Return confirmed tracks
        active_tracks = [
            {
                'track_id': t['track_id'],
                'bbox': t['bbox'],
                'confidence': t['confidence'],
                'frame_idx': frame_idx,
                'timestamp': timestamp
            }
            for t in self.tracks if t['hits'] >= self.min_hits
        ]
        
        return active_tracks
        
    def _create_track(self, detection: Dict[str, Any], frame_idx: int, timestamp: float):
        """Create new track"""
        track = {
            'track_id': self.next_id,
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'age': 0,
            'hits': 1,
            'history': [{
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'bbox': detection['bbox']
            }]
        }
        self.tracks.append(track)
        self.next_id += 1
        
    def _match(self, detections: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        """Match detections to tracks using IoU"""
        from scipy.optimize import linear_sum_assignment
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                iou_matrix[i, j] = self._iou(det['bbox'], track['bbox'])
                
        # Hungarian matching
        det_indices, track_indices = linear_sum_assignment(-iou_matrix)
        
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            if iou_matrix[det_idx, track_idx] >= self.iou_threshold:
                matches.append((det_idx, track_idx))
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(track_idx)
                
        return matches, unmatched_dets, unmatched_tracks
        
    @staticmethod
    def _iou(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def get_all_tracks(self) -> List[Dict[str, Any]]:
        """Get all tracks including their full history"""
        return [{
            'track_id': t['track_id'],
            'hits': t['hits'],
            'history': t['history']
        } for t in self.tracks]


class FaceEmbedder:
    """Extract face embeddings using ArcFace or FaceNet"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config['face_recognition']
        self.logger = logger
        self.model_name = self.config['model']
        
        self._load_model()
        
    def _load_model(self):
        """Load face recognition model"""
        self.logger.info(f"Loading face embedder: {self.model_name}")
        
        try:
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            self.model_type = 'facenet'
            self.logger.info("Using FaceNet for embeddings")
        except ImportError:
            self.logger.warning("FaceNet not available, using simple features")
            self.model = None
            self.model_type = 'simple'
            
    def extract(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Extract face embedding from cropped face region
        
        Args:
            frame: Full frame image
            bbox: Face bounding box [x1, y1, x2, y2]
            
        Returns:
            Face embedding vector
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return np.zeros(self.config['embedding_dim'])
            
        if self.model_type == 'facenet':
            import torch
            from torchvision import transforms
            
            # Preprocess
            face_crop = cv2.resize(face_crop, (160, 160))
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            face_tensor = transform(face_crop).unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.model(face_tensor).squeeze().numpy()
                
            return embedding
        else:
            # Simple histogram-based features
            face_crop = cv2.resize(face_crop, (64, 64))
            hist = cv2.calcHist([face_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)
            
            # Pad to embedding_dim
            embedding = np.zeros(self.config['embedding_dim'])
            embedding[:len(hist)] = hist
            
            return embedding


class FaceClusterer:
    """Cluster face tracks into person identities"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self.cluster_config = config['face_clustering']
        self.min_track_length = self.cluster_config['min_track_length']
        
    def cluster(self, track_embeddings: Dict[int, List[np.ndarray]]) -> Dict[int, int]:
        """
        Cluster tracks into person IDs
        
        Args:
            track_embeddings: Dict mapping track_id to list of embeddings
            
        Returns:
            Dict mapping track_id to person_id
        """
        # Filter short tracks
        valid_tracks = {
            tid: embs for tid, embs in track_embeddings.items()
            if len(embs) >= self.min_track_length
        }
        
        if len(valid_tracks) == 0:
            self.logger.warning("No valid tracks for clustering")
            return {}
            
        # Compute mean embedding per track
        track_ids = list(valid_tracks.keys())
        mean_embeddings = np.array([
            np.mean(valid_tracks[tid], axis=0) for tid in track_ids
        ])
        
        # Normalize
        mean_embeddings = mean_embeddings / (np.linalg.norm(mean_embeddings, axis=1, keepdims=True) + 1e-7)
        
        # Cluster
        method = self.cluster_config['method']
        
        if method == 'agglomerative':
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.cluster_config['distance_threshold'],
                linkage=self.cluster_config['linkage'],
                metric='cosine'
            )
            labels = clustering.fit_predict(mean_embeddings)
        else:
            # DBSCAN
            clustering = DBSCAN(
                eps=self.cluster_config.get('eps', 0.5),
                min_samples=self.cluster_config.get('min_samples', 2),
                metric='cosine'
            )
            labels = clustering.fit_predict(mean_embeddings)
            
        # Map track_id to person_id
        track_to_person = {track_ids[i]: f"person_{labels[i]}" for i in range(len(track_ids))}
        
        n_persons = len(set(labels))
        self.logger.info(f"Clustered {len(track_ids)} tracks into {n_persons} persons")
        
        return track_to_person


def process_video(video_path: str, config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Main processing function: detect, track, embed, cluster faces
    
    Returns:
        Dictionary with persons and their track information
    """
    from extract_media import MediaExtractor
    
    extractor = MediaExtractor(config, logger)
    detector = FaceDetector(config, logger)
    tracker = ByteTrack(config, logger)
    embedder = FaceEmbedder(config, logger)
    clusterer = FaceClusterer(config, logger)
    
    logger.info("Starting face detection and tracking...")
    
    # Storage for embeddings per track
    track_embeddings = defaultdict(list)
    
    # Process frames
    from tqdm import tqdm
    
    total_frames = 0
    for frame_idx, timestamp, frame in tqdm(extractor.frame_generator(video_path)):
        # Detect faces
        detections = detector.detect(frame)
        
        # Update tracker
        tracks = tracker.update(detections, frame_idx, timestamp)
        
        # Extract embeddings for each track
        for track in tracks:
            embedding = embedder.extract(frame, track['bbox'])
            track_embeddings[track['track_id']].append(embedding)
            
        total_frames += 1
        
    logger.info(f"Processed {total_frames} frames, found {len(track_embeddings)} tracks")
    
    # Get all track histories
    all_tracks = tracker.get_all_tracks()
    
    # Cluster tracks into persons
    track_to_person = clusterer.cluster(track_embeddings)
    
    # Build output structure
    persons = defaultdict(lambda: {
        'person_id': None,
        'track_ids': [],
        'segments': [],
        'embedding_mean': None
    })
    
    for track_id, person_id in track_to_person.items():
        # Find track history
        track_hist = next((t for t in all_tracks if t['track_id'] == track_id), None)
        
        if track_hist and len(track_hist['history']) > 0:
            persons[person_id]['person_id'] = person_id
            persons[person_id]['track_ids'].append(track_id)
            
            # Add segments
            history = track_hist['history']
            segment = {
                't0': history[0]['timestamp'],
                't1': history[-1]['timestamp'],
                'frames': len(history)
            }
            persons[person_id]['segments'].append(segment)
            
            # Compute mean embedding
            if track_id in track_embeddings:
                emb_mean = np.mean(track_embeddings[track_id], axis=0).tolist()
                persons[person_id]['embedding_mean'] = emb_mean
                
    result = {
        'persons': list(persons.values()),
        'total_tracks': len(all_tracks),
        'total_persons': len(persons)
    }
    
    return result


def main():
    """Run face detection, tracking, and clustering"""
    import sys
    from utils import load_config, setup_logging, save_json
    
    config = load_config()
    logger = setup_logging(config)
    
    if len(sys.argv) < 2:
        logger.error("Usage: python faces_track_cluster.py <video_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    output_path = os.path.join(config['paths']['output_dir'], 'tracks_faces.json')
    
    result = process_video(video_path, config, logger)
    
    save_json(result, output_path)
    logger.info(f"Saved face tracking results to {output_path}")
    logger.info(f"Found {result['total_persons']} unique persons")


if __name__ == '__main__':
    main()
