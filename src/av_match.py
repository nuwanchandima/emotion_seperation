"""
Audio-Visual matching - match persons (faces) to speakers using Hungarian algorithm
Handles overlapping speech and off-screen speakers
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


class AVMatcher:
    """Match visual persons to audio speakers"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.av_config = config['av_matching']
        self.logger = logger or logging.getLogger(__name__)
        
        self.time_window = self.av_config['time_window']
        self.overlap_threshold = self.av_config['overlap_threshold']
        self.confidence_threshold = self.av_config['confidence_threshold']
        
    def compute_temporal_overlap(
        self,
        person_segments: List[Dict[str, float]],
        speaker_segments: List[Dict[str, float]]
    ) -> float:
        """
        Compute total temporal overlap between person and speaker segments
        
        Returns:
            Overlap ratio (0-1)
        """
        total_overlap = 0.0
        total_speaker_time = sum(s['end'] - s['start'] for s in speaker_segments)
        
        for pseg in person_segments:
            for sseg in speaker_segments:
                start = max(pseg['t0'], sseg['start'])
                end = min(pseg['t1'], sseg['end'])
                
                if start < end:
                    total_overlap += (end - start)
                    
        if total_speaker_time > 0:
            return total_overlap / total_speaker_time
        return 0.0
        
    def build_cost_matrix(
        self,
        persons_data: Dict[str, Any],
        diarization_data: Dict[str, Any],
        asd_scores: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Build cost matrix for person-speaker matching
        
        Returns:
            (cost_matrix, person_ids, speaker_ids)
        """
        persons = persons_data['persons']
        speakers = list(diarization_data['speakers'].keys())
        
        n_persons = len(persons)
        n_speakers = len(speakers)
        
        # Initialize cost matrix (higher is better, will negate for Hungarian)
        cost_matrix = np.zeros((n_persons, n_speakers))
        
        person_ids = [p['person_id'] for p in persons]
        
        self.logger.info(f"Building cost matrix: {n_persons} persons × {n_speakers} speakers")
        
        for i, person in enumerate(persons):
            person_id = person['person_id']
            person_segments = person['segments']
            
            # Get ASD scores for this person
            person_asd = asd_scores['person_scores'].get(person_id, [])
            
            for j, speaker in enumerate(speakers):
                speaker_segments = diarization_data['speakers'][speaker]
                
                # Component 1: Temporal overlap
                temporal_overlap = self.compute_temporal_overlap(
                    person_segments, 
                    speaker_segments
                )
                
                # Component 2: ASD sync score
                avg_sync_score = 0.0
                if person_asd:
                    # Match ASD segments with speaker segments
                    matching_scores = []
                    
                    for asd_seg in person_asd:
                        for spk_seg in speaker_segments:
                            # Check overlap
                            start = max(asd_seg['start'], spk_seg['start'])
                            end = min(asd_seg['end'], spk_seg['end'])
                            
                            if start < end:
                                matching_scores.append(asd_seg['sync_score'])
                                
                    if matching_scores:
                        avg_sync_score = np.mean(matching_scores)
                        
                # Combined score (weighted)
                # Temporal overlap: 40%, ASD sync: 60%
                combined_score = 0.4 * temporal_overlap + 0.6 * avg_sync_score
                
                cost_matrix[i, j] = combined_score
                
        return cost_matrix, person_ids, speakers
        
    def match(
        self,
        persons_data: Dict[str, Any],
        diarization_data: Dict[str, Any],
        asd_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform person-speaker matching
        
        Returns:
            Dictionary with matches and confidence scores
        """
        self.logger.info("Performing audio-visual matching...")
        
        # Build cost matrix
        cost_matrix, person_ids, speaker_ids = self.build_cost_matrix(
            persons_data, diarization_data, asd_scores
        )
        
        # Hungarian algorithm (maximize, so negate costs)
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        
        # Build matches
        matches = []
        matched_speakers = set()
        
        for i, j in zip(row_ind, col_ind):
            confidence = cost_matrix[i, j]
            
            if confidence >= self.confidence_threshold:
                match = {
                    'person_id': person_ids[i],
                    'speaker_id': speaker_ids[j],
                    'confidence': float(confidence),
                    'notes': 'on-screen; strong sync'
                }
                matches.append(match)
                matched_speakers.add(speaker_ids[j])
                
                self.logger.info(
                    f"Matched {person_ids[i]} ↔ {speaker_ids[j]} "
                    f"(confidence: {confidence:.3f})"
                )
            else:
                self.logger.info(
                    f"Low confidence match {person_ids[i]} ↔ {speaker_ids[j]} "
                    f"({confidence:.3f}) - rejected"
                )
                
        # Find off-screen speakers
        for speaker in speaker_ids:
            if speaker not in matched_speakers:
                # Check if speaker has significant speaking time
                speaker_time = sum(
                    s['duration'] 
                    for s in diarization_data['speakers'][speaker]
                )
                
                if speaker_time > 1.0:  # At least 1 second
                    # Find best temporal match even if below threshold
                    best_overlap = 0.0
                    for i, person_id in enumerate(person_ids):
                        if cost_matrix[i, speaker_ids.index(speaker)] > best_overlap:
                            best_overlap = cost_matrix[i, speaker_ids.index(speaker)]
                            
                    match = {
                        'person_id': None,
                        'speaker_id': speaker,
                        'confidence': float(best_overlap),
                        'notes': 'off-screen speaker'
                    }
                    matches.append(match)
                    
                    self.logger.info(
                        f"Off-screen speaker: {speaker} "
                        f"({speaker_time:.1f}s speech)"
                    )
                    
        result = {
            'av_links': matches,
            'total_matches': len([m for m in matches if m['person_id'] is not None]),
            'off_screen_speakers': len([m for m in matches if m['person_id'] is None]),
            'cost_matrix': cost_matrix.tolist(),
            'person_ids': person_ids,
            'speaker_ids': speaker_ids
        }
        
        return result


def main():
    """Run audio-visual matching"""
    import sys
    from utils import load_config, setup_logging, save_json, load_json
    
    config = load_config()
    logger = setup_logging(config)
    
    # Load required inputs
    output_dir = config['paths']['output_dir']
    
    faces_path = os.path.join(output_dir, 'tracks_faces.json')
    diarization_path = os.path.join(output_dir, 'diarization.json')
    asd_path = os.path.join(output_dir, 'active_speaker.json')
    
    # Check files exist
    required_files = {
        'Face tracking': faces_path,
        'Diarization': diarization_path,
        'Active speaker': asd_path
    }
    
    for name, path in required_files.items():
        if not os.path.exists(path):
            logger.error(f"{name} results not found: {path}")
            logger.error("Run previous pipeline steps first")
            sys.exit(1)
            
    # Load data
    persons_data = load_json(faces_path)
    diarization_data = load_json(diarization_path)
    asd_scores = load_json(asd_path)
    
    # Run matching
    matcher = AVMatcher(config, logger)
    result = matcher.match(persons_data, diarization_data, asd_scores)
    
    # Save results
    output_path = os.path.join(output_dir, 'av_map.json')
    save_json(result, output_path)
    
    logger.info(f"A/V matching complete: {output_path}")
    logger.info(f"Total matches: {result['total_matches']}")
    logger.info(f"Off-screen speakers: {result['off_screen_speakers']}")


if __name__ == '__main__':
    main()
