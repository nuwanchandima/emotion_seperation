"""
Main pipeline orchestrator - runs all components in sequence
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_config, setup_logging, save_json, ProgressTracker
from extract_media import MediaExtractor
from faces_track_cluster import process_video as process_faces
from diarize import SpeakerDiarizer
from active_speaker import ActiveSpeakerDetector
from av_match import AVMatcher
from emotion_change import process_all_speakers
from export_clips import ClipExporter


def run_pipeline(video_path: str, config_path: str = 'config.yaml'):
    """
    Run complete A/V emotion detection pipeline
    
    Args:
        video_path: Path to input video file
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    logger = setup_logging(config)
    
    logger.info("=" * 80)
    logger.info("A/V EMOTION DETECTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input video: {video_path}")
    
    # Initialize progress tracker
    tracker = ProgressTracker(total_stages=7, logger=logger)
    
    # Paths
    data_dir = config['paths']['data_dir']
    output_dir = config['paths']['output_dir']
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # =================================================================
    # STAGE 1: Extract media
    # =================================================================
    tracker.next_stage("Extracting audio and validating video")
    
    extractor = MediaExtractor(config, logger)
    
    # Validate input
    video_info = extractor.validate_input(video_path)
    logger.info(f"Video duration: {video_info['duration']:.1f}s")
    
    # Extract audio
    audio_path = os.path.join(data_dir, 'audio.wav')
    extractor.extract_audio(video_path, audio_path)
    
    # =================================================================
    # STAGE 2: Face detection, tracking, clustering
    # =================================================================
    tracker.next_stage("Detecting and tracking faces")
    
    faces_output = os.path.join(output_dir, 'tracks_faces.json')
    faces_result = process_faces(video_path, config, logger)
    save_json(faces_result, faces_output)
    
    logger.info(f"Found {faces_result['total_persons']} unique persons")
    
    # =================================================================
    # STAGE 3: Speaker diarization
    # =================================================================
    tracker.next_stage("Running speaker diarization")
    
    diarizer = SpeakerDiarizer(config, logger)
    diarization_result = diarizer.diarize(audio_path)
    
    diarization_json = os.path.join(output_dir, 'diarization.json')
    diarization_rttm = os.path.join(output_dir, 'diarization.rttm')
    
    save_json(diarization_result, diarization_json)
    diarizer.export_rttm(diarization_result, diarization_rttm)
    
    logger.info(f"Found {diarization_result['num_speakers']} speakers")
    
    # =================================================================
    # STAGE 4: Active speaker detection
    # =================================================================
    tracker.next_stage("Computing active speaker scores")
    
    asd = ActiveSpeakerDetector(config, logger)
    asd_result = asd.score_person_segments(
        video_path, audio_path, faces_result
    )
    
    asd_output = os.path.join(output_dir, 'active_speaker.json')
    save_json(asd_result, asd_output)
    
    # =================================================================
    # STAGE 5: Audio-visual matching
    # =================================================================
    tracker.next_stage("Matching persons to speakers")
    
    matcher = AVMatcher(config, logger)
    av_result = matcher.match(faces_result, diarization_result, asd_result)
    
    av_output = os.path.join(output_dir, 'av_map.json')
    save_json(av_result, av_output)
    
    logger.info(
        f"Matched {av_result['total_matches']} persons to speakers, "
        f"{av_result['off_screen_speakers']} off-screen"
    )
    
    # =================================================================
    # STAGE 6: Emotion change detection
    # =================================================================
    tracker.next_stage("Detecting emotion changes")
    
    emotion_result = process_all_speakers(
        audio_path, diarization_result, av_result, config, logger
    )
    
    emotion_output = os.path.join(output_dir, 'emotion_changes.json')
    save_json(emotion_result, emotion_output)
    
    logger.info(f"Detected {emotion_result['total_changes']} emotion changes")
    
    # =================================================================
    # STAGE 7: Export clips
    # =================================================================
    tracker.next_stage("Exporting emotion change clips")
    
    exporter = ClipExporter(config, logger)
    clips_result = exporter.export_all_clips(
        video_path, emotion_result, av_result
    )
    
    clips_summary = os.path.join(output_dir, 'clips_summary.json')
    clips_manifest = os.path.join(output_dir, 'clips_manifest.md')
    
    save_json(clips_result, clips_summary)
    exporter.create_clip_manifest(clips_result, clips_manifest)
    
    logger.info(f"Exported {clips_result['total_clips']} clips")
    
    # =================================================================
    # Complete!
    # =================================================================
    tracker.complete()
    
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("")
    logger.info("Key results:")
    logger.info(f"  - Persons detected: {faces_result['total_persons']}")
    logger.info(f"  - Speakers detected: {diarization_result['num_speakers']}")
    logger.info(f"  - A/V matches: {av_result['total_matches']}")
    logger.info(f"  - Emotion changes: {emotion_result['total_changes']}")
    logger.info(f"  - Clips exported: {clips_result['total_clips']}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  - Face tracks: {faces_output}")
    logger.info(f"  - Diarization: {diarization_json}")
    logger.info(f"  - A/V mapping: {av_output}")
    logger.info(f"  - Emotion changes: {emotion_output}")
    logger.info(f"  - Clips summary: {clips_summary}")
    logger.info(f"  - Clips manifest: {clips_manifest}")
    logger.info("=" * 80)
    
    return {
        'faces': faces_result,
        'diarization': diarization_result,
        'av_mapping': av_result,
        'emotions': emotion_result,
        'clips': clips_result
    }


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="A/V Emotion Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python pipeline.py video.mp4
  
  # Use custom config
  python pipeline.py video.mp4 --config my_config.yaml
  
  # Specify output directory
  python pipeline.py video.mp4 --output results/
        """
    )
    
    parser.add_argument(
        'video',
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output',
        help='Override output directory from config'
    )
    
    args = parser.parse_args()
    
    # Check video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
        
    # Override output directory if specified
    if args.output:
        config = load_config(args.config)
        config['paths']['output_dir'] = args.output
        
        # Save temporary config
        import tempfile
        import yaml
        
        temp_config = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.yaml', 
            delete=False
        )
        yaml.dump(config, temp_config)
        temp_config.close()
        
        config_path = temp_config.name
    else:
        config_path = args.config
        
    try:
        # Run pipeline
        results = run_pipeline(args.video, config_path)
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Clean up temp config if created
        if args.output:
            try:
                os.unlink(config_path)
            except:
                pass


if __name__ == '__main__':
    main()
