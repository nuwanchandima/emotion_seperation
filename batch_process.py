"""
Batch processing script for multiple videos
Processes all videos in a directory and generates a summary report
"""

import os
import sys
import argparse
from pathlib import Path
import json
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import run_pipeline
from utils import load_config, setup_logging


def find_videos(directory, extensions=None):
    """Find all video files in directory"""
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    video_files = []
    for ext in extensions:
        video_files.extend(Path(directory).glob(f'**/*{ext}'))
    
    return sorted(video_files)


def process_batch(
    video_dir,
    output_base_dir='batch_outputs',
    config_path='config.yaml',
    resume=True,
    max_videos=None
):
    """
    Process all videos in a directory
    
    Args:
        video_dir: Directory containing videos
        output_base_dir: Base directory for all outputs
        config_path: Path to configuration file
        resume: Skip already processed videos
        max_videos: Maximum number of videos to process (None = all)
    """
    config = load_config(config_path)
    logger = setup_logging(config)
    
    logger.info("=" * 80)
    logger.info("BATCH PROCESSING MODE")
    logger.info("=" * 80)
    logger.info(f"Video directory: {video_dir}")
    logger.info(f"Output directory: {output_base_dir}")
    
    # Find all videos
    videos = find_videos(video_dir)
    
    if max_videos:
        videos = videos[:max_videos]
    
    logger.info(f"Found {len(videos)} videos to process")
    
    if len(videos) == 0:
        logger.error("No videos found!")
        return
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Track results
    results = []
    start_time = time.time()
    
    for idx, video_path in enumerate(videos, 1):
        video_name = video_path.stem
        logger.info(f"\n{'=' * 80}")
        logger.info(f"[{idx}/{len(videos)}] Processing: {video_name}")
        logger.info(f"{'=' * 80}")
        
        # Create output directory for this video
        video_output_dir = os.path.join(output_base_dir, video_name)
        
        # Check if already processed
        if resume and os.path.exists(os.path.join(video_output_dir, 'clips_summary.json')):
            logger.info(f"⏭️  Skipping {video_name} (already processed)")
            
            # Load existing results
            try:
                with open(os.path.join(video_output_dir, 'clips_summary.json')) as f:
                    summary = json.load(f)
                    
                results.append({
                    'video': str(video_path),
                    'video_name': video_name,
                    'status': 'skipped',
                    'message': 'Already processed'
                })
            except:
                pass
                
            continue
        
        # Process video
        video_start = time.time()
        
        try:
            # Update config with video-specific output directory
            import yaml
            
            video_config = config.copy()
            video_config['paths']['output_dir'] = video_output_dir
            
            # Save temporary config
            temp_config_path = os.path.join(video_output_dir, '_temp_config.yaml')
            os.makedirs(video_output_dir, exist_ok=True)
            
            with open(temp_config_path, 'w') as f:
                yaml.dump(video_config, f)
            
            # Run pipeline
            pipeline_results = run_pipeline(str(video_path), temp_config_path)
            
            # Clean up temp config
            try:
                os.remove(temp_config_path)
            except:
                pass
            
            video_time = time.time() - video_start
            
            # Store results
            result = {
                'video': str(video_path),
                'video_name': video_name,
                'status': 'success',
                'processing_time': video_time,
                'persons': pipeline_results['faces']['total_persons'],
                'speakers': pipeline_results['diarization']['num_speakers'],
                'av_matches': pipeline_results['av_mapping']['total_matches'],
                'emotion_changes': pipeline_results['emotions']['total_changes'],
                'clips_generated': pipeline_results['clips']['total_clips'],
                'output_dir': video_output_dir
            }
            
            results.append(result)
            
            logger.info(f"✓ {video_name} complete in {video_time:.1f}s")
            logger.info(f"  Persons: {result['persons']}, "
                       f"Speakers: {result['speakers']}, "
                       f"Changes: {result['emotion_changes']}, "
                       f"Clips: {result['clips_generated']}")
            
        except KeyboardInterrupt:
            logger.warning("\n\nBatch processing interrupted by user")
            results.append({
                'video': str(video_path),
                'video_name': video_name,
                'status': 'interrupted',
                'message': 'User cancelled'
            })
            break
            
        except Exception as e:
            video_time = time.time() - video_start
            
            logger.error(f"✗ {video_name} failed: {e}")
            
            results.append({
                'video': str(video_path),
                'video_name': video_name,
                'status': 'failed',
                'processing_time': video_time,
                'error': str(e)
            })
    
    # Summary
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 80)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    skipped = [r for r in results if r['status'] == 'skipped']
    
    logger.info(f"Total videos: {len(videos)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Skipped: {len(skipped)}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    
    if successful:
        avg_time = sum(r['processing_time'] for r in successful) / len(successful)
        logger.info(f"Average processing time: {avg_time:.1f}s per video")
    
    # Generate detailed report
    report_path = os.path.join(output_base_dir, 'batch_report.json')
    report = {
        'batch_info': {
            'video_directory': video_dir,
            'output_directory': output_base_dir,
            'total_videos': len(videos),
            'processed': len(successful),
            'failed': len(failed),
            'skipped': len(skipped),
            'total_time_seconds': total_time,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.now().isoformat()
        },
        'results': results
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nDetailed report saved to: {report_path}")
    
    # Generate markdown report
    md_report_path = os.path.join(output_base_dir, 'batch_report.md')
    
    with open(md_report_path, 'w') as f:
        f.write("# Batch Processing Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Videos Directory:** `{video_dir}`\n\n")
        f.write(f"**Output Directory:** `{output_base_dir}`\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total videos: {len(videos)}\n")
        f.write(f"- Successful: {len(successful)}\n")
        f.write(f"- Failed: {len(failed)}\n")
        f.write(f"- Skipped: {len(skipped)}\n")
        f.write(f"- Total time: {total_time/60:.1f} minutes\n\n")
        
        if successful:
            f.write("## Successful Videos\n\n")
            f.write("| Video | Persons | Speakers | Changes | Clips | Time (s) |\n")
            f.write("|-------|---------|----------|---------|-------|----------|\n")
            
            for r in successful:
                f.write(f"| {r['video_name']} | "
                       f"{r['persons']} | "
                       f"{r['speakers']} | "
                       f"{r['emotion_changes']} | "
                       f"{r['clips_generated']} | "
                       f"{r['processing_time']:.1f} |\n")
            
            f.write("\n")
        
        if failed:
            f.write("## Failed Videos\n\n")
            f.write("| Video | Error |\n")
            f.write("|-------|-------|\n")
            
            for r in failed:
                error_msg = r.get('error', 'Unknown error')[:100]
                f.write(f"| {r['video_name']} | {error_msg} |\n")
            
            f.write("\n")
        
        if skipped:
            f.write("## Skipped Videos\n\n")
            for r in skipped:
                f.write(f"- {r['video_name']}\n")
            f.write("\n")
    
    logger.info(f"Markdown report saved to: {md_report_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Batch processing complete!")
    logger.info("=" * 80 + "\n")
    
    return results


def main():
    """Command-line interface for batch processing"""
    parser = argparse.ArgumentParser(
        description="Batch process multiple videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in a directory
  python batch_process.py videos/
  
  # Custom output directory
  python batch_process.py videos/ --output batch_results/
  
  # Resume from previous run (skip completed)
  python batch_process.py videos/ --resume
  
  # Process first 5 videos only
  python batch_process.py videos/ --max 5
  
  # Use custom config
  python batch_process.py videos/ --config fast_config.yaml
        """
    )
    
    parser.add_argument(
        'video_dir',
        help='Directory containing videos to process'
    )
    
    parser.add_argument(
        '--output',
        default='batch_outputs',
        help='Output directory for all results (default: batch_outputs)'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already processed videos'
    )
    
    parser.add_argument(
        '--max',
        type=int,
        help='Maximum number of videos to process'
    )
    
    args = parser.parse_args()
    
    # Check video directory exists
    if not os.path.isdir(args.video_dir):
        print(f"Error: Video directory not found: {args.video_dir}")
        sys.exit(1)
    
    # Check config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run batch processing
    try:
        results = process_batch(
            video_dir=args.video_dir,
            output_base_dir=args.output,
            config_path=args.config,
            resume=args.resume,
            max_videos=args.max
        )
        
        # Exit code based on results
        failed = sum(1 for r in results if r['status'] == 'failed')
        sys.exit(1 if failed > 0 else 0)
        
    except KeyboardInterrupt:
        print("\n\nBatch processing cancelled by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\nBatch processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
