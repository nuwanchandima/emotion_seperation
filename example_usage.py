"""
Example usage of the A/V Emotion Detection Pipeline
Demonstrates how to use the pipeline programmatically
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import run_pipeline
from utils import load_json


def example_basic_usage():
    """Example 1: Basic pipeline usage"""
    print("=" * 60)
    print("Example 1: Basic Pipeline Usage")
    print("=" * 60)
    
    video_path = "path/to/your/video.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"⚠ Video not found: {video_path}")
        print("Update the video_path variable with your video file")
        return
    
    # Run pipeline with default config
    results = run_pipeline(video_path)
    
    # Access results
    print("\nResults:")
    print(f"  Persons detected: {results['faces']['total_persons']}")
    print(f"  Speakers detected: {results['diarization']['num_speakers']}")
    print(f"  Emotion changes: {results['emotions']['total_changes']}")
    print(f"  Clips exported: {results['clips']['total_clips']}")


def example_custom_config():
    """Example 2: Using custom configuration"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    import yaml
    from utils import load_config
    
    # Load default config
    config = load_config('config.yaml')
    
    # Customize settings
    config['video']['target_fps'] = 5  # Process fewer frames
    config['clips']['padding_before'] = 1.0  # Longer clips
    config['clips']['padding_after'] = 1.0
    config['change_detection']['penalty'] = 5  # More sensitive
    
    # Save temporary config
    temp_config_path = 'outputs/temp_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run with custom config
    video_path = "your_video.mp4"
    
    if os.path.exists(video_path):
        results = run_pipeline(video_path, temp_config_path)
        print("Custom pipeline complete!")
    else:
        print(f"⚠ Video not found: {video_path}")


def example_analyze_results():
    """Example 3: Analyzing pipeline outputs"""
    print("\n" + "=" * 60)
    print("Example 3: Analyzing Results")
    print("=" * 60)
    
    output_dir = "outputs"
    
    # Load all results
    try:
        faces = load_json(os.path.join(output_dir, 'tracks_faces.json'))
        diarization = load_json(os.path.join(output_dir, 'diarization.json'))
        av_map = load_json(os.path.join(output_dir, 'av_map.json'))
        emotions = load_json(os.path.join(output_dir, 'emotion_changes.json'))
        
    except FileNotFoundError as e:
        print(f"⚠ Results not found: {e}")
        print("Run the pipeline first!")
        return
    
    # Analyze persons
    print("\n👤 Person Analysis:")
    for person in faces['persons']:
        person_id = person['person_id']
        total_time = sum(seg['t1'] - seg['t0'] for seg in person['segments'])
        print(f"  {person_id}: {len(person['segments'])} segments, {total_time:.1f}s on screen")
    
    # Analyze speakers
    print("\n🔊 Speaker Analysis:")
    for speaker, segments in diarization['speakers'].items():
        total_time = sum(seg['duration'] for seg in segments)
        print(f"  {speaker}: {len(segments)} utterances, {total_time:.1f}s speaking")
    
    # Analyze mappings
    print("\n🔗 Person ↔ Speaker Mappings:")
    for link in av_map['av_links']:
        person = link['person_id'] or 'off-screen'
        speaker = link['speaker_id']
        confidence = link['confidence']
        print(f"  {person} ↔ {speaker} (confidence: {confidence:.2f})")
    
    # Analyze emotions
    print("\n😊 Emotion Changes:")
    for speaker, changes in emotions['emotion_changes'].items():
        print(f"  {speaker}: {len(changes)} emotion shifts")
        
        for change in changes[:3]:  # Show first 3
            t = change['t']
            from_label = change['from']['label']
            to_label = change['to']['label']
            print(f"    • t={t:.1f}s: {from_label} → {to_label}")
            
        if len(changes) > 3:
            print(f"    ... and {len(changes) - 3} more")


def example_batch_processing():
    """Example 4: Batch process multiple videos"""
    print("\n" + "=" * 60)
    print("Example 4: Batch Processing")
    print("=" * 60)
    
    videos = [
        "video1.mp4",
        "video2.mp4",
        "video3.mp4"
    ]
    
    results_summary = []
    
    for video in videos:
        if not os.path.exists(video):
            print(f"⚠ Skipping {video} (not found)")
            continue
            
        print(f"\n📹 Processing {video}...")
        
        try:
            # Custom output directory per video
            video_name = Path(video).stem
            output_dir = f"outputs/{video_name}"
            
            # Create custom config
            from utils import load_config
            import yaml
            
            config = load_config('config.yaml')
            config['paths']['output_dir'] = output_dir
            
            temp_config = f"outputs/{video_name}_config.yaml"
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)
            
            # Run pipeline
            results = run_pipeline(video, temp_config)
            
            # Store summary
            results_summary.append({
                'video': video,
                'status': 'success',
                'persons': results['faces']['total_persons'],
                'speakers': results['diarization']['num_speakers'],
                'changes': results['emotions']['total_changes'],
                'clips': results['clips']['total_clips']
            })
            
            print(f"✓ {video} complete")
            
        except Exception as e:
            print(f"✗ {video} failed: {e}")
            results_summary.append({
                'video': video,
                'status': 'failed',
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    
    for result in results_summary:
        if result['status'] == 'success':
            print(f"✓ {result['video']}")
            print(f"    Persons: {result['persons']}, "
                  f"Speakers: {result['speakers']}, "
                  f"Changes: {result['changes']}")
        else:
            print(f"✗ {result['video']}: {result['error']}")


def example_filter_clips():
    """Example 5: Filter clips by emotion type"""
    print("\n" + "=" * 60)
    print("Example 5: Filter Clips by Emotion")
    print("=" * 60)
    
    try:
        clips_summary = load_json('outputs/clips_summary.json')
    except FileNotFoundError:
        print("⚠ Clips summary not found. Run pipeline first!")
        return
    
    # Filter for specific emotion transitions
    target_transitions = [
        ('neutral', 'happy'),
        ('sad', 'happy'),
        ('angry', 'neutral')
    ]
    
    print("\nFiltering clips for transitions:")
    for from_e, to_e in target_transitions:
        print(f"  • {from_e} → {to_e}")
    
    print("\nMatching clips:")
    
    matched = 0
    for clip in clips_summary['clips']:
        from_label = clip['from_emotion'].get('label', '')
        to_label = clip['to_emotion'].get('label', '')
        
        if (from_label, to_label) in target_transitions:
            matched += 1
            print(f"  {matched}. {os.path.basename(clip['clip_path'])}")
            print(f"     Time: {clip['change_time']:.1f}s, "
                  f"Speaker: {clip['speaker_id']}")
    
    if matched == 0:
        print("  (No matching clips found)")
    else:
        print(f"\nFound {matched} matching clips")


def example_export_timeline():
    """Example 6: Export emotion timeline for visualization"""
    print("\n" + "=" * 60)
    print("Example 6: Export Timeline Data")
    print("=" * 60)
    
    try:
        emotions = load_json('outputs/emotion_changes.json')
        av_map = load_json('outputs/av_map.json')
    except FileNotFoundError:
        print("⚠ Results not found. Run pipeline first!")
        return
    
    # Build timeline for visualization
    timeline = []
    
    # Map speakers to persons
    speaker_to_person = {}
    for link in av_map['av_links']:
        if link['person_id']:
            speaker_to_person[link['speaker_id']] = link['person_id']
    
    # Extract all emotion changes
    for speaker, changes in emotions['emotion_changes'].items():
        person = speaker_to_person.get(speaker, speaker)
        
        for change in changes:
            timeline.append({
                'time': change['t'],
                'person': person,
                'speaker': speaker,
                'from_emotion': change['from']['label'],
                'to_emotion': change['to']['label'],
                'from_valence': change['from']['valence'],
                'from_arousal': change['from']['arousal'],
                'to_valence': change['to']['valence'],
                'to_arousal': change['to']['arousal']
            })
    
    # Sort by time
    timeline.sort(key=lambda x: x['time'])
    
    # Export as CSV for easy visualization
    import csv
    
    output_path = 'outputs/emotion_timeline.csv'
    
    with open(output_path, 'w', newline='') as f:
        if timeline:
            writer = csv.DictWriter(f, fieldnames=timeline[0].keys())
            writer.writeheader()
            writer.writerows(timeline)
    
    print(f"✓ Timeline exported to: {output_path}")
    print(f"  Total events: {len(timeline)}")
    print("\nSample events:")
    
    for event in timeline[:5]:
        print(f"  t={event['time']:.1f}s: {event['person']} "
              f"{event['from_emotion']} → {event['to_emotion']}")


def main():
    """Run all examples"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║      A/V Emotion Detection Pipeline - Examples           ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

These examples demonstrate different ways to use the pipeline.
Update the video paths in each example function before running.
    """)
    
    # Uncomment the examples you want to run:
    
    # example_basic_usage()
    # example_custom_config()
    example_analyze_results()  # This one works if you have existing results
    # example_batch_processing()
    example_filter_clips()
    example_export_timeline()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nTo run these examples:")
    print("  1. Update video paths in each function")
    print("  2. Uncomment the examples you want to run")
    print("  3. Run: python example_usage.py")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
