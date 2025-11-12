"""
Export video clips around emotion change points
Generates ffmpeg commands and clips the video
"""

import os
import logging
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path


class ClipExporter:
    """Export video clips around emotion changes"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.clip_config = config['clips']
        self.logger = logger or logging.getLogger(__name__)
        
        self.padding_before = self.clip_config['padding_before']
        self.padding_after = self.clip_config['padding_after']
        self.clip_format = self.clip_config['format']
        self.codec = self.clip_config['codec']
        self.include_labels = self.clip_config['include_emotion_label']
        
        self.clips_dir = config['paths']['clips_dir']
        os.makedirs(self.clips_dir, exist_ok=True)
        
    def generate_clip_name(
        self,
        speaker_id: str,
        change_idx: int,
        change_info: Dict[str, Any]
    ) -> str:
        """Generate descriptive clip filename"""
        if self.include_labels and 'from' in change_info and 'to' in change_info:
            from_label = change_info['from']['label']
            to_label = change_info['to']['label']
            timestamp = change_info['t']
            
            filename = (
                f"{speaker_id}_change_{change_idx:03d}_"
                f"{from_label}_to_{to_label}_"
                f"t{timestamp:.1f}s.{self.clip_format}"
            )
        else:
            timestamp = change_info['t']
            filename = f"{speaker_id}_change_{change_idx:03d}_t{timestamp:.1f}s.{self.clip_format}"
            
        return filename
        
    def extract_clip(
        self,
        video_path: str,
        start_time: float,
        duration: float,
        output_path: str,
        fast: bool = True
    ) -> bool:
        """
        Extract clip using ffmpeg
        
        Args:
            video_path: Input video path
            start_time: Start time in seconds
            duration: Duration in seconds
            output_path: Output clip path
            fast: If True, use copy codec (fast but less precise)
            
        Returns:
            True if successful
        """
        # Build ffmpeg command
        if fast and self.codec == 'copy':
            cmd = [
                'ffmpeg',
                '-y',  # overwrite
                '-ss', f'{start_time:.2f}',
                '-i', video_path,
                '-t', f'{duration:.2f}',
                '-c', 'copy',
                output_path
            ]
        else:
            # Re-encode (slower but more precise)
            cmd = [
                'ffmpeg',
                '-y',
                '-ss', f'{start_time:.2f}',
                '-i', video_path,
                '-t', f'{duration:.2f}',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'fast',
                output_path
            ]
            
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to extract clip: {e.stderr}")
            return False
            
    def export_all_clips(
        self,
        video_path: str,
        emotion_changes: Dict[str, Any],
        av_map: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export all emotion change clips
        
        Args:
            video_path: Path to source video
            emotion_changes: Emotion change detection results
            av_map: Optional A/V mapping to get person IDs
            
        Returns:
            Summary of exported clips
        """
        self.logger.info(f"Exporting clips to {self.clips_dir}")
        
        exported_clips = []
        failed_clips = []
        
        # Get speaker to person mapping if available
        speaker_to_person = {}
        if av_map:
            for link in av_map.get('av_links', []):
                if link['person_id']:
                    speaker_to_person[link['speaker_id']] = link['person_id']
                    
        # Process each speaker
        for speaker_id, changes in emotion_changes['emotion_changes'].items():
            # Use person ID if available
            output_id = speaker_to_person.get(speaker_id, speaker_id)
            
            self.logger.info(f"Processing {len(changes)} changes for {output_id}")
            
            for idx, change_info in enumerate(changes):
                change_time = change_info['t']
                
                # Calculate clip boundaries
                start_time = max(0, change_time - self.padding_before)
                duration = self.padding_before + self.padding_after
                
                # Generate output filename
                clip_name = self.generate_clip_name(output_id, idx, change_info)
                output_path = os.path.join(self.clips_dir, clip_name)
                
                # Extract clip
                success = self.extract_clip(
                    video_path, start_time, duration, output_path
                )
                
                if success:
                    clip_info = {
                        'speaker_id': speaker_id,
                        'person_id': speaker_to_person.get(speaker_id),
                        'change_index': idx,
                        'change_time': change_time,
                        'clip_path': output_path,
                        'start_time': start_time,
                        'duration': duration,
                        'from_emotion': change_info.get('from', {}),
                        'to_emotion': change_info.get('to', {})
                    }
                    exported_clips.append(clip_info)
                else:
                    failed_clips.append({
                        'speaker_id': speaker_id,
                        'change_index': idx,
                        'change_time': change_time
                    })
                    
        summary = {
            'total_clips': len(exported_clips),
            'failed_clips': len(failed_clips),
            'clips': exported_clips,
            'failures': failed_clips,
            'output_directory': self.clips_dir
        }
        
        self.logger.info(
            f"Exported {len(exported_clips)} clips "
            f"({len(failed_clips)} failed)"
        )
        
        return summary
        
    def create_clip_manifest(
        self,
        clips_summary: Dict[str, Any],
        output_path: str
    ):
        """Create a markdown manifest of all clips"""
        lines = [
            "# Emotion Change Clips\n",
            f"Total clips: {clips_summary['total_clips']}\n\n",
            "## Clips by Speaker/Person\n\n"
        ]
        
        # Group by speaker
        by_speaker = {}
        for clip in clips_summary['clips']:
            speaker = clip['person_id'] or clip['speaker_id']
            if speaker not in by_speaker:
                by_speaker[speaker] = []
            by_speaker[speaker].append(clip)
            
        for speaker, clips in sorted(by_speaker.items()):
            lines.append(f"### {speaker}\n\n")
            
            for clip in clips:
                from_label = clip['from_emotion'].get('label', 'unknown')
                to_label = clip['to_emotion'].get('label', 'unknown')
                
                lines.append(
                    f"- **Clip {clip['change_index']}** "
                    f"(t={clip['change_time']:.1f}s): "
                    f"{from_label} → {to_label}\n"
                )
                lines.append(f"  - File: `{os.path.basename(clip['clip_path'])}`\n")
                lines.append(f"  - Duration: {clip['duration']:.1f}s\n")
                
                if 'valence' in clip['from_emotion']:
                    v_from = clip['from_emotion']['valence']
                    a_from = clip['from_emotion']['arousal']
                    v_to = clip['to_emotion']['valence']
                    a_to = clip['to_emotion']['arousal']
                    
                    lines.append(
                        f"  - Emotion shift: "
                        f"(V:{v_from:.2f}, A:{a_from:.2f}) → "
                        f"(V:{v_to:.2f}, A:{a_to:.2f})\n"
                    )
                    
                lines.append("\n")
                
        # Write manifest
        with open(output_path, 'w') as f:
            f.writelines(lines)
            
        self.logger.info(f"Created clip manifest: {output_path}")


def main():
    """Export emotion change clips"""
    import sys
    from utils import load_config, setup_logging, save_json, load_json
    
    config = load_config()
    logger = setup_logging(config)
    
    if len(sys.argv) < 2:
        logger.error("Usage: python export_clips.py <video_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    # Load required data
    output_dir = config['paths']['output_dir']
    emotion_changes_path = os.path.join(output_dir, 'emotion_changes.json')
    av_map_path = os.path.join(output_dir, 'av_map.json')
    
    if not os.path.exists(emotion_changes_path):
        logger.error("Emotion changes not found. Run emotion_change.py first")
        sys.exit(1)
        
    emotion_changes = load_json(emotion_changes_path)
    
    # Load A/V map if available
    av_map = None
    if os.path.exists(av_map_path):
        av_map = load_json(av_map_path)
        
    # Export clips
    exporter = ClipExporter(config, logger)
    clips_summary = exporter.export_all_clips(video_path, emotion_changes, av_map)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'clips_summary.json')
    save_json(clips_summary, summary_path)
    
    # Create manifest
    manifest_path = os.path.join(output_dir, 'clips_manifest.md')
    exporter.create_clip_manifest(clips_summary, manifest_path)
    
    logger.info(f"Clip export complete!")
    logger.info(f"Summary: {summary_path}")
    logger.info(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
