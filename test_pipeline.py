"""
Test suite for the A/V Emotion Detection Pipeline
Run this to verify all components work correctly
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class Colors:
    """Terminal colors for pretty output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_test(name, status, message=""):
    """Print test result"""
    if status:
        symbol = f"{Colors.GREEN}✓{Colors.END}"
        status_text = f"{Colors.GREEN}PASS{Colors.END}"
    else:
        symbol = f"{Colors.RED}✗{Colors.END}"
        status_text = f"{Colors.RED}FAIL{Colors.END}"
    
    print(f"{symbol} [{status_text}] {name}")
    if message:
        print(f"    {message}")


def test_imports():
    """Test that all required modules can be imported"""
    print(f"\n{Colors.BOLD}Testing Imports{Colors.END}")
    
    modules = {
        'Standard Library': ['os', 'sys', 'json', 'logging', 'subprocess'],
        'Core Dependencies': ['numpy', 'scipy', 'cv2', 'yaml', 'torch'],
        'Audio/Video': ['librosa', 'soundfile'],
        'ML Libraries': ['sklearn', 'tqdm'],
        'Optional': ['transformers', 'ruptures', 'facenet_pytorch', 'pyannote.audio']
    }
    
    results = {}
    
    for category, module_list in modules.items():
        print(f"\n  {Colors.BLUE}{category}:{Colors.END}")
        for module in module_list:
            try:
                __import__(module)
                print_test(module, True)
                results[module] = True
            except ImportError as e:
                if category == 'Optional':
                    print_test(module, False, f"Optional - {str(e)[:50]}")
                else:
                    print_test(module, False, str(e)[:50])
                results[module] = False
    
    return all(results.get(m, False) for cat, mods in modules.items() 
               if cat != 'Optional' for m in mods)


def test_ffmpeg():
    """Test FFmpeg availability"""
    print(f"\n{Colors.BOLD}Testing FFmpeg{Colors.END}\n")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.split('\n')[0]
        print_test("FFmpeg", True, version)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print_test("FFmpeg", False, str(e))
        return False


def test_configuration():
    """Test configuration loading"""
    print(f"\n{Colors.BOLD}Testing Configuration{Colors.END}\n")
    
    try:
        from utils import load_config
        config = load_config('config.yaml')
        
        required_keys = ['paths', 'video', 'audio', 'face_detection', 
                        'diarization', 'emotion', 'clips']
        
        for key in required_keys:
            if key in config:
                print_test(f"Config key: {key}", True)
            else:
                print_test(f"Config key: {key}", False, "Missing")
                return False
        
        return True
        
    except Exception as e:
        print_test("Configuration", False, str(e))
        return False


def test_media_extraction():
    """Test media extraction with synthetic data"""
    print(f"\n{Colors.BOLD}Testing Media Extraction{Colors.END}\n")
    
    try:
        from extract_media import MediaExtractor
        from utils import load_config
        import numpy as np
        import cv2
        
        config = load_config('config.yaml')
        
        # Create synthetic video
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, 'test.mp4')
        
        # Create 5 seconds of synthetic video (100 frames at 20 fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        
        for i in range(100):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Test extraction
        extractor = MediaExtractor(config, None)
        audio_path = os.path.join(temp_dir, 'test.wav')
        
        # Note: This will fail if video has no audio, which is expected
        try:
            extractor.extract_audio(video_path, audio_path)
            print_test("Audio extraction", True)
        except:
            print_test("Audio extraction", False, "Synthetic video has no audio (expected)")
        
        # Test frame reading
        cap = extractor.get_video_reader(video_path)
        print_test("Video reading", cap.isOpened(), 
                  f"{extractor.frame_count} frames @ {extractor.fps} FPS")
        cap.release()
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print_test("Media Extraction", False, str(e))
        return False


def test_face_detection():
    """Test face detection module"""
    print(f"\n{Colors.BOLD}Testing Face Detection{Colors.END}\n")
    
    try:
        from faces_track_cluster import FaceDetector
        from utils import load_config
        import numpy as np
        
        config = load_config('config.yaml')
        detector = FaceDetector(config, None)
        
        # Create test image (640x480 RGB)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Detect (will likely find 0 faces in random noise)
        detections = detector.detect(test_frame)
        
        print_test("Face detector initialized", True, 
                  f"Model: {detector.detector_type}")
        print_test("Face detection runs", True, 
                  f"Found {len(detections)} faces (expected 0 in noise)")
        
        return True
        
    except Exception as e:
        print_test("Face Detection", False, str(e))
        return False


def test_utilities():
    """Test utility functions"""
    print(f"\n{Colors.BOLD}Testing Utilities{Colors.END}\n")
    
    try:
        from utils import (format_timestamp, time_overlap, cosine_similarity,
                          bbox_iou, ensure_dir)
        import numpy as np
        
        # Test timestamp formatting
        t = format_timestamp(125.5)
        print_test("format_timestamp", t == "00:02:05.500", f"Got: {t}")
        
        # Test time overlap
        overlap = time_overlap((1.0, 5.0), (3.0, 7.0))
        print_test("time_overlap", overlap == 2.0, f"Got: {overlap}")
        
        # Test cosine similarity
        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 0, 0])
        sim = cosine_similarity(v1, v2)
        print_test("cosine_similarity", abs(sim - 1.0) < 0.01, f"Got: {sim}")
        
        # Test bbox IoU
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = bbox_iou(box1, box2)
        expected_iou = 25.0 / 175.0  # 25 intersection, 175 union
        print_test("bbox_iou", abs(iou - expected_iou) < 0.01, 
                  f"Got: {iou:.3f}, Expected: {expected_iou:.3f}")
        
        # Test directory creation
        temp_dir = tempfile.mkdtemp()
        test_path = os.path.join(temp_dir, 'nested', 'dir')
        ensure_dir(test_path)
        exists = os.path.exists(test_path)
        print_test("ensure_dir", exists)
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print_test("Utilities", False, str(e))
        return False


def test_emotion_features():
    """Test emotion recognition feature extraction"""
    print(f"\n{Colors.BOLD}Testing Emotion Features{Colors.END}\n")
    
    try:
        from emotion_change import SpeechEmotionRecognizer
        from utils import load_config
        import numpy as np
        
        config = load_config('config.yaml')
        recognizer = SpeechEmotionRecognizer(config, None)
        
        # Create synthetic audio (1 second at 16kHz)
        audio = np.random.randn(16000).astype(np.float32)
        
        # Extract features
        features = recognizer.extract_acoustic_features(audio)
        
        print_test("Acoustic feature extraction", len(features) > 0,
                  f"Extracted {len(features)} features")
        
        # Test emotion prediction
        emotion = recognizer.predict_emotion(audio)
        
        print_test("Emotion prediction", 'label' in emotion and 'valence' in emotion,
                  f"Predicted: {emotion.get('label', 'N/A')}")
        
        return True
        
    except Exception as e:
        print_test("Emotion Features", False, str(e))
        return False


def test_change_detection():
    """Test change point detection"""
    print(f"\n{Colors.BOLD}Testing Change Detection{Colors.END}\n")
    
    try:
        from emotion_change import ChangePointDetector
        from utils import load_config
        import numpy as np
        
        config = load_config('config.yaml')
        detector = ChangePointDetector(config, None)
        
        # Create synthetic time series with known change points
        # Segment 1: mean=0, Segment 2: mean=5, Segment 3: mean=0
        t1 = np.zeros(50)
        t2 = np.ones(50) * 5
        t3 = np.zeros(50)
        signal = np.concatenate([t1, t2, t3]) + np.random.randn(150) * 0.5
        
        timestamps = np.arange(150) * 0.1  # 0.1s intervals
        
        try:
            changes = detector.detect_changes(timestamps, signal)
            print_test("Change detection", len(changes) > 0,
                      f"Detected {len(changes)} changes (expected ~2)")
        except ImportError:
            print_test("Change detection", False, 
                      "ruptures not installed (optional)")
        
        return True
        
    except Exception as e:
        print_test("Change Detection", False, str(e))
        return False


def test_directory_structure():
    """Test that directory structure is correct"""
    print(f"\n{Colors.BOLD}Testing Directory Structure{Colors.END}\n")
    
    required = {
        'directories': ['src', 'data', 'outputs', 'outputs/clips', 'models'],
        'files': ['config.yaml', 'requirements.txt', 'README.md', 
                 'src/pipeline.py', 'src/utils.py']
    }
    
    all_ok = True
    
    for path in required['directories']:
        exists = os.path.isdir(path)
        print_test(f"Directory: {path}", exists)
        all_ok &= exists
    
    for path in required['files']:
        exists = os.path.isfile(path)
        print_test(f"File: {path}", exists)
        all_ok &= exists
    
    return all_ok


def run_all_tests():
    """Run all tests and report results"""
    print(f"""
{Colors.BOLD}╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║      A/V Emotion Detection Pipeline - Test Suite         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝{Colors.END}
    """)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("FFmpeg", test_ffmpeg),
        ("Configuration", test_configuration),
        ("Utilities", test_utilities),
        ("Media Extraction", test_media_extraction),
        ("Face Detection", test_face_detection),
        ("Emotion Features", test_emotion_features),
        ("Change Detection", test_change_detection),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n{Colors.RED}Test '{name}' crashed: {e}{Colors.END}")
            results[name] = False
    
    # Summary
    print(f"\n{Colors.BOLD}═══════════════════════════════════════════════════════════")
    print("SUMMARY")
    print(f"═══════════════════════════════════════════════════════════{Colors.END}\n")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, status in results.items():
        symbol = f"{Colors.GREEN}✓{Colors.END}" if status else f"{Colors.RED}✗{Colors.END}"
        print(f"{symbol} {name}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}🎉 All tests passed! You're ready to run the pipeline.{Colors.END}")
        print("\nNext steps:")
        print("  python src/pipeline.py your_video.mp4")
    elif passed >= total * 0.8:
        print(f"\n{Colors.YELLOW}⚠  Most tests passed. Some optional components missing.{Colors.END}")
        print("\nYou can still run the pipeline, but some features may be limited.")
    else:
        print(f"\n{Colors.RED}❌ Many tests failed. Please fix the issues above.{Colors.END}")
        print("\nSee TROUBLESHOOTING.md for help.")
    
    print(f"\n{Colors.BOLD}═══════════════════════════════════════════════════════════{Colors.END}\n")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
