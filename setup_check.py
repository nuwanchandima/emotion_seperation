#!/usr/bin/env python
"""
Setup script for the A/V Emotion Detection Pipeline
Verifies dependencies and environment
"""

import sys
import subprocess
import os
from pathlib import Path


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_python_version():
    """Check Python version"""
    print_section("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✓ Python version OK")
        return True


def check_ffmpeg():
    """Check if FFmpeg is installed"""
    print_section("Checking FFmpeg")
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Get version from first line
        version_line = result.stdout.split('\n')[0]
        print(f"FFmpeg: {version_line}")
        print("✓ FFmpeg installed")
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found")
        print("\nInstall FFmpeg:")
        print("  Windows: https://ffmpeg.org/download.html")
        print("  Linux: sudo apt install ffmpeg")
        print("  Mac: brew install ffmpeg")
        return False


def check_cuda():
    """Check CUDA availability"""
    print_section("Checking CUDA (Optional)")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available - will use CPU")
            print("  (GPU recommended for better performance)")
            return False
            
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return False


def check_dependencies():
    """Check if key dependencies are installed"""
    print_section("Checking Dependencies")
    
    required = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'librosa': 'librosa',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm'
    }
    
    optional = {
        'pyannote.audio': 'pyannote.audio (diarization)',
        'transformers': 'transformers (emotion recognition)',
        'ruptures': 'ruptures (change detection)',
        'facenet_pytorch': 'FaceNet (face recognition)'
    }
    
    missing_required = []
    missing_optional = []
    
    print("\nRequired packages:")
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ❌ {name}")
            missing_required.append(name)
            
    print("\nOptional packages (for best performance):")
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name}")
            missing_optional.append(name)
            
    return len(missing_required) == 0, missing_required, missing_optional


def check_hf_token():
    """Check HuggingFace token for pyannote"""
    print_section("Checking HuggingFace Token (for pyannote)")
    
    token = os.environ.get('HF_TOKEN')
    
    if token:
        print(f"✓ HF_TOKEN found: {token[:10]}...")
        return True
    else:
        print("⚠ HF_TOKEN not set")
        print("\nFor speaker diarization, you need:")
        print("  1. Accept terms: https://huggingface.co/pyannote/speaker-diarization")
        print("  2. Get token: https://huggingface.co/settings/tokens")
        print("  3. Set environment variable:")
        print("     Windows: $env:HF_TOKEN='your_token'")
        print("     Linux/Mac: export HF_TOKEN='your_token'")
        return False


def check_directory_structure():
    """Verify directory structure"""
    print_section("Checking Directory Structure")
    
    required_dirs = [
        'src',
        'data',
        'outputs',
        'outputs/clips',
        'models'
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ (creating...)")
            os.makedirs(dir_path, exist_ok=True)
            all_ok = False
            
    required_files = [
        'config.yaml',
        'requirements.txt',
        'src/pipeline.py',
        'src/utils.py'
    ]
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing!)")
            all_ok = False
            
    return all_ok


def install_dependencies():
    """Offer to install dependencies"""
    print_section("Installing Dependencies")
    
    response = input("\nInstall all dependencies now? (y/n): ")
    
    if response.lower() == 'y':
        print("\nInstalling packages...")
        
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                check=True
            )
            print("\n✓ Dependencies installed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Installation failed: {e}")
            return False
    else:
        print("\nSkipping installation. Run manually:")
        print("  pip install -r requirements.txt")
        return False


def main():
    """Run all checks"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║      A/V Emotion Detection Pipeline - Setup Check        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # Run checks
    results['python'] = check_python_version()
    results['ffmpeg'] = check_ffmpeg()
    results['dirs'] = check_directory_structure()
    results['cuda'] = check_cuda()
    
    # Check dependencies (try to import)
    deps_ok, missing_required, missing_optional = check_dependencies()
    results['dependencies'] = deps_ok
    
    results['hf_token'] = check_hf_token()
    
    # Summary
    print_section("Summary")
    
    critical_ok = results['python'] and results['ffmpeg'] and results['dirs']
    
    if critical_ok and results['dependencies']:
        print("\n✓ All critical requirements met!")
        print("\nYou're ready to run the pipeline:")
        print("  python src/pipeline.py your_video.mp4")
        
        if not results['cuda']:
            print("\n⚠ Note: No GPU detected. Processing will be slower.")
            
        if not results['hf_token']:
            print("\n⚠ Note: Set HF_TOKEN for speaker diarization.")
            
    elif critical_ok and not results['dependencies']:
        print("\n⚠ Dependencies missing")
        
        if missing_required:
            print(f"\nRequired: {', '.join(missing_required)}")
            
        install_dependencies()
        
    else:
        print("\n❌ Critical requirements missing")
        print("\nPlease fix the issues above before proceeding.")
        
        if not results['python']:
            print("  • Install Python 3.8+")
        if not results['ffmpeg']:
            print("  • Install FFmpeg")
        if not results['dirs']:
            print("  • Check directory structure")
            
    print("\n" + "=" * 60)
    print("\nFor more help, see QUICKSTART.md")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
