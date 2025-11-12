#!/bin/bash
# Quick installation script for Linux/macOS
# Usage: bash install.sh

set -e  # Exit on error

echo "================================================"
echo "Emotion Detection Pipeline - Installation"
echo "================================================"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Found Python $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "❌ Error: Python 3.8 or higher required"
    echo "   You have Python $PYTHON_VERSION"
    exit 1
fi

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
    echo "⚠️  Warning: Python 3.13+ may have compatibility issues"
    echo "   Recommended: Python 3.8 - 3.12"
    echo ""
fi

# Check FFmpeg
echo "🔍 Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | awk '{print $3}')
    echo "✓ Found FFmpeg $FFMPEG_VERSION"
else
    echo "❌ Error: FFmpeg not found"
    echo "   Install with: sudo apt install ffmpeg  (Ubuntu/Debian)"
    echo "              or: brew install ffmpeg      (macOS)"
    exit 1
fi

# Ask about installation method
echo ""
echo "Choose installation method:"
echo "  1) pip (Traditional, most reliable)"
echo "  2) uv (Fast, modern)"
echo ""
read -p "Enter choice [1-2] (default: 1): " INSTALL_METHOD
INSTALL_METHOD=${INSTALL_METHOD:-1}

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."

if [ "$INSTALL_METHOD" = "2" ]; then
    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        echo "❌ Error: uv not found"
        echo "   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    
    echo "Using uv..."
    uv venv --python python3
    source .venv/bin/activate
    
    echo "📥 Installing dependencies..."
    uv pip sync requirements.txt
else
    echo "Using pip..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "📥 Upgrading pip..."
    pip install --upgrade pip
    
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ Installation complete!"
echo ""

# Check for GPU
echo "🔍 Checking for CUDA GPU..."
python3 -c "import torch; print('✓ CUDA available' if torch.cuda.is_available() else '⚠️  CUDA not available (CPU mode)')"

echo ""
echo "================================================"
echo "Next Steps:"
echo "================================================"
echo ""
echo "1. Set HuggingFace token (required for pyannote):"
echo "   export HF_TOKEN='your_token_here'"
echo "   Get token from: https://huggingface.co/settings/tokens"
echo ""
echo "2. Accept model terms at:"
echo "   https://huggingface.co/pyannote/speaker-diarization-3.1"
echo ""
echo "3. Verify installation:"
echo "   python setup_check.py"
echo ""
echo "4. Test the pipeline:"
echo "   python test_pipeline.py"
echo ""
echo "5. Process a video:"
echo "   python src/pipeline.py your_video.mp4"
echo ""
echo "📚 Documentation:"
echo "   - Quick Start: QUICKSTART.md"
echo "   - Full Guide: README.md"
echo "   - Troubleshooting: TROUBLESHOOTING.md"
echo ""
echo "================================================"
