# Quick installation script for Windows PowerShell
# Usage: .\install.ps1

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Emotion Detection Pipeline - Installation" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "🔍 Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1 | Out-String
    if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        Write-Host "Found Python $major.$minor.$($matches[3])" -ForegroundColor Green
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
            Write-Host "❌ Error: Python 3.8 or higher required" -ForegroundColor Red
            Write-Host "   You have Python $major.$minor" -ForegroundColor Red
            exit 1
        }
        
        if ($major -eq 3 -and $minor -ge 13) {
            Write-Host "⚠️  Warning: Python 3.13+ may have compatibility issues" -ForegroundColor Yellow
            Write-Host "   Recommended: Python 3.8 - 3.12" -ForegroundColor Yellow
            Write-Host ""
        }
    }
} catch {
    Write-Host "❌ Error: Python not found or not in PATH" -ForegroundColor Red
    Write-Host "   Download from: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Check FFmpeg
Write-Host "🔍 Checking FFmpeg..." -ForegroundColor Yellow
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-String -Pattern "ffmpeg version" | Select-Object -First 1
    Write-Host "✓ Found FFmpeg" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: FFmpeg not found" -ForegroundColor Red
    Write-Host "   Install with Chocolatey: choco install ffmpeg" -ForegroundColor Red
    Write-Host "   Or download from: https://ffmpeg.org/download.html" -ForegroundColor Red
    exit 1
}

# Ask about CUDA
Write-Host ""
$cudaChoice = Read-Host "Do you have NVIDIA GPU with CUDA? (y/n) [default: n]"
if ([string]::IsNullOrEmpty($cudaChoice)) { $cudaChoice = "n" }

# Create virtual environment
Write-Host ""
Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "📥 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "📥 Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow

if ($cudaChoice -eq "y") {
    Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "Installing PyTorch (CPU version)..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio
}

# Install remaining requirements
pip install -r requirements.txt

Write-Host ""
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host ""

# Check for GPU
Write-Host "🔍 Checking for CUDA GPU..." -ForegroundColor Yellow
python -c "import torch; print('✓ CUDA available' if torch.cuda.is_available() else '⚠️  CUDA not available (CPU mode)')"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Set HuggingFace token (required for pyannote):" -ForegroundColor Yellow
Write-Host '   $env:HF_TOKEN = "your_token_here"' -ForegroundColor White
Write-Host "   Get token from: https://huggingface.co/settings/tokens" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Accept model terms at:" -ForegroundColor Yellow
Write-Host "   https://huggingface.co/pyannote/speaker-diarization-3.1" -ForegroundColor White
Write-Host ""
Write-Host "3. Verify installation:" -ForegroundColor Yellow
Write-Host "   python setup_check.py" -ForegroundColor White
Write-Host ""
Write-Host "4. Test the pipeline:" -ForegroundColor Yellow
Write-Host "   python test_pipeline.py" -ForegroundColor White
Write-Host ""
Write-Host "5. Process a video:" -ForegroundColor Yellow
Write-Host "   python src/pipeline.py your_video.mp4" -ForegroundColor White
Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor Cyan
Write-Host "   - Quick Start: QUICKSTART.md" -ForegroundColor White
Write-Host "   - Full Guide: README.md" -ForegroundColor White
Write-Host "   - Troubleshooting: TROUBLESHOOTING.md" -ForegroundColor White
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "TIP: Keep this PowerShell window open to use the virtual environment" -ForegroundColor Yellow
Write-Host "     Or activate later with: .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
