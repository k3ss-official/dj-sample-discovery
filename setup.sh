#!/bin/bash

# DJ Sample Discovery - Setup Script
# Optimized for M4 Mini with 16GB RAM

set -e

echo "🎧 DJ Sample Discovery - Setup"
echo "================================"

# Check for required tools
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is required but not installed."
        exit 1
    fi
}

echo "Checking requirements..."
check_command node
check_command npm
check_command python3
check_command pip3
check_command ffmpeg

echo "✅ All requirements found"

# Create data directory
DATA_DIR="${DJ_DATA_DIR:-$HOME/DJ_Samples}"
echo "Creating data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"/{downloads,samples,stems,.cache,.temp}

# Install Node.js dependencies
echo ""
echo "📦 Installing Node.js dependencies..."
npm install

# Create Python virtual environment
echo ""
echo "🐍 Setting up Python environment..."
cd backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Install PyTorch for Apple Silicon (MPS support)
echo ""
echo "🍎 Configuring PyTorch for Apple Silicon..."
pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Download Demucs model (this may take a while)
echo ""
echo "🧠 Pre-downloading Demucs model (htdemucs_6s)..."
python3 -c "import demucs.pretrained; demucs.pretrained.get_model('htdemucs_6s')" 2>/dev/null || echo "Model will download on first use"

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  Created .env file. Please add your API keys for best results:"
    echo "   - SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET"
    echo "   - DISCOGS_TOKEN"
fi

cd ..

# Verify yt-dlp
echo ""
echo "📥 Verifying yt-dlp..."
source backend/venv/bin/activate
python3 -m yt_dlp --version

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  1. Start the backend:  cd backend && source venv/bin/activate && python server.py"
echo "  2. Start the frontend: npm run dev"
echo ""
echo "Or use Electron:  npm run dev"
echo ""
echo "Data will be stored in: $DATA_DIR"
