# 🎧 DJ Sample Discovery

AI-powered DJ track discovery and intelligent sampling tool with 6-stem separation. Built for Apple M4 Mini with 16GB RAM.

![DJ Sample Discovery](public/assets/icon.svg)

## ✨ Features

### Currently Implemented

- **🔍 Artist Discovery**: Search for artists using MusicBrainz, Discogs, and Spotify APIs
- **📅 Date Range Filtering**: Filter tracks by release date (presets or custom range)
- **🎵 Track Type Filtering**: Original / Remix / Collaboration / Production
- **📥 Auto-Download**: Fetches highest quality audio using yt-dlp from YouTube Music, YouTube, SoundCloud, Bandcamp
- **🧠 Intelligent Sampling**: AI-powered section detection (intro, verse, chorus, breakdown, drop, outro)
- **⚡ Energy Analysis**: Picks samples from high-energy sections (drops, choruses)
- **🎚️ Configurable Sample Length**: 4, 8, 16, 32, or 64 bars
- **📊 BPM & Key Detection**: Automatic tempo and musical key detection with Camelot notation
- **🔊 Waveform Visualization**: Real-time waveform display for tracks and samples
- **🎤 6-Stem Separation**: Demucs htdemucs_6s model (drums, bass, vocals, guitar, piano, other)
- **▶️ Sample Preview**: Loop-enabled playback with play/stop controls
- **💾 Grab & Discard**: Download samples as 24-bit WAV or discard unwanted ones
- **🖥️ Electron App**: Native desktop experience (macOS, Windows, Linux)

### Audio Quality

- **Format**: WAV (24-bit, 44.1kHz stereo)
- **Source Priority**: YouTube Music → YouTube → SoundCloud → Bandcamp
- **Processing**: FFmpeg for format conversion, librosa for analysis

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DJ Sample Discovery                          │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React + TypeScript)                                  │
│  ├── Search Panel (artist + filters)                            │
│  ├── Track List (selection + metadata)                          │
│  ├── Extraction Settings (bars, stems, section preference)      │
│  ├── Sample Cards (waveform, BPM, key, play/stop)              │
│  └── Audio Player (Web Audio API, loop support)                 │
├─────────────────────────────────────────────────────────────────┤
│  Backend (Python Flask + SocketIO)                              │
│  ├── Metadata Service (MusicBrainz, Discogs, Spotify)          │
│  ├── Download Service (yt-dlp multi-source)                     │
│  ├── Audio Analyzer (librosa - BPM, key, sections, energy)     │
│  ├── Sample Extractor (intelligent section-based extraction)   │
│  └── Stem Separator (Demucs htdemucs_6s, MPS acceleration)     │
├─────────────────────────────────────────────────────────────────┤
│  Storage                                                        │
│  ├── ~/DJ_Samples/downloads/   (full tracks)                   │
│  ├── ~/DJ_Samples/samples/     (extracted samples)             │
│  └── ~/DJ_Samples/stems/       (separated stems)               │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **macOS** with M4 Mini (or any Apple Silicon)
- **Node.js** 18+ and npm
- **Python** 3.10+
- **FFmpeg** (install via Homebrew: `brew install ffmpeg`)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dj-sample-discovery.git
cd dj-sample-discovery

# Run setup script
chmod +x setup.sh
./setup.sh

# Or manual setup:
npm install
cd backend && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Running the App

**Development Mode (recommended for debugging):**

```bash
# Terminal 1: Start Python backend
cd backend
source venv/bin/activate
python server.py

# Terminal 2: Start React frontend
npm run dev
```

**Electron Mode:**

```bash
npm run dev
# This starts both backend and frontend
```

### API Keys (Optional but Recommended)

Create `backend/.env` from `backend/.env.example`:

```bash
# Spotify (for better search results)
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret

# Discogs (for detailed credits)
DISCOGS_TOKEN=your_token
```

## 📱 Usage Guide

### 1. Search for Artist
- Enter artist name in the search box
- Select from autocomplete suggestions
- Choose date range (Last Year, 5 Years, 10 Years, or All Time)
- Select track types to include

### 2. Select Tracks
- Browse the track list
- Click tracks to select/deselect
- Use "Select All" for batch processing

### 3. Configure Extraction
- **Sample Length**: Choose bars (16 bars ≈ 32s at 120 BPM)
- **Section Preference**: Auto, Drop, Chorus, Breakdown, or Verse
- **Stem Separation**: Enable for 6-stem output

### 4. Extract & Preview
- Click "Extract Samples" to start processing
- Wait for download, analysis, and extraction
- Preview samples with Play/Stop buttons

### 5. Grab or Discard
- **Grab**: Downloads the 24-bit WAV to your chosen folder
- **Discard**: Removes the sample permanently

## 🎛️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/search/artists?q=` | GET | Search artists |
| `/api/artist/{name}/tracks` | GET | Get artist tracks with filters |
| `/api/download` | POST | Download a track |
| `/api/analyze` | POST | Analyze audio file |
| `/api/samples/extract` | POST | Extract intelligent samples |
| `/api/samples/custom` | POST | Extract custom time range |
| `/api/stems/separate` | POST | Separate into 6 stems |
| `/api/audio/{filename}` | GET | Stream audio file |

## ⚙️ Configuration

### Sample Extraction Settings

| Parameter | Options | Default |
|-----------|---------|---------|
| `bar_count` | 4, 8, 16, 32, 64 | 16 |
| `section_preference` | null (auto), drop, chorus, breakdown, verse | null |
| `extract_stems` | true/false | false |
| `max_samples` | 1-10 | 3 |

### Demucs Stem Options

- `drums` - Drum tracks
- `bass` - Bass lines
- `vocals` - Lead and backing vocals
- `guitar` - Guitar tracks
- `piano` - Piano/keys
- `other` - Everything else

## 🔧 Performance on M4 Mini

| Operation | Time (approx) |
|-----------|---------------|
| Track download | 5-15 seconds |
| Audio analysis | 3-5 seconds |
| Sample extraction | 1-2 seconds |
| Stem separation | 60-90 seconds |

Tips for best performance:
- Close other heavy applications during stem separation
- Process 3-5 tracks at a time max
- Demucs uses MPS (Metal Performance Shaders) for GPU acceleration

## 📂 Data Storage

All data is stored in `~/DJ_Samples/` (configurable via `DJ_DATA_DIR`):

```
~/DJ_Samples/
├── downloads/     # Full downloaded tracks
├── samples/       # Extracted samples (24-bit WAV)
├── stems/         # Separated stem folders
├── .cache/        # API response cache
├── .temp/         # Temporary processing files
└── dj_sampler.log # Application logs
```

## 🛠️ Development

### Project Structure

```
dj-sample-discovery/
├── src/                    # React frontend
│   ├── components/         # UI components
│   ├── hooks/              # Custom React hooks
│   ├── services/           # API client
│   └── types/              # TypeScript types
├── backend/                # Python backend
│   ├── services/           # Core services
│   ├── config.py           # Configuration
│   └── server.py           # Flask API server
├── electron/               # Electron main process
├── public/                 # Static assets
└── package.json            # Node dependencies
```

### Tech Stack

**Frontend:**
- React 18 + TypeScript
- Tailwind CSS
- Vite
- Web Audio API
- Axios

**Backend:**
- Flask + Flask-SocketIO
- yt-dlp (audio download)
- librosa (audio analysis)
- Demucs (stem separation)
- MusicBrainz / Discogs / Spotify APIs

## 🚧 Future Enhancements

- [ ] Harmonic mixing suggestions (Camelot wheel compatibility)
- [ ] AI-powered "similar samples" discovery
- [ ] Export to DAW formats (Ableton .als, Logic .band)
- [ ] Batch stem export
- [ ] Sample library management
- [ ] Rekordbox/Serato crate export
- [ ] Real-time collaborative crates
- [ ] Mobile companion app

## 📄 License

MIT License - Feel free to use, modify, and distribute.

## 🙏 Credits

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Audio downloading
- [Demucs](https://github.com/facebookresearch/demucs) - Stem separation
- [librosa](https://librosa.org/) - Audio analysis
- [MusicBrainz](https://musicbrainz.org/) - Music metadata

---

**Built with ❤️ for DJs who dig deep**
