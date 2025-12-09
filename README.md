# 🎧 DJ Sample Discovery - SOTA Edition

**AI-Powered Track Discovery & Intelligent Sampling Tool**

> State-of-the-art music structure analysis, harmonic mixing, and 6-stem separation optimized for Apple M4 Mini

[![Version](https://img.shields.io/badge/version-2.0.0--SOTA-blue.svg)](https://github.com/k3ss-official/dj-sample-discovery)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://python.org)
[![React](https://img.shields.io/badge/react-18.2-blue.svg)](https://reactjs.org)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

---

## 🚀 What's New in SOTA Edition

### 🎯 State-of-the-Art Features

| Feature | Description |
|---------|-------------|
| **Advanced Structure Analysis** | Self-similarity matrix-based segmentation detects intro, verse, chorus, breakdown, drop, bridge, outro |
| **Smart Sample Selection** | AI avoids silence, picks energy peaks, aligns to beat grid, scores loop quality |
| **Full Camelot Wheel** | 24-key harmonic mixing with compatible key suggestions and energy boost mixing |
| **Mashup Potential Scoring** | Calculate compatibility between any two samples for mashups |
| **Audio Fingerprinting** | Chromaprint-inspired similarity detection across your sample library |
| **Semantic Audio Search** | Find samples by characteristics (bright, percussive, energetic, etc.) |
| **DAW Export** | Rekordbox XML, Serato crates, M3U playlists with metadata |
| **Beat Quantization** | Perfect bar-aligned loop points with downbeat detection |

---

## ✨ Core Features

### 🔍 **Track Discovery**
- Search artists via MusicBrainz, Discogs, Spotify APIs
- Filter by date range (1990-2024)
- Track type filtering: Original, Remix, Collaboration, Production

### 🎵 **Intelligent Audio Download**
- Best quality audio via yt-dlp (YouTube Music, SoundCloud, Bandcamp)
- Automatic source selection
- 24-bit WAV output format

### 🧠 **SOTA Structure Analysis**
- Self-similarity matrix segmentation
- Beat tracking with downbeat detection
- Section classification: intro, verse, chorus, breakdown, drop, bridge, outro
- Energy profile mapping
- Silence detection & avoidance

### 🎹 **Harmonic Analysis**
- Accurate key detection using Krumhansl-Schmuckler profiles
- Full Camelot wheel mapping (24 keys)
- Compatible key suggestions for mixing:
  - **Same key** (100% match)
  - **+1/-1** (smooth transitions)
  - **Relative major/minor** (mood change)
  - **Energy boost** (+7 semitones)

### 🎚️ **Intelligent Sampling**
- Configurable sample length: 4, 8, 16, 32, 64 bars
- Section preference (drop, chorus, breakdown, etc.)
- Score-based sample point ranking:
  - Energy score
  - Beat alignment score
  - Silence avoidance score
  - Loop quality score

### 🎛️ **6-Stem Separation (Demucs)**
- htdemucs_6s model
- Stems: drums, bass, vocals, guitar, piano, other
- Apple M4 MPS acceleration
- Per-sample or batch processing

### 📊 **Mashup Potential**
- BPM compatibility analysis
- Harmonic compatibility scoring
- Energy level matching
- Structure complementarity
- Overall recommendation: Excellent / Good / Possible / Difficult / Avoid

### 🔊 **Audio Fingerprinting**
- Spectral peak extraction
- Constellation map generation
- Multi-dimensional similarity scoring
- Duplicate detection
- Similar sample finder

### 📁 **DAW Export**
- **Rekordbox XML**: Full metadata, cue points, colors
- **Serato**: CSV and M3U8 for Smart Crates
- **M3U8**: Universal playlist format
- **JSON**: Complete backup/transfer

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Electron Desktop                          │
├─────────────────────────────────────────────────────────────────┤
│  React + TypeScript Frontend                                     │
│  ├── SearchPanel (Artist/Date/Type filters)                     │
│  ├── TrackList (Selection & batch)                              │
│  ├── SOTAPanel (Camelot wheel, structure viz)                   │
│  ├── MashupScorer (Compatibility calculator)                    │
│  ├── SampleCard (Waveform, play/grab/discard)                   │
│  └── ExtractionSettings (Bars, stems, section)                  │
├─────────────────────────────────────────────────────────────────┤
│  Flask + SocketIO Backend (Python)                               │
│  ├── Metadata Service (MusicBrainz, Discogs, Spotify)           │
│  ├── Download Service (yt-dlp multi-source)                     │
│  ├── SOTA Analyzer (Structure, beats, harmony)                  │
│  ├── Sample Extractor (Intelligent point selection)             │
│  ├── Stem Separator (Demucs htdemucs_6s)                        │
│  ├── Harmonic Mixer (Camelot wheel, mashup scoring)             │
│  ├── Audio Fingerprint (Similarity, duplicates)                 │
│  └── DAW Exporter (Rekordbox, Serato, M3U)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- FFmpeg (for audio processing)

### Quick Start

```bash
# Clone repository
git clone https://github.com/k3ss-official/dj-sample-discovery.git
cd dj-sample-discovery

# Run setup script
./setup.sh

# Or manual setup:
npm install
cd backend && pip install -r requirements.txt && cd ..
```

### Environment Variables (Optional)

Create `.env` in `backend/` directory:

```env
# Optional API keys for enhanced metadata
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
DISCOGS_TOKEN=your_token

# Configuration
DJ_DATA_DIR=~/DJ_Samples
DJ_HOST=127.0.0.1
DJ_PORT=5555
DJ_DEBUG=false
```

---

## 🚀 Usage

### Start the Application

```bash
# Terminal 1: Start Python backend
cd backend
source venv/bin/activate  # If using venv
python server_sota.py

# Terminal 2: Start React frontend
npm run dev

# Access at http://localhost:5173
```

### Or with Electron (Desktop App)

```bash
npm run dev  # Starts both backend and Electron app
```

---

## 🎯 Workflow

### 1. Search
- Enter artist name
- Set date range (optional)
- Filter track types

### 2. Select
- Browse discovered tracks
- Select tracks for sampling
- Configure extraction settings:
  - Bar count: 4/8/16/32/64
  - Section preference
  - Enable stem separation

### 3. Extract
- Watch progress in real-time
- Automatic download → analysis → extraction

### 4. Review Samples
- **Waveform visualization**
- **Play/Stop** with Web Audio
- **Camelot key display**
- **Compatible keys** for mixing
- **Grab** (download to folder)
- **Discard** (remove sample)

### 5. Advanced Features
- **Mashup Scorer**: Compare any two samples
- **Structure View**: See track segments
- **Export**: Rekordbox/Serato/M3U

---

## 📊 API Endpoints

### SOTA Analysis
```
POST /api/sota/analyze
POST /api/sota/structure
```

### Harmonic Mixing
```
GET  /api/harmonic/compatible?key=Am
POST /api/harmonic/mix-score
POST /api/harmonic/mashup-score
POST /api/harmonic/suggest-order
```

### Fingerprinting
```
POST /api/fingerprint/generate
POST /api/fingerprint/compare
POST /api/fingerprint/find-similar
POST /api/semantic/describe
```

### DAW Export
```
POST /api/export/rekordbox
POST /api/export/serato
POST /api/export/m3u
POST /api/export/json
```

---

## 🔧 Configuration

### Sample Bar Options
```python
SAMPLE_BAR_OPTIONS = [4, 8, 16, 32, 64]
DEFAULT_SAMPLE_BARS = 16
```

### Demucs Settings (M4 Optimized)
```python
DEMUCS_MODEL = 'htdemucs_6s'
DEMUCS_DEVICE = 'mps'  # Apple Silicon GPU
DEMUCS_STEMS = ['drums', 'bass', 'vocals', 'guitar', 'piano', 'other']
```

### Audio Quality
```python
SAMPLE_RATE = 44100
BIT_DEPTH = 24
AUDIO_FORMAT = 'wav'
```

---

## 📁 Project Structure

```
dj-sample-discovery/
├── backend/
│   ├── server_sota.py      # SOTA Flask server
│   ├── server.py           # Legacy server
│   ├── config.py           # Configuration
│   ├── requirements.txt    # Python dependencies
│   └── services/
│       ├── sota_analyzer.py       # SOTA structure analysis
│       ├── harmonic_mixer.py      # Camelot wheel & mashup
│       ├── audio_fingerprint.py   # Fingerprinting
│       ├── daw_exporter.py        # Rekordbox/Serato
│       ├── audio_analyzer.py      # BPM/Key detection
│       ├── sample_extractor.py    # Sample extraction
│       ├── stem_separator.py      # Demucs integration
│       ├── download_service.py    # yt-dlp downloader
│       └── metadata_service.py    # API aggregation
├── src/
│   ├── App.tsx             # Main React app
│   ├── main.tsx            # Entry point
│   └── components/
│       ├── SOTAPanel.tsx          # SOTA features UI
│       ├── MashupScorer.tsx       # Mashup calculator
│       ├── SearchPanel.tsx        # Artist search
│       ├── TrackList.tsx          # Track selection
│       ├── SampleCard.tsx         # Sample preview
│       ├── Waveform.tsx           # Wavesurfer.js
│       └── ExtractionSettings.tsx # Settings panel
├── electron/
│   ├── main.js             # Electron main process
│   └── preload.js          # IPC bridge
├── package.json            # NPM dependencies
├── setup.sh                # Installation script
└── README.md               # This file
```

---

## 🛠️ Tech Stack

### Backend
- **Python 3.10+**
- **Flask** + Flask-SocketIO
- **librosa** - Audio analysis
- **Demucs** - Stem separation
- **yt-dlp** - Audio download
- **MusicBrainz/Discogs/Spotify APIs** - Metadata

### Frontend
- **React 18** + TypeScript
- **Tailwind CSS** - Styling
- **wavesurfer.js** - Waveform visualization
- **Web Audio API** - Audio playback

### Desktop
- **Electron 28** - Desktop wrapper

---

## 📈 Performance (M4 Mini 16GB)

| Operation | Time |
|-----------|------|
| SOTA Analysis (5min track) | ~15 seconds |
| Sample Extraction | ~2 seconds |
| Stem Separation (6 stems) | ~45 seconds |
| Fingerprint Generation | ~3 seconds |

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Credits

- **Demucs** by Facebook Research
- **yt-dlp** maintainers
- **librosa** for audio analysis
- **MusicBrainz**, **Discogs**, **Spotify** APIs

---

## 🔮 Roadmap

- [ ] Real-time audio preview in browser
- [ ] Cloud sync for sample library
- [ ] Ableton/FL Studio export
- [ ] AI-powered "similar artists" discovery
- [ ] Batch stem export
- [ ] Mobile companion app
- [ ] Plugin versions (VST/AU)

---

**Made with ❤️ for DJs and Producers**

*Built for M4 Mini • Powered by AI • SOTA Quality*
