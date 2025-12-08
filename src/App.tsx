/**
 * DJ Sample Discovery - Main Application
 */
import React, { useState, useEffect, useCallback } from 'react';
import SearchPanel from './components/SearchPanel';
import TrackList from './components/TrackList';
import SampleCard from './components/SampleCard';
import ExtractionSettings from './components/ExtractionSettings';
import { Waveform } from './components/Waveform';
import { useMultiAudioPlayer } from './hooks/useAudioPlayer';
import {
  checkHealth,
  getAppInfo,
  getArtistTracks,
  downloadTrack,
  extractSamples,
  deleteSample,
  getAudioUrl,
  getStemInfo,
} from './services/api';
import type { Track, Sample, SearchFilters, StemInfo } from './types';

type AppStep = 'search' | 'select' | 'extract' | 'samples';

interface ProcessingTrack {
  track: Track;
  status: 'pending' | 'downloading' | 'analyzing' | 'extracting' | 'done' | 'error';
  error?: string;
  filePath?: string;
}

function App() {
  // App state
  const [step, setStep] = useState<AppStep>('search');
  const [isConnected, setIsConnected] = useState(false);
  const [stemInfo, setStemInfo] = useState<StemInfo | null>(null);
  
  // Search state
  const [searchFilters, setSearchFilters] = useState<SearchFilters | null>(null);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [selectedTrackIds, setSelectedTrackIds] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  
  // Extraction settings
  const [barCount, setBarCount] = useState(16);
  const [extractStems, setExtractStems] = useState(false);
  const [selectedStems, setSelectedStems] = useState<string[]>(['drums', 'bass', 'vocals', 'other']);
  const [sectionPreference, setSectionPreference] = useState<string | null>(null);
  
  // Processing state
  const [processingTracks, setProcessingTracks] = useState<ProcessingTrack[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Samples state
  const [samples, setSamples] = useState<Sample[]>([]);
  
  // Audio player
  const { currentTrackId, isPlaying, playTrack, stopAll } = useMultiAudioPlayer();

  // Check backend connection
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await checkHealth();
        setIsConnected(true);
        
        // Get stem info
        const info = await getStemInfo();
        setStemInfo(info);
      } catch (error) {
        console.error('Backend not connected:', error);
        setIsConnected(false);
      }
    };
    
    checkConnection();
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, []);

  // Handle search
  const handleSearch = useCallback(async (filters: SearchFilters) => {
    setIsSearching(true);
    setSearchFilters(filters);
    
    try {
      const result = await getArtistTracks(filters.artist, {
        dateFrom: filters.dateFrom,
        dateTo: filters.dateTo,
        trackTypes: filters.trackTypes,
      });
      
      setTracks(result.tracks);
      setSelectedTrackIds([]);
      setStep('select');
    } catch (error) {
      console.error('Search error:', error);
      alert('Failed to search for tracks. Please try again.');
    } finally {
      setIsSearching(false);
    }
  }, []);

  // Toggle track selection
  const handleToggleTrack = useCallback((trackId: string) => {
    setSelectedTrackIds(prev =>
      prev.includes(trackId)
        ? prev.filter(id => id !== trackId)
        : [...prev, trackId]
    );
  }, []);

  // Process selected tracks
  const handleStartExtraction = useCallback(async () => {
    const selected = tracks.filter(t => selectedTrackIds.includes(t.id));
    if (selected.length === 0) return;
    
    setStep('extract');
    setIsProcessing(true);
    setSamples([]);
    
    // Initialize processing state
    setProcessingTracks(selected.map(track => ({
      track,
      status: 'pending',
    })));
    
    // Process each track sequentially
    const allSamples: Sample[] = [];
    
    for (let i = 0; i < selected.length; i++) {
      const track = selected[i];
      
      // Update status to downloading
      setProcessingTracks(prev => prev.map((p, idx) =>
        idx === i ? { ...p, status: 'downloading' } : p
      ));
      
      try {
        // Download track
        const downloadResult = await downloadTrack(track.artist, track.title);
        
        if (!downloadResult.success || !downloadResult.file_path) {
          throw new Error(downloadResult.error || 'Download failed');
        }
        
        // Update status to extracting
        setProcessingTracks(prev => prev.map((p, idx) =>
          idx === i ? { ...p, status: 'extracting', filePath: downloadResult.file_path } : p
        ));
        
        // Extract samples
        const extractResult = await extractSamples(downloadResult.file_path, {
          artist: track.artist,
          title: track.title,
          barCount,
          sectionPreference: sectionPreference || undefined,
          extractStems,
          selectedStems: extractStems ? selectedStems : undefined,
          maxSamples: 3,
        });
        
        allSamples.push(...extractResult.samples);
        
        // Update status to done
        setProcessingTracks(prev => prev.map((p, idx) =>
          idx === i ? { ...p, status: 'done' } : p
        ));
        
      } catch (error: any) {
        console.error(`Error processing ${track.title}:`, error);
        setProcessingTracks(prev => prev.map((p, idx) =>
          idx === i ? { ...p, status: 'error', error: error.message } : p
        ));
      }
    }
    
    setSamples(allSamples);
    setIsProcessing(false);
    
    // Auto-advance to samples if we have any
    if (allSamples.length > 0) {
      setTimeout(() => setStep('samples'), 1000);
    }
  }, [tracks, selectedTrackIds, barCount, sectionPreference, extractStems, selectedStems]);

  // Handle sample playback
  const handlePlaySample = useCallback((sample: Sample) => {
    const filename = sample.file_path.split('/').pop() || '';
    const url = getAudioUrl(filename);
    playTrack(sample.id, url);
  }, [playTrack]);

  // Handle grab (download)
  const handleGrabSample = useCallback(async (sample: Sample) => {
    // Check if we have electron API
    if (window.electronAPI) {
      const destFolder = await window.electronAPI.selectFolder();
      if (destFolder) {
        const filename = sample.file_path.split('/').pop() || 'sample.wav';
        const destination = `${destFolder}/${filename}`;
        await window.electronAPI.copyFile(sample.file_path, destination);
        alert(`Sample saved to ${destination}`);
      }
    } else {
      // Browser fallback - open download link
      const filename = sample.file_path.split('/').pop() || '';
      const url = getAudioUrl(filename);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
    }
  }, []);

  // Handle discard
  const handleDiscardSample = useCallback(async (sample: Sample) => {
    try {
      await deleteSample(sample.id);
      setSamples(prev => prev.filter(s => s.id !== sample.id));
      if (currentTrackId === sample.id) {
        stopAll();
      }
    } catch (error) {
      console.error('Failed to discard sample:', error);
    }
  }, [currentTrackId, stopAll]);

  // Reset to search
  const handleNewSearch = useCallback(() => {
    setStep('search');
    setTracks([]);
    setSelectedTrackIds([]);
    setSamples([]);
    setProcessingTracks([]);
    stopAll();
  }, [stopAll]);

  return (
    <div className="min-h-screen bg-dj-dark">
      {/* Header */}
      <header className="border-b border-gray-800 bg-dj-darker">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-dj-accent to-dj-purple flex items-center justify-center">
                <span className="text-2xl">🎧</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">DJ Sample Discovery</h1>
                <p className="text-xs text-gray-500">AI-Powered Track Sampling</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {/* Connection status */}
              <div className="flex items-center gap-2 text-sm">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-gray-400">
                  {isConnected ? 'Connected' : 'Offline'}
                </span>
              </div>
              
              {/* Step indicator */}
              <div className="flex items-center gap-2">
                {['search', 'select', 'extract', 'samples'].map((s, i) => (
                  <div
                    key={s}
                    className={`flex items-center ${i > 0 ? 'ml-2' : ''}`}
                  >
                    {i > 0 && (
                      <div className={`w-8 h-0.5 ${step === s || ['select', 'extract', 'samples'].indexOf(step) >= i ? 'bg-dj-accent' : 'bg-gray-700'}`} />
                    )}
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                        step === s
                          ? 'bg-dj-accent text-black'
                          : ['select', 'extract', 'samples'].indexOf(step) >= ['search', 'select', 'extract', 'samples'].indexOf(s)
                          ? 'bg-dj-accent/20 text-dj-accent'
                          : 'bg-gray-800 text-gray-500'
                      }`}
                    >
                      {i + 1}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Search Step */}
        {step === 'search' && (
          <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-2">
                Find Your Artist
              </h2>
              <p className="text-gray-400">
                Search for an artist and discover their tracks for sampling
              </p>
            </div>
            
            <div className="card">
              <SearchPanel onSearch={handleSearch} isLoading={isSearching} />
            </div>
          </div>
        )}

        {/* Track Selection Step */}
        {step === 'select' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Track List */}
            <div className="lg:col-span-2">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-2xl font-bold text-white">
                    {searchFilters?.artist}'s Tracks
                  </h2>
                  <p className="text-gray-400">{tracks.length} tracks found</p>
                </div>
                <button
                  onClick={handleNewSearch}
                  className="text-sm text-gray-400 hover:text-white"
                >
                  ← New Search
                </button>
              </div>
              
              <div className="card">
                <TrackList
                  tracks={tracks}
                  selectedTracks={selectedTrackIds}
                  onToggleSelect={handleToggleTrack}
                  onSelectAll={() => setSelectedTrackIds(tracks.map(t => t.id))}
                  onDeselectAll={() => setSelectedTrackIds([])}
                />
              </div>
            </div>
            
            {/* Settings Sidebar */}
            <div className="space-y-4">
              <div className="card">
                <h3 className="text-lg font-semibold text-white mb-4">
                  Extraction Settings
                </h3>
                <ExtractionSettings
                  barCount={barCount}
                  onBarCountChange={setBarCount}
                  extractStems={extractStems}
                  onExtractStemsChange={setExtractStems}
                  selectedStems={selectedStems}
                  onSelectedStemsChange={setSelectedStems}
                  sectionPreference={sectionPreference}
                  onSectionPreferenceChange={setSectionPreference}
                  stemsAvailable={stemInfo?.available ?? false}
                />
              </div>
              
              <button
                onClick={handleStartExtraction}
                disabled={selectedTrackIds.length === 0}
                className="btn-primary w-full py-4 text-lg"
              >
                Extract Samples ({selectedTrackIds.length} tracks)
              </button>
            </div>
          </div>
        )}

        {/* Extraction Progress Step */}
        {step === 'extract' && (
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-2">
                {isProcessing ? 'Extracting Samples...' : 'Extraction Complete'}
              </h2>
              <p className="text-gray-400">
                {isProcessing
                  ? 'Please wait while we analyze and extract samples'
                  : `Found ${samples.length} samples`}
              </p>
            </div>
            
            <div className="card space-y-3">
              {processingTracks.map((pt, i) => (
                <div
                  key={pt.track.id}
                  className={`flex items-center gap-4 p-4 rounded-lg ${
                    pt.status === 'done' ? 'bg-green-500/10' :
                    pt.status === 'error' ? 'bg-red-500/10' :
                    pt.status !== 'pending' ? 'bg-dj-accent/10' :
                    'bg-gray-800/50'
                  }`}
                >
                  <div className={`w-3 h-3 rounded-full ${
                    pt.status === 'done' ? 'bg-green-500' :
                    pt.status === 'error' ? 'bg-red-500' :
                    pt.status === 'pending' ? 'bg-gray-500' :
                    'bg-dj-accent animate-pulse'
                  }`} />
                  
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-white truncate">{pt.track.title}</div>
                    <div className="text-sm text-gray-400">{pt.track.artist}</div>
                  </div>
                  
                  <div className="text-sm">
                    {pt.status === 'pending' && <span className="text-gray-500">Waiting...</span>}
                    {pt.status === 'downloading' && <span className="text-blue-400">Downloading...</span>}
                    {pt.status === 'analyzing' && <span className="text-purple-400">Analyzing...</span>}
                    {pt.status === 'extracting' && <span className="text-dj-accent">Extracting...</span>}
                    {pt.status === 'done' && <span className="text-green-400">✓ Complete</span>}
                    {pt.status === 'error' && <span className="text-red-400">{pt.error}</span>}
                  </div>
                </div>
              ))}
            </div>
            
            {!isProcessing && samples.length > 0 && (
              <button
                onClick={() => setStep('samples')}
                className="btn-primary w-full mt-6 py-4 text-lg"
              >
                View Samples →
              </button>
            )}
            
            {!isProcessing && samples.length === 0 && (
              <button
                onClick={handleNewSearch}
                className="btn-secondary w-full mt-6 py-4"
              >
                Start New Search
              </button>
            )}
          </div>
        )}

        {/* Samples View Step */}
        {step === 'samples' && (
          <div>
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-2xl font-bold text-white">
                  Extracted Samples
                </h2>
                <p className="text-gray-400">
                  {samples.length} samples ready for preview
                </p>
              </div>
              <button
                onClick={handleNewSearch}
                className="btn-secondary"
              >
                New Search
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {samples.map((sample) => (
                <SampleCard
                  key={sample.id}
                  sample={sample}
                  isPlaying={currentTrackId === sample.id && isPlaying}
                  onPlay={() => handlePlaySample(sample)}
                  onStop={stopAll}
                  onGrab={() => handleGrabSample(sample)}
                  onDiscard={() => handleDiscardSample(sample)}
                  showStems={extractStems}
                />
              ))}
            </div>
            
            {samples.length === 0 && (
              <div className="text-center py-20 text-gray-500">
                <p className="text-lg">No samples extracted yet</p>
                <button
                  onClick={() => setStep('select')}
                  className="btn-primary mt-4"
                >
                  Go Back to Selection
                </button>
              </div>
            )}
          </div>
        )}
      </main>
      
      {/* Footer */}
      <footer className="border-t border-gray-800 py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-6 text-center text-sm text-gray-500">
          DJ Sample Discovery • Built for M4 Mini • Powered by Demucs & yt-dlp
        </div>
      </footer>
    </div>
  );
}

// Type declaration for Electron API
declare global {
  interface Window {
    electronAPI?: {
      selectFolder: () => Promise<string | undefined>;
      copyFile: (source: string, destination: string) => Promise<{ success: boolean; error?: string }>;
      openFolder: (path: string) => Promise<void>;
      isElectron: boolean;
    };
  }
}

export default App;
