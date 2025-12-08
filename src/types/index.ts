// Track Types
export interface Track {
  id: string;
  title: string;
  artist: string;
  artist_id: string;
  album?: string;
  release_date?: string;
  duration_ms?: number;
  track_type: 'original' | 'remix' | 'collaboration' | 'production';
  bpm?: number;
  key?: string;
  genres: string[];
  labels: string[];
  producers: string[];
  remixers: string[];
  featuring: string[];
  isrc?: string;
  spotify_id?: string;
  musicbrainz_id?: string;
  discogs_id?: string;
  youtube_url?: string;
  soundcloud_url?: string;
  bandcamp_url?: string;
  cover_art_url?: string;
  popularity?: number;
  source: string;
}

export interface Artist {
  id: string;
  name: string;
  aliases: string[];
  genres: string[];
  country?: string;
  formed_year?: number;
  image_url?: string;
  spotify_id?: string;
  musicbrainz_id?: string;
  discogs_id?: string;
}

// Analysis Types
export interface Section {
  type: string;
  start_time: number;
  end_time: number;
  energy_level: number;
  is_breakdown: boolean;
  is_drop: boolean;
  confidence: number;
}

export interface AnalysisResult {
  file_path: string;
  duration: number;
  bpm: number;
  key: string;
  time_signature: number;
  energy_profile: number[];
  sections: Section[];
  best_sample_points: SamplePoint[];
  waveform_peaks: number[];
}

export interface SamplePoint {
  start_time: number;
  end_time: number;
  duration: number;
  section_type: string;
  energy_level: number;
  priority: number;
  bar_count: number;
  is_drop: boolean;
  is_breakdown: boolean;
}

// Sample Types
export interface Sample {
  id: string;
  source_file: string;
  source_track: string;
  source_artist: string;
  start_time: number;
  end_time: number;
  duration: number;
  bar_count: number;
  section_type: string;
  energy_level: number;
  bpm: number;
  key: string;
  file_path: string;
  waveform_peaks: number[];
  stems_available: boolean;
  stems: Record<string, string>;
}

// Stem Types
export interface StemInfo {
  available: boolean;
  model: string;
  device: string;
  stems: string[];
  stems_dir: string;
}

export interface StemResult {
  success: boolean;
  original_file: string;
  stems: Record<string, string>;
  error?: string;
}

// Download Types
export interface DownloadResult {
  success: boolean;
  file_path?: string;
  title?: string;
  artist?: string;
  duration?: number;
  source?: string;
  url?: string;
  error?: string;
}

// Search Filters
export interface SearchFilters {
  artist: string;
  dateFrom?: string;
  dateTo?: string;
  trackTypes: ('original' | 'remix' | 'collaboration' | 'production')[];
}

// App State Types
export interface AppState {
  currentStep: 'search' | 'results' | 'samples' | 'export';
  selectedArtist?: Artist;
  selectedTracks: Track[];
  extractedSamples: Sample[];
  filters: SearchFilters;
  barCount: number;
  extractStems: boolean;
  selectedStems: string[];
  exportPath?: string;
}

// API Response Types
export interface ApiResponse<T> {
  data?: T;
  error?: string;
}

export interface TrackSearchResponse {
  artist: string;
  filters: {
    date_from?: string;
    date_to?: string;
    track_types?: string[];
  };
  count: number;
  tracks: Track[];
}

export interface SampleExtractionResponse {
  source_file: string;
  samples: Sample[];
}
