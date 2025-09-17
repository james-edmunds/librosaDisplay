# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Flask-based web application for analyzing and visualizing audio features using the librosa library. It provides a beautiful dark-themed interface for uploading MP3 files and generating comprehensive audio feature visualizations.

## Development Commands

- **Install dependencies**: `pip install -r requirements.txt`
- **Run the application**: `python app.py` (starts on http://localhost:8080)
- **Add test audio files**: Place MP3 files in `data/audio/` directory

## Architecture

### Core Components

- **Flask Backend (`app.py`)**: Single-file Flask application that handles audio processing and visualization generation
- **Audio Processing**: Uses librosa for comprehensive feature extraction including MFCCs, spectrograms, chromagrams, tempo detection, and harmonic-percussive separation
- **Visualization Engine**: Matplotlib-based system generating 14 different plot types with dark theme styling
- **Web Interface (`templates/index.html`)**: Single-page application with responsive design and real-time audio playback

### Key Technical Details

- **Audio Directory**: All MP3 files must be placed in `data/audio/` folder
- **Plot Generation**: All visualizations are generated server-side and returned as base64-encoded PNG images
- **Styling**: Uses matplotlib's 'dark_background' style with custom color schemes (#00ff88, #45b7d1, #ff6b6b)
- **Error Handling**: Graceful degradation when individual visualizations fail
- **Audio Serving**: Direct file serving through Flask for web audio playback

### Visualization Types

The application generates 14 different visualization types:
- Basic: waveform, spectrogram, mel_spectrogram, mfcc, chroma
- Advanced: spectral_features, tempo_beats, onset_strength, harmonic_percussive
- Comparative: mfcc_deltas, chroma_comparison
- Specialized: tonnetz, poly_features, advanced_spectral

### API Endpoints

- `GET /`: Main application interface
- `GET /api/files`: Returns list of available MP3 files
- `GET /api/analyze/<filename>`: Processes audio file and returns all visualizations
- `GET /data/audio/<filename>`: Serves audio files for web playback

## Dependencies

Core libraries: Flask, librosa, numpy, matplotlib, seaborn. All audio processing depends on librosa's comprehensive feature extraction capabilities.