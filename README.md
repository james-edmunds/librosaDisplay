# 🎵 Librosa Audio Feature Analyzer

A beautiful web application for analyzing and visualizing audio features using librosa.

## Features

- **Comprehensive Audio Analysis**: Extract and visualize multiple librosa features including:
  - Waveforms
  - Spectrograms (linear and mel-frequency)
  - MFCCs (Mel-frequency Cepstral Coefficients)
  - Chromagrams
  - Spectral features (centroid, rolloff, zero-crossing rate)
  - Tempo and beat tracking

- **Interactive Web Interface**: Simple, beautiful dark-themed interface
- **Audio Playback**: Play MP3 files directly in the browser while viewing visualizations
- **Real-time Processing**: Features are computed on-demand when you select a file

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add MP3 Files**:
   ```bash
   # Place your MP3 files in the data/audio directory
   cp your_music.mp3 data/audio/
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Open in Browser**:
   Navigate to `http://localhost:5000`

## Usage

1. Select an MP3 file from the dropdown menu
2. Click "Analyze Audio" to process the file
3. View the generated visualizations and listen to the audio
4. Each visualization shows different aspects of the audio:
   - **Waveform**: Raw audio signal over time
   - **Spectrogram**: Frequency content over time
   - **Mel-frequency Spectrogram**: Perceptually-weighted frequency analysis
   - **MFCCs**: Compact representation used in speech/music analysis
   - **Chromagram**: Pitch class profiles over time
   - **Spectral Features**: Various frequency-domain characteristics
   - **Beat Tracking**: Detected beats and estimated tempo

## Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`
- MP3 files in the `data/audio` directory

## Project Structure

```
librosaDisplay/
├── app.py              # Flask backend
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Web interface
└── data/
    └── audio/         # MP3 files go here
```

Enjoy exploring your audio files! 🎶