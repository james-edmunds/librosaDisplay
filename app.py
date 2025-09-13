from flask import Flask, render_template, jsonify, request, send_file
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
import json

app = Flask(__name__)

# Configuration
AUDIO_FOLDER = 'data/audio'
PLOT_STYLE = 'dark_background'
plt.style.use(PLOT_STYLE)

def get_audio_files():
    """Get list of MP3 files in the audio directory"""
    if not os.path.exists(AUDIO_FOLDER):
        return []
    return [f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.mp3')]

def extract_all_features(file_path):
    """Extract comprehensive librosa features from audio file"""
    try:
        # Load audio
        y, sr = librosa.load(file_path)
        
        # Basic features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Energy features
        rms_energy = librosa.feature.rms(y=y)[0]
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        
        # Onset detection and strength
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.times_like(onset_frames, sr=sr)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Harmonic and percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Tempo confidence
        tempo_confidence = librosa.beat.tempo(y=y, sr=sr)[1] if len(librosa.beat.tempo(y=y, sr=sr)) > 1 else 0.0
        
        # Delta features (first derivatives)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Mel-frequency features
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        
        # Poly features
        poly_features = librosa.feature.poly_features(y=y, sr=sr)
        
        return {
            'audio_data': y,
            'sample_rate': sr,
            'tempo': float(tempo),
            'tempo_confidence': float(tempo_confidence),
            'beats': beats,
            'spectral_centroids': spectral_centroids,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_flatness': spectral_flatness,
            'zero_crossing_rate': zero_crossing_rate,
            'rms_energy': rms_energy,
            'mfccs': mfccs,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2,
            'chroma': chroma,
            'chroma_cqt': chroma_cqt,
            'spectral_contrast': spectral_contrast,
            'tonnetz': tonnetz,
            'onset_frames': onset_frames,
            'onset_times': onset_times,
            'onset_strength': onset_strength,
            'y_harmonic': y_harmonic,
            'y_percussive': y_percussive,
            'mel_spectrogram': mel_spectrogram,
            'poly_features': poly_features
        }
    except Exception as e:
        return {'error': str(e)}

def create_visualization(features, plot_type):
    """Create beautiful visualizations for different feature types"""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1e1e1e')
    
    y = features['audio_data']
    sr = features['sample_rate']
    
    if plot_type == 'waveform':
        time = np.linspace(0, len(y) / sr, len(y))
        ax.plot(time, y, color='#00ff88', alpha=0.8, linewidth=0.5)
        ax.set_title('Waveform', color='white', size=16)
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Amplitude', color='white')
        
    elif plot_type == 'spectrogram':
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax, cmap='viridis')
        ax.set_title('Spectrogram', color='white', size=16)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
    elif plot_type == 'mel_spectrogram':
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='plasma')
        ax.set_title('Mel-frequency Spectrogram', color='white', size=16)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
    elif plot_type == 'mfcc':
        img = librosa.display.specshow(features['mfccs'], x_axis='time', ax=ax, cmap='coolwarm')
        ax.set_title('MFCCs', color='white', size=16)
        ax.set_ylabel('MFCC Coefficients', color='white')
        fig.colorbar(img, ax=ax)
        
    elif plot_type == 'chroma':
        img = librosa.display.specshow(features['chroma'], y_axis='chroma', x_axis='time', ax=ax, cmap='Blues')
        ax.set_title('Chromagram', color='white', size=16)
        fig.colorbar(img, ax=ax)
        
    elif plot_type == 'spectral_features':
        time = np.linspace(0, len(features['spectral_centroids']) * 512 / sr, len(features['spectral_centroids']))
        ax.plot(time, features['spectral_centroids'], label='Spectral Centroid', color='#ff6b6b', linewidth=2)
        ax.plot(time, features['spectral_rolloff'], label='Spectral Rolloff', color='#4ecdc4', linewidth=2)
        ax.plot(time, features['zero_crossing_rate'] * 1000, label='ZCR (x1000)', color='#45b7d1', linewidth=2)
        ax.set_title('Spectral Features', color='white', size=16)
        ax.set_xlabel('Time (s)', color='white')
        ax.legend()
        
    elif plot_type == 'tempo_beats':
        beat_times = librosa.frames_to_time(features['beats'], sr=sr)
        ax.vlines(beat_times, -1, 1, color='#ff9f43', alpha=0.8, linewidth=2, label=f'Beats (Tempo: {features["tempo"]:.1f} BPM)')
        
        time = np.linspace(0, len(y) / sr, len(y))
        ax.plot(time, y, color='#00ff88', alpha=0.6, linewidth=0.5)
        ax.set_title('Beat Tracking', color='white', size=16)
        ax.set_xlabel('Time (s)', color='white')
        ax.legend()
        
    elif plot_type == 'advanced_spectral':
        time = np.linspace(0, len(features['spectral_bandwidth']) * 512 / sr, len(features['spectral_bandwidth']))
        ax.plot(time, features['spectral_bandwidth'], label='Spectral Bandwidth', color='#ff6b6b', linewidth=2)
        ax.plot(time, features['spectral_flatness'] * 1000, label='Spectral Flatness (x1000)', color='#4ecdc4', linewidth=2)
        ax.plot(time, features['rms_energy'] * 10, label='RMS Energy (x10)', color='#45b7d1', linewidth=2)
        ax.set_title('Advanced Spectral Features', color='white', size=16)
        ax.set_xlabel('Time (s)', color='white')
        ax.legend()
        
    elif plot_type == 'onset_strength':
        time = librosa.frames_to_time(np.arange(len(features['onset_strength'])), sr=sr)
        ax.plot(time, features['onset_strength'], color='#ff9f43', linewidth=2)
        onset_times = librosa.frames_to_time(features['onset_frames'], sr=sr)
        ax.vlines(onset_times, 0, max(features['onset_strength']), color='#ff6b6b', alpha=0.8, linewidth=2, label='Detected Onsets')
        ax.set_title('Onset Strength Function', color='white', size=16)
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Onset Strength', color='white')
        ax.legend()
        
    elif plot_type == 'harmonic_percussive':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.patch.set_facecolor('#1e1e1e')
        
        time = np.linspace(0, len(features['y_harmonic']) / sr, len(features['y_harmonic']))
        ax1.plot(time, features['y_harmonic'], color='#00ff88', alpha=0.8, linewidth=0.5)
        ax1.set_title('Harmonic Component', color='white', size=14)
        ax1.set_ylabel('Amplitude', color='white')
        ax1.tick_params(colors='white')
        
        ax2.plot(time, features['y_percussive'], color='#ff6b6b', alpha=0.8, linewidth=0.5)
        ax2.set_title('Percussive Component', color='white', size=14)
        ax2.set_xlabel('Time (s)', color='white')
        ax2.set_ylabel('Amplitude', color='white')
        ax2.tick_params(colors='white')
        
        for axis in [ax1, ax2]:
            axis.spines['bottom'].set_color('white')
            axis.spines['top'].set_color('white')
            axis.spines['right'].set_color('white')
            axis.spines['left'].set_color('white')
        
        plt.tight_layout()
        ax = None  # Skip the normal axis formatting
        
    elif plot_type == 'mfcc_deltas':
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        fig.patch.set_facecolor('#1e1e1e')
        
        img1 = librosa.display.specshow(features['mfccs'], x_axis='time', ax=ax1, cmap='coolwarm')
        ax1.set_title('MFCCs', color='white', size=14)
        ax1.set_ylabel('MFCC Coefficients', color='white')
        
        img2 = librosa.display.specshow(features['mfcc_delta'], x_axis='time', ax=ax2, cmap='coolwarm')
        ax2.set_title('MFCC Deltas (1st derivative)', color='white', size=14)
        ax2.set_ylabel('MFCC Deltas', color='white')
        
        img3 = librosa.display.specshow(features['mfcc_delta2'], x_axis='time', ax=ax3, cmap='coolwarm')
        ax3.set_title('MFCC Delta-Deltas (2nd derivative)', color='white', size=14)
        ax3.set_ylabel('MFCC Delta-Deltas', color='white')
        ax3.set_xlabel('Time (s)', color='white')
        
        for axis in [ax1, ax2, ax3]:
            axis.tick_params(colors='white')
            axis.spines['bottom'].set_color('white')
            axis.spines['top'].set_color('white')
            axis.spines['right'].set_color('white')
            axis.spines['left'].set_color('white')
        
        plt.tight_layout()
        ax = None  # Skip the normal axis formatting
        
    elif plot_type == 'chroma_comparison':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.patch.set_facecolor('#1e1e1e')
        
        img1 = librosa.display.specshow(features['chroma'], y_axis='chroma', x_axis='time', ax=ax1, cmap='Blues')
        ax1.set_title('Chroma STFT', color='white', size=14)
        
        img2 = librosa.display.specshow(features['chroma_cqt'], y_axis='chroma', x_axis='time', ax=ax2, cmap='Greens')
        ax2.set_title('Chroma CQT (Constant-Q Transform)', color='white', size=14)
        ax2.set_xlabel('Time (s)', color='white')
        
        for axis in [ax1, ax2]:
            axis.tick_params(colors='white')
            axis.spines['bottom'].set_color('white')
            axis.spines['top'].set_color('white')
            axis.spines['right'].set_color('white')
            axis.spines['left'].set_color('white')
        
        plt.tight_layout()
        ax = None  # Skip the normal axis formatting
        
    elif plot_type == 'tonnetz':
        img = librosa.display.specshow(features['tonnetz'], y_axis='tonnetz', x_axis='time', ax=ax, cmap='viridis')
        ax.set_title('Tonnetz (Harmonic Network)', color='white', size=16)
        fig.colorbar(img, ax=ax)
        
    elif plot_type == 'poly_features':
        img = librosa.display.specshow(features['poly_features'], x_axis='time', ax=ax, cmap='plasma')
        ax.set_title('Polynomial Features', color='white', size=16)
        ax.set_ylabel('Polynomial Coefficients', color='white')
        fig.colorbar(img, ax=ax)
    
    if ax is not None:
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', facecolor='#1e1e1e', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_string = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return img_string

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/files')
def get_files():
    files = get_audio_files()
    return jsonify(files)

@app.route('/api/analyze/<filename>')
def analyze_file(filename):
    file_path = os.path.join(AUDIO_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    features = extract_all_features(file_path)
    if 'error' in features:
        return jsonify(features), 500
    
    # Create all visualizations
    visualizations = {}
    plot_types = ['waveform', 'spectrogram', 'mel_spectrogram', 'mfcc', 'chroma', 'spectral_features', 
                  'tempo_beats', 'advanced_spectral', 'onset_strength', 'harmonic_percussive', 
                  'mfcc_deltas', 'chroma_comparison', 'tonnetz', 'poly_features']
    
    for plot_type in plot_types:
        try:
            visualizations[plot_type] = create_visualization(features, plot_type)
        except Exception as e:
            visualizations[plot_type] = f"Error creating {plot_type}: {str(e)}"
    
    return jsonify({
        'filename': filename,
        'tempo': features['tempo'],
        'visualizations': visualizations
    })

@app.route('/data/audio/<filename>')
def serve_audio(filename):
    return send_file(os.path.join(AUDIO_FOLDER, filename))

if __name__ == '__main__':
    # Create audio directory if it doesn't exist
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8080)