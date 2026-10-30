import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import io

st.set_page_config(page_title="Signal Processing & Noise Removal", page_icon="📡", layout="wide")

st.title("📡 Signal Processing and Noise Removal Using Python")
st.caption("A beginner-friendly ECE project: create or upload a signal, add noise, filter it, and compare results.")

with st.sidebar:
    st.header("Signal settings")
    source = st.radio("Signal source", ["Generate sample signal", "Upload WAV audio"])
    fs = st.slider("Sampling frequency (Hz)", 500, 10000, 2000, 100) if source == "Generate sample signal" else 2000
    duration = st.slider("Duration (seconds)", 1, 10, 3) if source == "Generate sample signal" else 3
    noise_level = st.slider("Noise strength", 0.0, 1.5, 0.35, 0.05)
    st.header("Filter settings")
    filter_type = st.selectbox("Filter type", ["Low-pass", "High-pass", "Band-pass"])
    if filter_type == "Low-pass":
        cutoff = st.slider("Cutoff frequency (Hz)", 50, max(100, fs // 2 - 50), min(400, fs // 2 - 50), 10)
        low_cut, high_cut = None, cutoff
    elif filter_type == "High-pass":
        cutoff = st.slider("Cutoff frequency (Hz)", 20, max(100, fs // 2 - 50), min(150, fs // 2 - 50), 10)
        low_cut, high_cut = cutoff, None
    else:
        low_cut = st.slider("Lower cutoff (Hz)", 20, max(100, fs // 2 - 100), min(100, fs // 2 - 100), 10)
        high_cut = st.slider("Upper cutoff (Hz)", low_cut + 10, max(low_cut + 20, fs // 2 - 10), min(max(low_cut + 20, 500), fs // 2 - 10), 10)
    order = st.slider("Filter order", 1, 8, 4)

def make_sample(fs, duration):
    t = np.arange(int(fs * duration)) / fs
    clean = 0.7 * np.sin(2 * np.pi * 50 * t) + 0.35 * np.sin(2 * np.pi * 120 * t)
    return t, clean

def normalize_audio(data):
    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        data = data.mean(axis=1)
    peak = np.max(np.abs(data)) if data.size else 0
    return data / peak if peak else data

if source == "Generate sample signal":
    t, clean = make_sample(fs, duration)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, noise_level, len(clean))
    noisy = clean + noise
else:
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded is None:
        st.info("Upload a WAV file to begin. You can switch to 'Generate sample signal' to explore the project.")
        st.stop()
    sample_rate, raw = wavfile.read(io.BytesIO(uploaded.getvalue()))
    fs = int(sample_rate)
    clean = normalize_audio(raw)
    t = np.arange(len(clean)) / fs
    rng = np.random.default_rng(7)
    noisy = clean + rng.normal(0, noise_level, len(clean))

nyquist = fs / 2
# Ensure cutoff values remain valid if uploaded audio has a low sample rate.
if filter_type == "Low-pass":
    wn, btype = high_cut / nyquist, "lowpass"
elif filter_type == "High-pass":
    wn, btype = low_cut / nyquist, "highpass"
else:
    wn, btype = [low_cut / nyquist, high_cut / nyquist], "bandpass"

if isinstance(wn, list):
    wn = [max(0.001, min(x, 0.999)) for x in wn]
else:
    wn = max(0.001, min(wn, 0.999))

sos = signal.butter(order, wn, btype=btype, output="sos")
try:
    filtered = signal.sosfiltfilt(sos, noisy)
except ValueError:
    filtered = signal.sosfilt(sos, noisy)

def snr_db(reference, observed):
    noise = observed - reference
    p_signal = np.mean(reference ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise <= 1e-15:
        return float("inf")
    return 10 * np.log10(max(p_signal, 1e-15) / p_noise)

def plot_time(ax, x, y, title, color=None):
    ax.plot(x, y, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

st.subheader("Signal comparison")
c1, c2, c3 = st.columns(3)
with c1:
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_time(ax, t, clean, "Original / reference signal")
    st.pyplot(fig)
    plt.close(fig)
with c2:
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_time(ax, t, noisy, "Signal with added noise")
    st.pyplot(fig)
    plt.close(fig)
with c3:
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_time(ax, t, filtered, "Filtered signal")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Frequency spectrum")
def spectrum(x, fs):
    f, p = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
    return f, 10 * np.log10(p + 1e-12)

fig, ax = plt.subplots(figsize=(10, 4))
for data, label in [(clean, "Reference"), (noisy, "Noisy"), (filtered, "Filtered")]:
    f, p = spectrum(data, fs)
    ax.plot(f, p, label=label, linewidth=1)
ax.set_title("Power spectral density (Welch method)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power / Hz (dB)")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)
plt.close(fig)

metric1, metric2 = st.columns(2)
metric1.metric("SNR before filtering", f"{snr_db(clean, noisy):.2f} dB")
metric2.metric("SNR after filtering", f"{snr_db(clean, filtered):.2f} dB")
st.caption("SNR metrics use the reference signal, available for the generated demo and for uploaded WAV files treated as the reference before noise is added.")

wav_buffer = io.BytesIO()
audio_int16 = np.int16(np.clip(filtered / (np.max(np.abs(filtered)) + 1e-12), -1, 1) * 32767)
wavfile.write(wav_buffer, fs, audio_int16)
st.download_button("Download filtered signal (WAV)", data=wav_buffer.getvalue(), file_name="filtered_signal.wav", mime="audio/wav")

with st.expander("How the project works"):
    st.markdown("""
    1. **Signal generation / input:** Create a composite signal or load a WAV file.
    2. **Noise addition:** Add Gaussian noise to represent a noisy measurement or channel.
    3. **Filter design:** Design a Butterworth low-pass, high-pass, or band-pass filter.
    4. **Filtering:** Apply the filter to suppress unwanted frequency components.
    5. **Evaluation:** Compare time plots, frequency spectra, and signal-to-noise ratio (SNR).

    A Butterworth filter has a smooth frequency response. `sosfiltfilt` filters forward and backward to reduce phase distortion for offline processing.
    """)
