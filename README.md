# Signal Processing and Noise Removal Using Python

A simple ECE project that demonstrates noise addition and filtering using Python. The Streamlit interface lets you generate a sample signal or upload a WAV file, choose a Butterworth filter, visualize time/frequency plots, compare SNR, and download the filtered WAV.

## Features
- Sample composite signal (50 Hz + 120 Hz)
- Add Gaussian noise with adjustable strength
- Low-pass, high-pass, and band-pass Butterworth filters
- Time-domain comparison: reference, noisy, filtered
- Frequency-domain comparison using Welch power spectral density
- SNR before/after filtering
- Download filtered output as WAV

## Tech stack
Python, NumPy, SciPy, Matplotlib, Streamlit

## Run locally
Recommended: Python 3.10 or newer.

```bash
python -m venv .venv
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

Install packages:
```bash
pip install -r requirements.txt
```

Run:
```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Project structure
```text
signal_noise_removal_project/
├── app.py
├── requirements.txt
└── README.md
```

## Important note
For a generated demo, the original clean signal is known, so SNR is calculated meaningfully. For an uploaded WAV, the uploaded audio is treated as the reference and noise is synthetically added. The project is an educational offline signal-processing demonstration, not a certified noise-cancellation device.
