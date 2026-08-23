# 📡 DSP Toolbox

A desktop application for Digital Signal Processing, built with Python and Tkinter as part of a DSP course. It provides a single GUI for generating, visualizing, and processing discrete signals, with built-in test cases to validate each operation against expected results.

## ✨ Features

- **Signal generation** — create sine/cosine waves with configurable amplitude, frequency, sampling frequency, and phase shift (with sampling theorem validation)
- **File I/O** — load signals from text files, save results (filter coefficients, DCT coefficients, resampled signals) back to disk
- **Arithmetic operations** — Add, Subtract, Multiply, Square, Normalize, Accumulate
- **Quantization** — configurable by number of bits or number of levels, with quantization error reporting
- **Frequency domain** — DFT, IDFT, DC component removal (time and frequency domain)
- **Time domain manipulation** — Shift, Fold, Convolution, Cross-correlation, Moving average, Sharpening (1st/2nd derivative)
- **DCT** — Discrete Cosine Transform with configurable number of saved coefficients
- **FIR filtering** — Low-pass, High-pass, Band-pass, and Band-stop filters using windowing (Rectangular, Hanning, Hamming, Blackman) based on stop-band attenuation
- **Resampling** — combined decimation (M) and interpolation (L) using the FIR filter
- **Built-in test validation** — each operation compares its output against an expected results file and reports pass/fail via message boxes

## 🗂️ Project Structure

```
├── Task 1/ … Task 7/    # Weekly task deliverables and test files
├── DSP.py                 # Main application (GUI + all signal processing logic)
├── icon.png                 # Window icon
├── img.png                    # GUI background image
```

## 🛠️ Tech Stack

- Python
- Tkinter (GUI)
- NumPy / SciPy
- Matplotlib (signal plotting)
- Pillow (image handling)

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib pillow
   ```
3. Run the app:
   ```bash
   python DSP.py
   ```

## 👥 Team

- Shreef Mahmoud
- Saif Hossam

## 📄 License

Academic project — for coursework purposes.
