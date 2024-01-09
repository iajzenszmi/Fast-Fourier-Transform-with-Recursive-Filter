import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Parameters
sampling_rate = 1000  # Hz
duration = 1  # second
frequencies = [5, 50, 100]  # Hz

# Time array
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate a composite signal
signal = np.sum([np.sin(2 * np.pi * f * t) for f in frequencies], axis=0)

# Recursive filter (simple first-order low-pass filter)
class RecursiveFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y_prev = 0

    def apply(self, x):
        y = self.alpha * x + (1 - self.alpha) * self.y_prev
        self.y_prev = y
        return y

# Create and apply the filter
alpha = 0.1  # Filter coefficient
filter = RecursiveFilter(alpha)
filtered_signal = np.array([filter.apply(x) for x in signal])

# Fourier analysis
fft_original = fft(signal)
fft_filtered = fft(filtered_signal)
freq = np.fft.fftfreq(len(t), d=1/sampling_rate)

# Plotting
plt.figure(figsize=(12, 8))

# Original Signal
plt.subplot(2, 2, 1)
plt.plot(t, signal)
plt.title("Original Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# Filtered Signal
plt.subplot(2, 2, 2)
plt.plot(t, filtered_signal)
plt.title("Filtered Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# FFT Original Signal
plt.subplot(2, 2, 3)
plt.plot(freq, np.abs(fft_original))
plt.title("FFT of Original Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.xlim([0, 150])

# FFT Filtered Signal
plt.subplot(2, 2, 4)
plt.plot(freq, np.abs(fft_filtered))
plt.title("FFT of Filtered Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.xlim([0, 150])

plt.tight_layout()
plt.show()
