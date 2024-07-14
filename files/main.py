import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Function to plot audio channels
def plot_audio_channels(audio_data, sample_rate):
    num_channels = audio_data.shape[1]
    duration = len(audio_data) / sample_rate

    plt.figure(figsize=(10, 6))
    time = np.linspace(0, duration, num=len(audio_data))

    for i in range(num_channels):
        plt.subplot(num_channels, 1, i+1)
        plt.plot(time, audio_data[:, i])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Channel {}'.format(i+1))

    plt.tight_layout()
    plt.show()

# Path to the .wav file
file_path = 'original_audio.wav'

# Read the WAV file
sample_rate, audio_data = wavfile.read(file_path)

# Plot the audio channels
plot_audio_channels(audio_data, sample_rate)

