import numpy as np
import pyaudio
import matplotlib.pyplot as plt

 
chunk = 512    
rate = 44100    
channels = 1    
format = pyaudio.paInt16    
 
p = pyaudio.PyAudio()
stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

  
fig, (splaxis, fftaxis, bandsaxis) = plt.subplots(1, 3, figsize=(15, 8))
fig.suptitle("Real-Time Audio Visualization with SPL and Frequency Bands")

  
splaxis.axis('off')
bgbar = plt.Rectangle((0.2, 0), 0.6, 1, color="lightgray", alpha=0.5)
splaxis.add_patch(bgbar)
indicatorbar = plt.Rectangle((0.2, 0), 0.6, 0, color="cyan", alpha=0.8)
splaxis.add_patch(indicatorbar)
Cspltext = splaxis.text(0.5, 1.05, "Current SPL: 0 dB", ha="center", va="center", fontsize=14, color="black")
Mspltext = splaxis.text(0.5, 1.1, "Max SPL: 0 dB", ha="center", va="center", fontsize=14, color="black")
splaxis.set_ylim(0, 1)

  
x = np.fft.fftfreq(chunk, d=1/rate)[:chunk // 2]    
bars_fft = fftaxis.bar(x, np.zeros(len(x)), color="blue", width=rate/chunk)
fftaxis.set_xlim(0, 2000)    
fftaxis.set_ylim(0, 50)    
fftaxis.set_xlabel("Frequency (Hz)")
fftaxis.set_ylabel("Magnitude")

  
bandsaxis.set_xlim(0, 3)
bandsaxis.set_ylim(0, 100)
bandsaxis.set_xticks([0.5, 1.5, 2.5])
bandsaxis.set_xticklabels(["Bass", "Mid", "Treble"])
band_bars = bandsaxis.bar([0.5, 1.5, 2.5], [0, 0, 0], color=["blue", "green", "red"], width=0.5)
bandsaxis.set_ylabel("Magnitude (dB)")

maxval = 0


  
def calculate_spl(inputdata):
    rms = np.sqrt(np.mean(inputdata**2))
    spldb = 20 * np.log10(rms + 1e-16)    
    return spldb

  
def get_color(spldb):
    colors = ["blue", "green", "yellowgreen", "yellow", "gold", "orange", "darkorange", "red", "darkred", "purple"]
    index = min(int(spldb // 10), len(colors) - 1)
    return colors[index]

  
def smooth_signal(magfft, windowsize=10):
    return np.convolve(magfft, np.ones(windowsize) / windowsize, mode='same')

  
def visualisation():
    global maxval
    try:
        while True:
              
            data = stream.read(chunk, exception_on_overflow=False)
            inputdata = np.frombuffer(data, dtype=np.int16)

              
            spldb = calculate_spl(inputdata)
            spldb = max(spldb, 0)
            if spldb > maxval:
                maxval = spldb

              
            barhei = min(spldb / 100, 1)
            indicatorbar.set_height(barhei)
            indicatorbar.set_color(get_color(spldb))
            Cspltext.set_text(f"Current SPL: {spldb:.2f} dB")
            Mspltext.set_text(f"Max SPL: {maxval:.2f} dB")

              
            fftVal = np.fft.fft(inputdata)
            magfft = np.abs(fftVal[:chunk // 2])
            finalOutSmooth = smooth_signal(magfft)

              
            for i, b in enumerate(bars_fft):
                b.set_height(finalOutSmooth[i])

              
            bassMag = np.mean(finalOutSmooth[(x >= 20) & (x < 250)])
            midMag = np.mean(finalOutSmooth[(x >= 250) & (x < 4000)])
            trebleMag = np.mean(finalOutSmooth[(x >= 4000) & (x < 20000)])

              
            band_bars[0].set_height(bassMag)
            band_bars[1].set_height(midMag)
            band_bars[2].set_height(trebleMag)

            plt.pause(0.05)
    except KeyboardInterrupt:
        print("Stopping visualizer...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

  
print("Starting real-time audio visualizer with SPL and frequency components...")
visualisation()