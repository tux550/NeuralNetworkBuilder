import librosa as lb
import matplotlib.pyplot as plt

def wav2fv(wav_path, fv_size=128, visualize = False):
  # Load wav
  signal, sample_rate = lb.load(wav_path)

  # Wav to feature vector
  fv = lb.feature.mfcc(y=signal, n_mfcc=fv_size, sr=sample_rate)
  
  # Visualize
  if visualize:
    plt.figure(figsize=(25,10))
    lb.display.specshow(fv,x_axis="time",sr=sample_rate)
    plt.colorbar(format="%+2f")
    plt.show()

  # Return
  return fv

if __name__ == "__main__":
    fv = wav2fv("../raw_dataset/sound_dataset/Song/Actor_01/03-02-01-01-01-01-01.wav",13, visualize=True)
    print(fv.shape)
    fv = wav2fv("../raw_dataset/sound_dataset/Speech/Actor_23/03-01-03-02-01-02-23.wav",13, visualize=True)
    print(fv.shape)
