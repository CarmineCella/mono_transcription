import matplotlib.pyplot as plt
import numpy as np
import librosa

def analyzeAudio(audioPath):
  audioVals = {}
  y, sr = librosa.load(audioPath)
  audioVals['f0'], voiced_flag, audioVals['voiced_probs'] = librosa.pyin(y,
                                              fmin=librosa.note_to_hz('C2'),
                                              fmax=librosa.note_to_hz('C7'))
  audioVals['f0mean'] = np.nanmean(audioVals['f0'])

  return audioVals

def plotAudioVals(audioVals,audioPath,plotTitle,dataName):
  y, sr = librosa.load(audioPath)
  times = librosa.times_like(audioVals[dataName])
  D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
  fig, ax = plt.subplots()
  img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
  ax.set(title=plotTitle)
  fig.colorbar(img, ax=ax, format="%+2.f dB")
  ax.plot(times, audioVals[dataName], label=dataName, color='cyan', linewidth=3)
  ax.legend(loc='upper right')
  plt.show()
  return

audioPath = 'files/trumpet.wav'

features = analyzeAudio(audioPath)

plotAudioVals(features,audioPath,'pYIN fundamental frequency estimation','f0')

clean_f0 = np.nan_to_num (features['f0'], nan=features['f0mean'])
print (clean_f0)

