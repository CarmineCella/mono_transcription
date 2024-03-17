import matplotlib.pyplot as plt
import numpy as np
import librosa

def analyzeAudio(audioPath):
  audioVals = {}
  y, sr = librosa.load(audioPath)
  audioVals['specCent'] =librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512, freq=None).transpose ()

  audioVals['f0'], voiced_flag, audioVals['voiced_probs'] = librosa.pyin(y,
                                              fmin=librosa.note_to_hz('C2'),
                                              fmax=librosa.note_to_hz('C7'))
  audioVals['f0mean'] = np.nanmean(audioVals['f0'])

  return audioVals, sr


def getOnsetsEnv(sig,sr=44100):
    o_env = librosa.onset.onset_strength(y=sig, sr=sr)
    times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

    D = np.abs(librosa.stft(sig,n_fft=4096))
    plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), hop_length=512, x_axis='time', y_axis='log')
    plt.title('Estimated from Onset Envelope')
    plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(times, o_env, label='Onset strength')
    plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    plt.axis('tight')
    plt.legend(frameon=True, framealpha=0.75)
    return onset_frames

def getOnsetsWrapper(filename):
    sig , sr = librosa.load(filename,mono=True,sr=None)
    
    # estimate onsets from an onset envelope
    onset_frames = getOnsetsEnv(sig,sr)
    print('Estimated onsets:')
    estimated = librosa.frames_to_time(onset_frames, sr=sr)
    print(estimated)
    return estimated

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

features, sr = analyzeAudio(audioPath)
plotAudioVals(features,audioPath,'pYIN fundamental frequency estimation','f0')

onsets = getOnsetsWrapper (audioPath)
print (onsets)

print ("to be continued...")



