import matplotlib.pyplot as plt
import numpy as np
import librosa
from midiutil import MIDIFile

def analyzeAudio(audioPath):
  audioVals = {}
  y, sr = librosa.load(audioPath)
  audioVals['f0'], voiced_flag, audioVals['voiced_probs'] = librosa.pyin(y,
                                              fmin=librosa.note_to_hz('C2'),
                                              fmax=librosa.note_to_hz('C7'))
  audioVals['f0mean'] = np.nanmean(audioVals['f0'])
  audioVals['rms'] = librosa.feature.rms (y=y).transpose ()

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
    estimated = librosa.frames_to_time(onset_frames, sr=sr)
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

audioPath = 'files/Vox.wav'

features, sr = analyzeAudio(audioPath)
plotAudioVals(features,audioPath,'pYIN fundamental frequency estimation','f0')
clean_f0 = np.nan_to_num (features['f0'], nan=features['f0mean'])

print ("MIDI pitches: ")
pitches = librosa.hz_to_midi (clean_f0)

onsets = getOnsetsWrapper (audioPath)
print ("onsets in sec.:")
print (onsets)

# hop size is 512
locations = onsets*sr/512

notes = pitches[locations.astype(int)]  
print ("MIDI pitches and corresponding notes on onsets: ")
print (notes)
print (librosa.midi_to_note (notes))

volumes = features['rms'][locations.astype (int)]

track    = 0
channel  = 0
time     = 0
tempo    = 60

MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                      # automatically)
MyMIDI.addTempo(track, time, tempo)

midi_old = 0
onset_old = -1000
timegate = .1 # seconds
for i, pitch in enumerate(notes.astype(int)):
    print (onsets[i] - onset_old)
    if onsets[i] - onset_old > timegate:
      print ("\ttaken")
      midi_time = onsets[i]
      MyMIDI.addNote(track, channel, pitch, midi_time, midi_time - midi_old, 100)
      midi_old = midi_time
    
    onset_old = onsets[i]

with open("transcription.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)


print ("to be continued...")
