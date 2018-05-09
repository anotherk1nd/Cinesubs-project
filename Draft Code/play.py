from python_speech_features import mfcc
import wave #This can only handle 2 channels and is less complete that soundfile
import soundfile as sf
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()# set decides the aesthetic parameters. Don't know if i need this.
plt.close('all')

data,rate = sf.read('outputaudio.wav')
print(data[:,0]) #First channel
print(rate)
print(sp.shape(data)) #(540672,6) I think this is the number of samples (48000*11secs = 528000, so must be slightly longer than 11s)and 6 channels
print(help(mfcc))
features = mfcc(data[:,0],samplerate=48000,nfft=1200) #A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector. #NFFT size should be equal or greater to frame lengthhttps://github.com/jameslyons/python_speech_features/issues/33
print(sp.shape(features)) #(1125,13) ie ~11s/0.01s windows, 13 columns for different cepstrums: numcep – the number of cepstrum to return, default 13, from http://python-speech-features.readthedocs.io/en/latest/
#plt.plot(features)
#plt.legend()
#ax = sns.heatmap(features)
#plt.show()

