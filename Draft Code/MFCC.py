import scipy as sp
from python_speech_features import mfcc
import librosa as lsa
import soundfile as sf

class sound_preprocessing:
    #Here we implement the MFCC extraction as a class, maybe to be fused with preprocessing later
    def __init__(self, sound_fn):
        #self.data = []
        #self.rate = 0.0
        self.data, self.rate = sf.read(sound_fn)
        #self.rate = sf.read(sound_fn)
        self.data = self.data[0:10,0] #alt+5 alt+6 for square brackets, 0 for first channel, we take small sample for testing
        self.nfft = 0
        self.features = []

    def mfcc(self,nfft):
        #nfft is the number of
        #A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
        # #NFFT size should be equal or greater to frame lengthhttps://github.com/jameslyons/python_speech_features/issues3
        self.nfft = nfft
        self.features = mfcc(self.data,self.rate, self.nfft)

snd = sound_preprocessing('gotS07E01.wav')
features = snd.mfcc(nfft=1200)
# When we run this on the whole sound file, it uses so much memory that it is interrupted automatically, exit code 137 ended with sigkill,
# will have to find a way to optimise this.
#print(snd.data, snd.rate)
print(features)


# features = mfcc(data[1,0],samplerate=48000,nfft=1200) #A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector. #NFFT size should be equal or greater to frame lengthhttps://github.com/jameslyons/python_speech_features/issues/33
# print(features)
# print(sp.shape(features)) #(1125,13) ie ~11s/0.01s windows, 13 columns for different cepstrums: numcep – the number of cepstrum to return, default 13, from http://python-speech-features.readthedocs.io/en/latest/
# print(features[1124])
# index = sp.arange(len(features))
# index = index.reshape(len(features),1) # we need to reshape the index row vector into a column vector before we can append with hstack https://scipython.com/book/chapter-6-numpy/examples/vstack-and-hstack/
# featuresI = sp.hstack((index,features)) #FIRST COLUMN CONTAINS INDEX WHICH WE WILL BE TRYING TO PREDICT
# features = sp.append(index,features,axis=1)
# print(sp.shape(index))
# print(sp.shape(featuresI))
# print(featuresI)