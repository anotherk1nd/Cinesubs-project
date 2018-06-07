import librosa as lsa
from python_speech_features import mfcc
import pandas as pd

class audio:
    #Here we implement the audio preprocessing, with the aim of possibly integrating this with the peprocessing method.

    def __init__(self,mp3,sr,duration):
        self.duration = duration
        self.audio, self.sr = lsa.load(mp3, sr=sr,duration=self.duration)
        self.df = pd.DataFrame({'data': self.audio})
        self.mfccs = []


    def mfcc(self):
        self.mfccs = mfcc(self.df['data'],samplerate=self.sr,nfft=512)

#fun = audio('gotS07E01_16k.mp3',16000,5)
#fun.mfcc()
#print(fun.mfccs)