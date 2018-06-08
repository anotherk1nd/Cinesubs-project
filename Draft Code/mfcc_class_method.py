import librosa as lsa
from python_speech_features import mfcc
import pandas as pd
import scipy as sp
import numpy as np

class MyAppLookupError(LookupError):
    '''raise this when there's a lookup error for my app'''

class audio:
    #Here we implement the audio preprocessing, with the aim of possibly integrating this with the peprocessing method.

    def __init__(self,mp3,sr,duration):
        self.duration = duration
        self.audio, self.sr = lsa.load(mp3, sr=sr,duration=self.duration)
        self.df = pd.DataFrame({'data': self.audio})
        self.mfccs = []


    def mfcc(self):
        try:
            self.mfccs = mfcc(self.df['data'],samplerate=self.sr,nfft=512)
            if np.isnan(np.sum(self.mfccs)): #This returns boolean value describing whether nan is present
                raise ValueError
        except ValueError:
            print("WARNING: nan values present, check these only occur at beginning and end of file, if so continue to replace with float zeros")
            print("Length of mfcc array: ",len(self.mfccs))
            print("nan indices: ",np.where(np.isnan(self.mfccs)))
            var = input("Are nan's only at beginning and end? y/n")
            if var == "y":
                self.mfccs[np.where(np.isnan(self.mfccs))] = float(0.0)
                print("Enclosing nan's replaced with float(0.0)")
            else:
                print("Process terminated")


fun = audio('gotS07E01_16k.mp3',16000,5)
fun.mfcc()
a = fun.mfccs
# print(a[0])
# print(a[1])
# print(np.isnan(np.sum(a))) #This returns boolean value describing whether nan is present
# print(np.where(np.isnan(a))) #This finds the indices where nan values occur and confirms they only appear in 1st and last entries
# print(a[np.where(np.isnan(a))])
# a[np.where(np.isnan(a))] = float(0.0) #This sets nan values to be zero. Need to check that they only occur at the beginning and end first
# print(a)
# a[np.argwhere(np.isnan(a))] = float(0)
# print(a[0])
# print(a[1])
#print(np.argwhere(np.isnan(a)))
#print(np.isnan(np.sum(a)))
#print(sp.where(a==np.nan)) doesnt work
#print(np.isnan()) dw
#print(a[a==['nan']]) dw
#print(type(a))
#cond = np.linalg.cond(a)
#print(cond)
