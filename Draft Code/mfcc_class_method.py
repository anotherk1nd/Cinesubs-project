import librosa as lsa
from python_speech_features import mfcc
import pandas as pd
import scipy as sp
import numpy as np
import time
import timeit

class audio:
    #Here we implement the audio preprocessing, with the aim of possibly integrating this with the peprocessing method.

    def __init__(self,mp3,sr,duration):
        self.duration = duration
        #t0 = time.time()
        #self.audio, self.sr = lsa.load(mp3, sr=sr,duration=self.duration)
        #t00 = time.time()
        #self.time0 = t00 - t0
        #self.df = pd.DataFrame({'data': self.audio})
        self.mp3 = mp3
        self.sr = sr
        self.mfccs = []
        #self.time = 0.0
        #self.mfcc_fn = ""

    def audio_load(self,sr):
        load_t0 = time.time()
        self.audio,self.sr = lsa.load(self.mp3, sr=sr,duration=self.duration)
        load_t1 = time.time()
        self.df = pd.DataFrame({'data': self.audio})
        self.audio_load_time = load_t1 - load_t0

    def mfcc(self):
        try:
            t1 = time.time()
            self.mfccs = mfcc(self.df['data'],samplerate=self.sr,nfft=512)
            t2 = time.time()
            self.mfcc_time = t2 - t1

            if np.isnan(np.sum(self.mfccs)): #This returns boolean value describing whether nan is present
                raise ValueError
        except ValueError:
            print("WARNING: nan values present, check these only occur at beginning and end of file, if so continue to replace with float zeros")
            print("Length of mfcc array: ",len(self.mfccs))
            print("nan indices: ",np.where(np.isnan(self.mfccs))) #need to clean up the output to only print row numbers i think but may miss non nan entries if i do
            var = input("Are nan's only at beginning and end? y/n")
            if var == "y":
                self.mfccs[np.where(np.isnan(self.mfccs))] = float(0.0)
                print("Enclosing nan's replaced with float(0.0)")
            else:
                print("Process terminated")

    def save_mfcc(self,save_fn):
        #sp.savetxt(fn,self.mfccs, delimiter=',')
        sp.save(save_fn,self.mfccs) #saves in binary format with extension npy

    def load_mfcc(self,load_fn):
        #self.mfccs = sp.loadtxt(load_fn,delimiter=',')
        #self.mfccs = pd.read_csv(load_fn,header=None).values #much faster than sp.loadtxt!
        self.mfccs = sp.load(load_fn)

#fun = audio('gotS07E01_16k.mp3',sr=16000,duration=3587)
#time_audioload = timeit.timeit(lambda:fun.audio_load(sr=fun.sr),number=1,globals=globals())
# number of runs, using global variables runs in the global environment and so actually assigns the variables during run time,
# rather than running in a kind of sandbox environment (local) where no changes to the global environment are made.
# Need a lambda function since it needs to time a function, but without the lambda we are really just passing an object of the class fun
#fun.mfcc()
#time_mfccsave = timeit.timeit(lambda:fun.save_mfcc("full_mfccs.npy"),number=1,globals=globals())
#time_mfccload = timeit.timeit(lambda:fun.load_mfcc("full_mfccs.npy"),number=1,globals=globals())
#print("Audio load time",time_audioload)
#print("Save time",time_mfccsave)
#print("MFCC load time",time_mfccload)

#fun.mfcc()
#a = fun.mfccs
#print(a)
#fun.save_mfcc()
#print(fun.time0,fun.time)
#time1 = time.time()
#fun.load_mfcc('mfcc.csv')
#time11 = time.time()
#print(fun.mfccs)
#print("load mp3 time", fun.time0)
#print("mfcc time", fun.time)
#print("load mfcc time",time11-time1)
#b = fun.mfccs
#print(sp.array_equal(a,b)) #returns true
# print(a)
# print(b)
#print(timeit.timeit(fun.mfcc))
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
