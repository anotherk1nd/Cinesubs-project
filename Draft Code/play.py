from python_speech_features import mfcc
import wave #This can only handle 2 channels and is less complete that soundfile
import soundfile as sf
import csv
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()# set decides the aesthetic parameters. Don't know if i need this.
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
#from preprocessing import preprocessing
plt.close('all')
import librosa as lsa # we need to have ffmpeg installed on the lappytop to work, not within python
import scipy as sp
sp.set_printoptions(threshold=sp.nan)
from pydub.utils import mediainfo
from pydub import AudioSegment
import pandas as pd
from pandas.util.testing import assert_frame_equal
pd.set_option('display.max_columns', None)
import time

# data,rate = sf.read('gotS07E01.wav')
# print(data,rate)
#audionat = lsa.core.load('outputaudio.mp3',sr=None)

#Convert wav to mp3
#AudioSegment.from_wav('gotS07E01.wav').export('gotS07E01.mp3',format="mp3")

#info = mediainfo('gotS07E01_16k.mp3')
#sr = info['sample_rate']
#sr = int(sr)
#print(type(sr))
audio,sr = lsa.load('gotS07E01_16k.mp3', sr=16000) #sr=None preserves the native sample rate, ITS SUPPOSED TO BUT PRODUCES DIFF RESULTS, CHECK NOTES DOC
print(sp.shape(audio))
#print(audio,sr)
#print(sp.shape(audio))
#print(audio)
#print(audio[2000:2500])
#array = [audio]
#print(array[-1])
df = pd.DataFrame({'data':audio})
#print(df)
#print(df.iloc[0,200:500])
t0 = time.time()
features = mfcc(df['data'],samplerate=16000,nfft=512) #nfft=512 for 16000, check notes for info on this.
t1 = time.time()
print(features) #for some reason we get a nan for first and final entry
time = t1-t0
print(time)
#nan=sp.isnan(np.sum(features))
#print(sp.where(features=="nan"))
#nan_rows = df[df['data'].isnull()]

#df.to_csv("got_16k.csv",sep=",")


#shitty cunt of csv changed the shape of the dataframe, fuck that shit

# df1 = pd.read_csv('got_16k.csv',header=None)
# print(type(df))
# print(type(df1))
# df.drop(df.index[0], inplace=True)
# print(type(df1))
#
# #print(df1)
# assert_frame_equal(df,df1)
# df += 1e-6
# print(df)
# print(df.iloc[0,-13]) #NaN placeholder
#print(df.iloc[0,:])
#print(df.iloc[0,80000]) #indexed from 1 with iloc, column 0 has just 0 placeholder, this is a ROW vector (well matrix with dimension 1 in terms of number of rows.
#features = mfcc(df.iloc[0,1:],samplerate=16000,nfft=1200) #A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature
#print(features)
#with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
#     print(df)
#print(df.iloc[[1,1]])
#print(sp.array_equal(audionat,audio)) #apparently the 2 are different, even though with sr=none should detect the sample rate and use accordinglz
#print(sp.shape(audio))
#print(type(audio))
#print(sr)
#np.savetxt("/Users/joshfenech/Documents/Linux Documents Backup May/Documents/Shared Documents/MLDM/Project/Draft Code/mp3_array.csv",audio,delimiter=',',fmt="%d")
# with open('mp3_array.csv') as csvDataFile:
#     csvReader = csv.reader(csvDataFile)

#

# subs = preprocessing('gotS07E01.srt')
#subs.open_subs()

# with open('pb_array.csv') as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     index = []
#     window_time = []
#     pb = []
#     for row in readCSV:
#         ind = row[0]
#         w = row[1]
#         p = row[2]
#         index.append(ind)
#         window_time.append(w)
#         pb.append(p)

#print(index,window_time,pb)

#print(data[:,0]) #First channel^
#print(rate)
#print(sp.shape(data)) #(540672,6) I think this is the number of samples (48000*11secs = 528000, so must be slightly longer than 11s)and 6 channels
#print(help(mfcc))
#features = mfcc(data[1,0],samplerate=48000,nfft=1200) #A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector. #NFFT size should be equal or greater to frame lengthhttps://github.com/jameslyons/python_speech_features/issues3
#print(features)
#print(sp.shape(features)) #(1125,13) ie ~11s/0.01s windows, 13 columns for different cepstrums: numcep – the number of cepstrum to return, default 13, from http://python-speech-features.readthedocs.io/en/latest/
#print(features[1124])
#index = sp.arange(len(features))
#index = index.reshape(len(features),1) # we need to reshape the index row vector into a column vector before we can append with hstack https://scipython.com/book/chapter-6-numpy/examples/vstack-and-hstack/
#featuresI = sp.hstack((index,features)) #FIRST COLUMN CONTAINS INDEX WHICH WE WILL BE TRYING TO PREDICT
#features = sp.append(index,features,axis=1)
#print(sp.shape(index))
#print(sp.shape(featuresI))
#print(featuresI)
#X = featuresI[:,1:]
#y = featuresI[:,0] #THIS IS WRONG, WE WILL NEVER BE ABLE TO PREDICT THE INDEX SINCE IT HAS ABSOLUTELY NO CORRELATION WITH THE FEATURES


#X_train, X_test, y_train, y_test = train_test_split(X,y)

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X_train, y_train)
# print(clf.predict(X_test))
# print(y_test)
#print(clf.score(X_test,y_test))

#plt.plot(features)
#plt.legend()
#ax = sns.heatmap(features)
#plt.show()

