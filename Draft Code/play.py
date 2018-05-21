from python_speech_features import mfcc
import wave #This can only handle 2 channels and is less complete that soundfile
import soundfile as sf
import csv
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()# set decides the aesthetic parameters. Don't know if i need this.
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
plt.close('all')

data,rate = sf.read('gotS07E01.wav',)
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

#print(data[:,0]) #First channel
#print(rate)
#print(sp.shape(data)) #(540672,6) I think this is the number of samples (48000*11secs = 528000, so must be slightly longer than 11s)and 6 channels
#print(help(mfcc))
features = mfcc(data[1,0],samplerate=48000,nfft=1200) #A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector. #NFFT size should be equal or greater to frame lengthhttps://github.com/jameslyons/python_speech_features/issues/33
print(features)
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

