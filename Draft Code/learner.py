#This is following the guide: https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-3

from preprocessing import *
from mfcc_class_method import *
import scipy as sp
import numpy as np
sp.random.seed(4321) #For reproducibility
np.random.seed(4321) #For reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils

class learner(preprocessing, audio):
    #pass
    def __init__(self,subsfn,mp3,sr,duration):
        preprocessing.__init__(self,subsfn)
        audio.__init__(self,mp3,sr,duration)

# fun = audio('gotS07E01_16k.mp3',sr=16000,duration=3587)
# fun.load_mfcc("full_mfccs.npy")
# print(fun.mfccs)
# print(sp.shape(fun.mfccs))
# subs = preprocessing('gotS07E01.srt')
# subs.load_pb_array("pb_array.csv")

fun2 = learner('gotS07E01.srt','gotS07E01_16k.mp3',sr=16000,duration=3587)
fun2.load_mfcc("full_mfccs.npy")
print(fun2.mfccs)
fun2.load_pb_array("pb_array.csv")
print(fun2.pb_array)


# a = learner('gotS07E01.srt','gotS07E01_16k.mp3',16000,5)
# a.open_subs()
# a.pb_array_init()
# a.pb_array_fill()
# a.mfcc()
# print(a.mfccs)