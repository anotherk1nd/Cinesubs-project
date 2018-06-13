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
from sklearn.model_selection import train_test_split

class learner(preprocessing, audio):
    #pass
    def __init__(self,subsfn,mp3,sr,duration=None):
        preprocessing.__init__(self,subsfn)
        audio.__init__(self,mp3,sr,duration)


    def len_check(self):
        # self.tconv(-1)
        # self.duration_sub = self.times[1] #gives time when last sub disappears off screen
        # self.duration_audio = lsa.get_duration(self.audio,sr=self.sr) #must have loaded mp3 first
        # if self.duration_sub > self.duration_audio:
        if len(self.mfccs) > len(self.pb_array):
            print("mfcc array longer than pb array")
            self.mfccs = self.mfccs[:len(self.pb_array),:]
        elif len(self.mfccs) < len(self.pb_array):
            print("pb_array longer than mfcc array")
            self.pb_array = self.pb_array[:len(self.mfccs),:]
        else:
            print("array lengths match")


    def train_test_splitting(self,test_size=0.33):
        X = self.mfccs
        y = self.pb_array
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = self.test_size, random_state = 42)
        print(X_train.shape)

    #def build_net(self):

# fun = audio('gotS07E01_16k.mp3',sr=16000,duration=3587)
# fun.load_mfcc("full_mfccs.npy")
# print(fun.mfccs)
# print(sp.shape(fun.mfccs))
# subs = preprocessing('gotS07E01.srt')
# subs.load_pb_array("pb_array.csv")

fun2 = learner('gotS07E01.srt','gotS07E01_16k.mp3',sr=16000,duration=None) #using duration=None should give full file. Pb array is filled based on last sub
#fun2.open_subs()
#fun2.audio_load()
#print(fun2.duration_true)
#fun2.audio_load(sr=fun2.sr)
#fun2.mfcc()
#fun2.save_mfcc("full_mfccs.npy")
fun2.load_mfcc("full_mfccs.npy")
fun2.load_pb_array("pb_array.npy")
fun2.len_check()

#print(len(fun2.mfccs),len(fun2.pb_array))
#print(fun2.mfccs)
#fun2.load_pb_array("pb_array.npy")
#print(sp.shape(fun2.mfccs),sp.shape(fun2.pb_array))
#fun2.train_test_splitting()



# a = learner('gotS07E01.srt','gotS07E01_16k.mp3',16000,5)
# a.open_subs()
# a.pb_array_init()
# a.pb_array_fill()
# a.mfcc()
# print(a.mfccs)