from preprocessing import *
from mfcc_class_method import *
import scipy as sp
import numpy as np
sp.random.seed(4321) #For reproducibility
np.random.seed(4321) #For reproducibility

class learner(preprocessing, audio):
    #pass
    def __init__(self,subsfn,mp3,sr,duration):
        preprocessing.__init__(self,subsfn)
        audio.__init__(self,mp3,sr,duration)




# a = learner('gotS07E01.srt','gotS07E01_16k.mp3',16000,5)
# a.open_subs()
# a.pb_array_init()
# a.pb_array_fill()
# a.mfcc()
# print(a.mfccs)