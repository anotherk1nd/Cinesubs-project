from preprocessing import *
from mfcc_class_method import *

class learner(preprocessing, audio):
    pass
    # def __init__(self,subsfn,mp3,sr,duration):
    #     learner.__init__(self,subsfn)
    #     learner.__init__(self, mp3,sr,duration)


a = learner('gotS07E01.srt')
a.open_subs()
a.pb_array_init()
a.pb_array_fill()
print(a.pb_array)