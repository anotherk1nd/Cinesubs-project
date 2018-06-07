from preprocessing import *
from mfcc_class_method import *

class learner(preprocessing, audio):
    def __init__(self,):
        # We don't need to define the object variables of other classes here, only new ones I think
         learner.
    #     learner.__init__(self, mp3,sr,duration)


a = learner('gotS07E01.srt')
a.open_subs()
a.pb_array_init()
a.pb_array_fill()
print(a.pb_array)