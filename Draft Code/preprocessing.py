import pysrt as srt
import scipy as sp
from datetime import datetime, date
import pandas as pd
import math as m
import timeit

class preprocessing:
    '''
    Here we define the different methods for all the preprocessing required to produce the inputs and outputs for the learning process
    Required modules: pysrt,scipy,datetime.datetime,datetime.date I HAVE IMPORTED WITH ABBREVIATIONS IE SCIPY AS SP, THIS WOULD REQUIRE SPECIFIC IMPORT ALIASES AND SHOULD BE CORRECTED!!
    '''

    #We define the class variables that are shared across the functions. However we need to use a function to extract these variables, and hence maybe need to use
    #the concept of inheritance, where a class is declared subsequent to a previous class and takes the outputs of the previous class as initialisers of new class (or somethin like that!)
    # filename is a string, requires full file path of must be run from correct directory

    #def __init__(self,subsfn,pb_array,subs_times,count,counter): # we take the variables we want to initialise, then assign these as class attributes with the self.variable statements
    def __init__(self, subsfn):
        self.subsfn = subsfn
        self.pb_array = [[],[]]
        self.subs_times = []
        #self.count = 0 count is a locally used variable within the pb_array_init function and hence need not be initialised
        self.counter = 0 # counter is implemented by pb_array_fill when applying the tconv function and so needs to be initialised as it is shared between functions
        self.subs = 0
        self.times = []

    def open_subs(self):
        # Opens the srt file,
        #self.subs = (srt.open('%s') % (self.subsfn))
        self.subs = (srt.open(self.subsfn))

    def tconv(self,counter):
        # We define a function that will take each subs entries which uses datatime objects that can't be handled easily, and convert to seconds from start, returning
        # array [time_start,time_end].
        # We define the counter externally so that it can be applied in the while true loop
        # I THINK WE SHOULD MAKE THIS WORK INDEPENDENTLY (WITHOUT EXTERNAL COUNTER) SO THAT IT CAN BE APPLIED DIRECTLY
        #print(self.subs)
        fst_start = self.subs[counter].start.to_time()
        time_start = datetime.combine(date.min, fst_start) - datetime.min
        time_start = time_start.total_seconds()
        fst_end = self.subs[counter].end.to_time()
        time_end = datetime.combine(date.min, fst_end) - datetime.min
        time_end = time_end.total_seconds()
        times = [time_start,time_end]
        self.times = times
        #print(self.times)

    #def myfunction(self,subfn):

    def pb_array_init(self):
        # initialise the probability array to the correct length
        self.tconv(-1) # This assigns the times of the last sub to self.times
        last_sub = self.times[1] #This gives time that last sub is removed from screen
        #print(last_sub)
        #print(m.ceil(last_sub))
        len = m.ceil(last_sub)
        #print(type(len))
        self.pb_array = sp.zeros((len*100,3))
        count = sp.arange(len*100)
        self.pb_array[:, 0] = count[:]  # an index for each sample starting at 0
        self.pb_array[:, 1] = self.pb_array[:,0] * 0.01  # 0.01 is the size of the MFCC windows used in play.py, this is the starting point for each window

    def pb_array_fill(self):
        i = 0
        j = 0  # i is counter over subs array, j is counter over sample array (stored in pb_array)
        while True:
            if i + 1 > len(self.pb_array) - 1:  # matrices are indexed from 0 but the length returns the 'normal' length i.e len([1]) returns 1 not 0
                print('pb_array length exceeded')
                print(i)
                break
            if j > len(self.subs) - 1:
                print('substimes array length exceeded')
                print(j)
                break
            self.tconv(j)  # substimes are stored in self.times within this class function
            #print(self.times)
            # print(subs_times)
            # print(pb_array[i,1])
            # print(pb_array[i + 1, 1])
            # end = subs[j].end.to_time()
            #print(self.)
            if self.pb_array[i, 1] >= self.times[0]:
                if self.pb_array[i + 1, 1] < self.times[1]:  # we only stored window start times in pb_array, so need to compare to i+1
                    print('Within subs window')
                    self.pb_array[i, 2] = 1
                    #print(self.pb_array[i,2])
                    i = i + 1
                elif self.pb_array[i + 1, 1] >= self.times[1]:
                    print('Subs window exceeded')
                    j = j + 1
            else:
                print('Next segment')
                #print(i)
                i = i + 1

    def save_pb_array(self,fn):
        #sp.savetxt('pb_array',self.pb_array)
        sp.save(fn,self.pb_array)

    def load_pb_array(self,fn):
        #self.pb_array = sp.loadtxt(fn)
        #self.pb_array = pd.read_csv(fn, header=None).values  # much faster than sp.loadtxt!
        self.pb_array=sp.load(fn) #Must be .npy!

#sp.set_printoptions(threshold=sp.nan)
#subs = preprocessing('gotS07E01.srt')
#subs.open_subs()
# #subs.tconv(-1)
#
# #print(subs.times[1])
#subs.pb_array_init()
#subs.pb_array_fill()
#print(subs.pb_array)
#time_pbfill = timeit.timeit(lambda:subs.pb_array_fill(),number=1,globals=globals())
#print(time_pbfill)
#print(subs.pb_array)

# #We can then pass the object to a variable if we liked
# # pb_array = subs.pb_array
# # print(pb_array)
# # print(pb_array == subs.pb_array)
#subs.save_pb_array("pb_array.npy")
# subs.load_pb_array("pb_array.npy")
# print(subs.pb_array)