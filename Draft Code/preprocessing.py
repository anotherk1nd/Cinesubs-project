class preprocessing:
    '''
    Here we define the different methods for all the preprocessing required to produce the inputs and outputs for the learning process
    '''

    #We define the class variables that are shared across the functions. However we need to use a function to extract these variables, and hence maybe need to use
    #the concept of inheritance, where a class is declared subsequent to a previous class and takes the outputs of the previous class as initialisers of new class (or somethin like that!)
    # filename is a string, requires full file path of must be run from correct directory

    def __init__(self,subsfn,pb_array,subs_times,count,counter): # we take the variables we want to initialise, then assign these as class attributes with the self.variable statements
        self.subsfn = subsfn
        self.pb_array = []
        self.subs_times = subs_times
        self.count = count
        self.counter = counter
        self.subs = 0
        self.shit = 0

    def open_subs(self):
        # Opens the srt file,
        self.subs = (srt.open('%s') % (self.subsfn))

    def tconv(self,counter):
        # We define a function that will take each subs entries which uses datatime objects that can't be handled easily, and convert to seconds from start, returning
        # array [time_start,time_end].
        # We define the counter externally so that it can be applied in the while true loop
        fst_start = self.subs[counter].start.to_time()
        time_start = datetime.combine(date.min, fst_start) - datetime.min
        time_start = time_start.total_seconds()
        fst_end = self.subs[counter].end.to_time()
        time_end = datetime.combine(date.min, fst_end) - datetime.min
        time_end = time_end.total_seconds()
        times = [time_start,time_end]
        return times

    #def myfunction(self,subfn):

    def pb_array_init(self):
        self.last_sub = tconv(self.subs, -1)[1]
        pb_array = sp.zeros((int(last_sub), 3))
        count = sp.arange(int(last_sub))
        pb_array[:, 0] = count[:]  # an index for each sample starting at 0
        pb_array[:, 1] = pb_array[:,0] * 0.01  # 0.01 is the size of the MFCC windows used in play.py, this is the starting point for each window
        self.pb_array = pb_array

