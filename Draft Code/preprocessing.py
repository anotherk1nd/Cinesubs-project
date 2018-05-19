class preprocessing:
    '''
    Here we define the different methods for all the preprocessing required to produce the inputs and outputs for the learning process
    '''

    #We define the class variables that are shared across the functions. However we need to use a function to extract these variables, and hence maybe need to use
    #the concept of inheritance, where a class is declared subsequent to a previous class and takes the outputs of the previous class as initialisers of new class (or somethin like that!)
    # filename is a string, requires full file path of must be run from correct directory

    def mywav_extract(fn, fn_out): #THIS IS INCOMPLETE
        fn_out_wav = fn_out + '.wav'
        os.system(ffmpeg -i sample.avi -vn -ab 128 outputaudio.mp3)

    def __init__(self):
        self.subs = subs
        self.pb_array = pb_array
        self.subs_times = subs_times
        self.count = count