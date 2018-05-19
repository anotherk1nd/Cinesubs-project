import pysrt as srt
import scipy as sp
from datetime import datetime, date

subs = srt.open('gotS07E01.srt') #THIS SUBSTRACK IS LONGER THAN THE FILM, SOME ADDITIONAL SUBS, PERHAPS FOR AN ALTERNATIVE VERSION!
#print(sp.shape(subs))#758 entries, 1 column I think
#print(type(subs[1])) # <class 'pysrt.srtitem.SubRipItem'>
#print(subs[0]) # subtitle index (from 1), start->stop time, text
'''
1
00:00:03,202 --> 00:00:08,327
♪ (PIANO PLAYING) ♪
'''
#fst = subs[1] #
#fst_start = subs[1].start.to_time() #this works
#print(fst_start)
#print(fst[1]) doesn't work
#print(fst.text) # ♪ (PIANO PLAYING) ♪
#print(fst.start.to_time()) #00:00:03.202000
#print(type(fst.start.to_time())) #<class 'datetime.time'>
#print(fst.end.to_time()) #00:00:08.327000
#begin = fst.start.to_time()
#print(begin)
#end = fst.end.to_time()
# The following 2 lines allow us to convert the duration into seconds
#duration = datetime.combine(date.min, end) - datetime.combine(date.min, begin)
#print(duration.total_seconds())
#The following 2 lines allow us to convert the times into seconds from start, should probably write this up into a definition https://stackoverflow.com/questions/5259882/subtract-two-times-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
#time_end = datetime.combine(date.min, end) - datetime.min
#print(time_end.total_seconds())
# I need to find a way to convert the times in fst to an array where each entry represents a sample of the mfcc and is tagged with 1 or 0 depending whether
# there are subtitles present or not
# I think I should look at each entry of the subtitles time and compare the time of each entry in the pb_array, and fill it this way


#We define a function that will take each subs entries which uses datatime objects that can't be handled easily, and convert to seconds from start, returning
#array [time_start,time_end]
def tconv(subs,i):
    fst_start = subs[i].start.to_time()
    time_start = datetime.combine(date.min, fst_start) - datetime.min
    time_start = time_start.total_seconds()
    fst_end = subs[i].end.to_time()
    time_end = datetime.combine(date.min, fst_end) - datetime.min
    time_end = time_end.total_seconds()
    times = [time_start,time_end]
    return times

last_sub = tconv(subs,-1)[1]
print(last_sub)
print(subs[-1])
pb_array = sp.zeros((int(last_sub),3))
count = sp.arange(int(last_sub))
pb_array[:,0] = count[:] # an index for each sample starting at 0
pb_array[:,1] = pb_array[:,0]*0.01 #0.01 is the size of the MFCC windows used in play.py, this is the starting point for each window

#print(pb_array)
#print(tconv(subs,1))
#print(len(subs))
sp.set_printoptions(threshold=sp.nan)
#print(subs)
#print(pb_array)

i=0
j=0 # i is counter over subs array, j is counter over sample array (stored in pb_array)
while True:
    if i +1 > len(pb_array)-1: #matrices are indexed from 0 but the length returns the 'normal' length i.e len([1]) returns 1 not 0
        print('substimes length exceeded')
        print(i)
        break
    if j > len(subs)-1:
        print('pb array length exceeded')
        print(j)
        break
    subs_times = tconv(subs,j) #substimes
    #print(subs_times)
    #print(pb_array[i,1])
    #print(pb_array[i + 1, 1])
    #end = subs[j].end.to_time()
    if pb_array[i,1]>= subs_times[0]:
        if pb_array[i + 1, 1]< subs_times[1]: #we only stored start times in pb_array, so need to compare to i+1
            print('Within subs window')
            pb_array[i,2] = 1
            i = i+1
        elif pb_array[i + 1, 1]>= subs_times[1]:
            print('Subs window exceeded')
            j = j+1
    else:
        i = i+1

#sp.savetxt('pb_array.csv',pb_array,delimiter=',')

#print(pb_array)
#print(i,j)
#print(tconv(subs,-1))

#print((end - datetime.time.min)).total_seconds()
#print(type(end)) # <class 'datetime.time'>
#print(subs[-1])

# print(end.total_seconds()) # doesnt work, 'datetime.time' object has no attribute 'total_seconds'

