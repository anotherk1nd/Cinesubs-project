import pysrt as srt
import scipy as sp
from datetime import datetime, date

subs = srt.open('gotS07E01.srt')
#print(sp.shape(subs))#758 entries, 1 column I think
#print(type(subs[1])) # <class 'pysrt.srtitem.SubRipItem'>
#print(subs[0]) # subtitle index (from 1), start->stop time, text
'''
1
00:00:03,202 --> 00:00:08,327
♪ (PIANO PLAYING) ♪
'''
fst = subs[0] #
#print(fst[1]) doesn't work
#print(fst.text) # ♪ (PIANO PLAYING) ♪
#print(fst.start.to_time()) #00:00:03.202000
#print(type(fst.start.to_time())) #<class 'datetime.time'>
#print(fst.end.to_time()) #00:00:08.327000
begin = fst.start.to_time()
end = fst.end.to_time()
# The following 2 lines allow us to convert the duration into seconds
duration = datetime.combine(date.min, end) - datetime.combine(date.min, begin)
print(duration.total_seconds())
#The following 2 lines allow us to convert the times into seconds from start, should probably write this up into a definition https://stackoverflow.com/questions/5259882/subtract-two-times-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
time_end = datetime.combine(date.min, end) - datetime.min
print(time_end.total_seconds())
#print((end - datetime.time.min)).total_seconds()
#print(type(end)) # <class 'datetime.time'>
#print(subs[-1])

# print(end.total_seconds()) # doesnt work, 'datetime.time' object has no attribute 'total_seconds'

