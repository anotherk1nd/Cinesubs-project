# My implementation as method of movie wav extract - takes movie and extracts wav file of audio,

import subprocess
import os
import sys
import ffmpeg


#filename is a string, requires full file path of must be run from correct directory
def mywav_extract(fn,fn_out):
    fn_out_wav = fn_out + '.wav'

    # subprocess.call(['ffmpeg', '-i', fn, '-codec:a', 'pcm_s16le', '-ac', '1', fn_out_wav])
os.system(ffmpeg -i sample.avi -vn -ab 128 outputaudio.mp3)
#mywav_extract()


