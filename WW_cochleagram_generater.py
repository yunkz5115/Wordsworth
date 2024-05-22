# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:08:38 2024

@author: Yunkai Zhu
"""
import matplotlib.pyplot as plt
import numpy as np
import torch as ch
import chcochleagram
import os
from time import perf_counter
from scipy.io import wavfile
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#-----------------------------------------------------------------------------#
device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
print(device)
#-----------------------------------------------------------------------------#
### Args used for multiple stages of the cochleagram operations
signal_size = 36000 # Length of the input audio signal (currently must be fixed, due to filter construction)
sr = 24000 # Sampling rate of the input audio
pad_factor = 1.5 # Zero padding applied to the waveform, so the end signal is length pad_factor*signal_length
use_rfft = True # Whether to use rfft operations when appropriate (recommended)

### Define the cochlear filters using ERBCosFilters. 
# These are the arguments used for filter construction of ERBCosFilters. See helpers/erb_filters.py for 
# more documentation. 
half_cos_filter_kwargs = {
    'n':50, # Number of filters to evenly tile the space
    'low_lim':50, # Lowest center frequency for full filter (if lowpass filters are used they can be centered lower)
    'high_lim':8000, # Highest center frequency 
    'sample_factor':4, # Positive integer that determines how densely ERB function will be sampled
    'full_filter':False, # Whether to use the full-filter. Must be False if rFFT is true. 
}
# These arguments are for the CochFilters class (generic to any filters). 
coch_filter_kwargs = {'use_rfft':use_rfft,
                      'pad_factor':pad_factor,
                      'filter_kwargs':half_cos_filter_kwargs}

# This (and most) cochleagrams use ERBCosFilters, however other types of filterbanks can be 
# constructed for linear spaced filters or different shapes. Make a new CochlearFilter class for 
# these. 
filters = chcochleagram.cochlear_filters.ERBCosFilters(signal_size,
                                                       sr, 
                                                       **coch_filter_kwargs)

print('filter set up success.')
### Look at the filters 
# Filters are in form [filter_idx, frequency, (real_value, complex_value)] where for ERB filters 
# the complex component is 0. Newer versions of pytorch have complex types, but complex values are 
# represented with an extra dimension for now. 
for f_idx in range(filters.coch_filters.shape[0]):
    plt.plot(filters.filter_extras['freqs'], filters.coch_filters[f_idx,:,0]) 
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter Magnitude')
plt.close()
print('filter plot success.')
#-----------------------------------------------------------------------------#
### Define an envelope extraction operation
# Use the analytic amplitude of the hilbert transform here. Other types of envelope extraction 
# are also implemented in envelope_extraction.py. Can use Identity if want the raw subbands. 
envelope_extraction = chcochleagram.envelope_extraction.HilbertEnvelopeExtraction(signal_size,
                                                                                  sr, 
                                                                                  use_rfft, 
                                                                                  pad_factor)

### Define a downsampling operation
# Downsample the extracted envelopes. Can use Identity if want the raw subbands. 
env_sr = 200 # Sampling rate after downsampling
downsampling_kwargs = {'window_size':1001} # Parameters for the downsampling filter (see downsampling.py)
downsampling_op = chcochleagram.downsampling.SincWithKaiserWindow(sr, env_sr, **downsampling_kwargs)

### Define a compression operation.
compression_kwargs = {'power':0.3, # Power compression of 0.3 
                      'offset':1e-8, # Offset for numerical stability in backwards pass
                      'scale':1, # Optional multiplicative value applied to the envelopes before compression 
                      'clip_value':100} # Clip the gradients for this compression for stability
compression = chcochleagram.compression.ClippedGradPowerCompression(**compression_kwargs)
#-----------------------------------------------------------------------------#
#Define cochleagram
cochleagram = chcochleagram.cochleagram.Cochleagram(filters, 
                                                    envelope_extraction,
                                                    downsampling_op,
                                                    compression=compression)
#-----------------------------------------------------------------------------#
#Train set
filepath='root_path/DeepLearning_Superset/train/'
words = os.listdir(filepath)
for i in range(len(words)):
    t_start = perf_counter()
    name = os.listdir(''.join([filepath,words[i]]))
    kkk = 0;
    for j in range(len(name)):
        fs,audio = wavfile.read(''.join([filepath,words[i],'/',name[j]]))
        if len(audio)<signal_size:
            audio = np.append(audio,np.zeros(signal_size-len(audio)))
        elif len(audio)>signal_size:
            audio = np.delete(audio,np.arange(signal_size,len(audio)),0)
        x = ch.autograd.Variable(ch.Tensor(audio), requires_grad=True)
        y = cochleagram(x)
        
        path = ''.join(['./DL_Set/NS_Chcochleagram/train/',words[i]])
        os.makedirs(path, exist_ok=True)
        np.savetxt(''.join([path,'/',name[j][0:(len(name[j])-4)],'.txt'])
                   ,np.squeeze(y.detach().numpy()))
        
        if j<3:
            # Plot the cochleagram
            path = ''.join(['./DL_Set/NS_Chcochleagram_figure/train/',words[i]])
            os.makedirs(path, exist_ok=True)
            fig1=plt.figure()
            plt.imshow(np.squeeze(y.detach().numpy()), origin='lower', extent=(0, y.shape[1], 0, y.shape[0]))
            
            ## Depending on the temporal padding the cochleagram length may not be exactly equal env_sr*signal_size/sr
            # Because of this, set the x-axis tick labels based on the original audio. 
            num_ticks = 9
            x_tick_numbers = [t_num*y.shape[-1]/(num_ticks-1) for t_num in range(num_ticks)]
            x_tick_labels = [t_num*signal_size/sr/(num_ticks-1) for t_num in range(num_ticks)]
            plt.xticks(x_tick_numbers, x_tick_labels)
            plt.xlabel('Time (s)')
            
            ## Label the frequency axis based on the center frequencies for the ERB filters. 
            filters.filter_extras['cf']
            # Use ticks starting at the lowest non-lowpass filter center frequency. 
            y_ticks = [y_t+3 for y_t in plt.yticks()[0] if y_t<=y.shape[0]]
            plt.yticks(y_ticks, [int(round(filters.filter_extras['cf'][int(f_num)])) for f_num in y_ticks])
            plt.ylabel('Frequency (Hz)')
            plt.title(words[i])
            t_end = perf_counter()
            fig1.savefig(''.join([path,'/',name[j][0:(len(name[j])-4)],'.svg']),dpi=600)
            plt.close()
            
        kkk = kkk + 1
        if kkk>(len(name)/80):
            kkk = 0
            print ('=>',end="")
        t_end = perf_counter()
    
    print(' duration = ',np.around(t_end-t_start,3),' s ')
    
print(' NS Train set finished. ')
#-----------------------------------------------------------------------------#
#Test set
filepath='root_path/DeepLearning_Superset/test/'
words = os.listdir(filepath)
for i in range(len(words)):
    t_start = perf_counter()
    name = os.listdir(''.join([filepath,words[i]]))
    kkk = 0;
    for j in range(len(name)):
        fs,audio = wavfile.read(''.join([filepath,words[i],'/',name[j]]))
        if len(audio)<signal_size:
            audio = np.append(audio,np.zeros(signal_size-len(audio)))
        elif len(audio)>signal_size:
            audio = np.delete(audio,np.arange(signal_size,len(audio)),0)
        x = ch.autograd.Variable(ch.Tensor(audio), requires_grad=True)
        y = cochleagram(x)
        
        path = ''.join(['./DL_Set/NS_Chcochleagram/test/',words[i]])
        os.makedirs(path, exist_ok=True)
        np.savetxt(''.join([path,'/',name[j][0:(len(name[j])-4)],'.txt'])
                   ,np.squeeze(y.detach().numpy()))
        
        if j<3:
            # Plot the cochleagram
            path = ''.join(['./DL_Set/NS_Chcochleagram_figure/test/',words[i]])
            os.makedirs(path, exist_ok=True)
            fig1=plt.figure()
            plt.imshow(np.squeeze(y.detach().numpy()), origin='lower', extent=(0, y.shape[1], 0, y.shape[0]))
            
            ## Depending on the temporal padding the cochleagram length may not be exactly equal env_sr*signal_size/sr
            # Because of this, set the x-axis tick labels based on the original audio. 
            num_ticks = 9
            x_tick_numbers = [t_num*y.shape[-1]/(num_ticks-1) for t_num in range(num_ticks)]
            x_tick_labels = [t_num*signal_size/sr/(num_ticks-1) for t_num in range(num_ticks)]
            plt.xticks(x_tick_numbers, x_tick_labels)
            plt.xlabel('Time (s)')
            
            ## Label the frequency axis based on the center frequencies for the ERB filters. 
            filters.filter_extras['cf']
            # Use ticks starting at the lowest non-lowpass filter center frequency. 
            y_ticks = [y_t+3 for y_t in plt.yticks()[0] if y_t<=y.shape[0]]
            plt.yticks(y_ticks, [int(round(filters.filter_extras['cf'][int(f_num)])) for f_num in y_ticks])
            plt.ylabel('Frequency (Hz)')
            plt.title(words[i])
            t_end = perf_counter()
            fig1.savefig(''.join([path,'/',name[j][0:(len(name[j])-4)],'.svg']),dpi=600)
            plt.close()
            
        kkk = kkk + 1
        if kkk>(len(name)/80):
            kkk = 0
            print ('=>',end="")
        t_end = perf_counter()
    
    print(' duration = ',np.around(t_end-t_start,3),' s ')
    
print(' NS Test set finished. ')

#-----------------------------------------------------------------------------#
#Full set
filepath='root_path/Wordsworth_v1.0/'
words = os.listdir(filepath)
for i in range(len(words)):
    t_start = perf_counter()
    name = os.listdir(''.join([filepath,words[i]]))
    kkk = 0;
    for j in range(len(name)):
        fs,audio = wavfile.read(''.join([filepath,words[i],'/',name[j]]))
        if len(audio)<signal_size:
            audio = np.append(audio,np.zeros(signal_size-len(audio)))
        elif len(audio)>signal_size:
            audio = np.delete(audio,np.arange(signal_size,len(audio)),0)
        x = ch.autograd.Variable(ch.Tensor(audio), requires_grad=True)
        y = cochleagram(x)
        
        path = ''.join(['./DL_Set/NS_Chcochleagram/Full/',words[i]])
        os.makedirs(path, exist_ok=True)
        np.savetxt(''.join([path,'/',name[j][0:(len(name[j])-4)],'.txt'])
                   ,np.squeeze(y.detach().numpy()))
        
        if j<3:
            # Plot the cochleagram
            path = ''.join(['./DL_Set/NS_Chcochleagram_figure/Full/',words[i]])
            os.makedirs(path, exist_ok=True)
            fig1=plt.figure()
            plt.imshow(np.squeeze(y.detach().numpy()), origin='lower', extent=(0, y.shape[1], 0, y.shape[0]))
            
            ## Depending on the temporal padding the cochleagram length may not be exactly equal env_sr*signal_size/sr
            # Because of this, set the x-axis tick labels based on the original audio. 
            num_ticks = 9
            x_tick_numbers = [t_num*y.shape[-1]/(num_ticks-1) for t_num in range(num_ticks)]
            x_tick_labels = [t_num*signal_size/sr/(num_ticks-1) for t_num in range(num_ticks)]
            plt.xticks(x_tick_numbers, x_tick_labels)
            plt.xlabel('Time (s)')
            
            ## Label the frequency axis based on the center frequencies for the ERB filters. 
            filters.filter_extras['cf']
            # Use ticks starting at the lowest non-lowpass filter center frequency. 
            y_ticks = [y_t+3 for y_t in plt.yticks()[0] if y_t<=y.shape[0]]
            plt.yticks(y_ticks, [int(round(filters.filter_extras['cf'][int(f_num)])) for f_num in y_ticks])
            plt.ylabel('Frequency (Hz)')
            plt.title(words[i])
            t_end = perf_counter()
            fig1.savefig(''.join([path,'/',name[j][0:(len(name[j])-4)],'.svg']),dpi=600)
            plt.close()
            
        kkk = kkk + 1
        if kkk>(len(name)/80):
            kkk = 0
            print ('=>',end="")
        t_end = perf_counter()
    
    print(' duration = ',np.around(t_end-t_start,3),' s ')
    
print(' NS Full set finished. ')
