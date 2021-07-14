def proc_data(fnames, tstart, tsample, ind, dur, Nevents):

    # check if gpus available
    import tensorflow as tf
    import os

    # import the package dependencies
    from glob import glob as gb #get filenames from path
    from readgsi import readgsi # read gsi file
    from calibrate_DASAR import calibrate_DASAR # calibrate gsi file
    from spect_dB import spect_dB # compute spectrograms
    from vector_direction import vector_direction # compute azigrams
    from directional_detector import directional_detector # azigram detector
    import numpy as np # number computation package
    import time # for calculating runtime
    import pandas as pd
    
    from scipy.signal import butter, lfilter

    import importlib

    # # # # # # # # # # #
    t00= time.time()

    from set_params import nF, nL, dT, dF, tbuffer, dpath, maxlen, Fs, nfft, ovlap, bandlim

    nT = np.floor(maxlen*Fs + tbuffer).astype(int)
    input_matrix = np.zeros((Nevents, nT, 3), dtype=float)
    # load X data
    tlen = 0
    for ii,fn in enumerate(fnames):
        print('Now processing ' + fn)
        omi, t, header = readgsi(fn, tstart[ii], tsample[ii], 'seconds')
        x = calibrate_DASAR(omi, 'DASARC') # able to do array or single-channel vector
        xbrefa = header['brefa']+8#

        loc_ind = ind[ii]
        loc_dur = dur[ii]
           
        td= time.time()
        for ev in range(len(loc_ind)):
           # print(ii)
            i0 = (loc_ind[ev] - tbuffer).astype(int) 
            i1 = (loc_ind[ev] + np.floor(loc_dur[ev]*Fs) + tbuffer).astype(int)
            i1 = (i0 + np.floor(maxlen*Fs+tbuffer)).astype(int)

            xindex = range(i0, i1)
            Wn = np.array((bandlim[0], bandlim[1]))/(0.5*Fs)
            b,a = butter(4, Wn, btype='band')

            input_matrix[ev + tlen, 0:len(xindex), 0] = lfilter(b,a,x[0,xindex]) # Z
            input_matrix[ev + tlen, 0:len(xindex), 1] = lfilter(b,a,x[1,xindex])  # Y
            input_matrix[ev + tlen, 0:len(xindex), 2] = lfilter(b,a,x[2,xindex])  # X

        tlen = len(loc_dur) + tlen
        print(time.time()-td)


    return input_matrix
    
    