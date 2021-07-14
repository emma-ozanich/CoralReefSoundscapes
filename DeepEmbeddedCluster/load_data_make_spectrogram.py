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
    
    import math
    
    import matplotlib.pyplot as plt
    from scipy.stats import vonmises
    import scipy.signal as sp

    import importlib

    # # # # # # # # # # #
    t00= time.time()

    from set_params import nF, nL, dT, dF, tbuffer, dpath, maxlen, Fs, nfft, ovlap, bandlim

    input_matrix = np.zeros((Nevents, nF, nL+1, 3), dtype=float)
    # load X data
    tlen = 0
    for ii,fn in enumerate(fnames):
        print('Now processing ' + fn)
        omi, t, header = readgsi(fn, tstart[ii], tsample[ii], 'seconds')
        x = calibrate_DASAR(omi, 'DASARC') # able to do array or single-channel vector
        xbrefa = header['brefa']+8

        loc_ind = ind[ii]
        loc_dur = dur[ii]
           
        td= time.time()
        for ev in range(len(loc_ind)):
           # print(ii)
            i0 = (loc_ind[ev] - tbuffer).astype(int) 
            i1 = (loc_ind[ev] + np.floor(loc_dur[ev]*Fs) + tbuffer).astype(int)
            i1 = (i0 + np.floor(2*maxlen*Fs+tbuffer)).astype(int)

            xindex = range(i0, i1)

            # find X spectra
            Sx1dB, S1x, F, t = spect_dB( x[0,xindex], header['Fs'], nfft, ovlap )
            Sx2dB, S2x, F, t = spect_dB( x[1,xindex], header['Fs'], nfft, ovlap )
            Sx3dB, S3x, F, t = spect_dB( x[2,xindex], header['Fs'], nfft, ovlap )

            adx, rdx = vector_direction(S1x, S2x, S3x, xbrefa, F, [300])

            mxl = np.minimum(nL+1, adx.shape[1])

            iF = (F>= bandlim[0])*(F<=bandlim[1])
            
            vec_adx = adx[iF,0:mxl].flatten()
            vec_adx = vec_adx*np.pi/180 # convert to radians
            vec_adx[vec_adx>np.pi] = vec_adx[vec_adx>np.pi]-2*np.pi
            vec_adx[vec_adx<-np.pi] = vec_adx[vec_adx<-np.pi]+2*np.pi
          #  hist, bin_edges = np.histogram(vec_adx, density=True, bins=180)
          #  kappa, mu, fscale = vonmises.fit(vec_adx, fscale=1)
          #  mu = bin_edges[hist.argmax()]
           # print(kappa, mu)
            
            mu = 92
            mu = math.radians(mu)
            if mu>np.pi:
                mu = mu-2*np.pi
            if mu<-np.pi:
                mu = 2*np.pi+mu
            kappa=0.04
            kappa=3
           # print(adx.shape)
         #   x = np.linspace(vonmises.ppf(0.01, kappa,loc=mu),
         #       vonmises.ppf(0.99, kappa,loc=mu), 100)
         #   plt.plot(bin_edges[0:-1],hist/hist.max())
          #  vpdf = vonmises.pdf(vec_adx, kappa, loc=mu)
            #tmppdf = np.concatenate((vpdf[x>np.pi], vpdf[x<=np.pi and x>=-np.pi], vpdf[x<-np.pi]))
            #vpdf = tmppdf
           # I = np.argsort(vec_adx)
           # plt.plot(vec_adx[I]*180/np.pi , vpdf[I]/vpdf.max())
        #    plt.ylim((0.15, 0.17))
           # plt.show()
            
          #  vpdf = np.reshape(vpdf, np.shape(Sx1dB[iF,0:mxl]))
            input_matrix[ev + tlen, 0:len(iF[iF>0]), 0:mxl, 0] = Sx1dB[iF,0:mxl] # X
         #   input_matrix[ev + tlen, 0:len(iF[iF>0]), 0:mxl, 0] = sp.medfilt2d(np.multiply(Sx1dB[iF,0:mxl], vpdf),kernel_size=(7,3)) # X
            input_matrix[ev + tlen, 0:len(iF[iF>0]), 0:mxl, 1] = Sx2dB[iF,0:mxl] # X
            input_matrix[ev + tlen, 0:len(iF[iF>0]), 0:mxl, 2] = Sx3dB[iF,0:mxl] # X
            
          #  plt.imshow(Sx1dB[iF,0:mxl])
          #  plt.axis('scaled')
          #  plt.colorbar()
          #  plt.show()
          #  plt.imshow(input_matrix[ev+tlen, 0:len(iF[iF>0]), 0:mxl, 0])
          #  plt.axis('scaled')
          #  plt.colorbar()
          #  plt.show()

        tlen = len(loc_dur) + tlen
        print(time.time()-td)


    return input_matrix
    
    