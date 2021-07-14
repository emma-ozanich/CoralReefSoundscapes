import numpy as np

Fs = 1000 # sensor sample rate
nfft = 256# 256 # nfft length
ovlap = 0.9 # nfft overlap
novlap = np.round(nfft*ovlap)
tbuffer = np.floor( 0.25*Fs ) # buffer, in samples, for spectrogram/azigram
bandlim = [100, 450]
dF = Fs/nfft
dT = (nfft-novlap)/Fs#(1-ovlap)*nfft/Fs
nF = np.ceil(np.diff(bandlim)/dF)
nF = int(nF[0]) # number of bandwidth frequency bins

maxlen = 0.5-tbuffer/Fs # maximum allowed length (from detector)
nL =  int(np.floor( (maxlen+tbuffer/Fs) / dT )) # maximum number of samples

dpath = '../../../Hawaii_data/DASARs/PB20gsif/PB20X0/'
spath = ''