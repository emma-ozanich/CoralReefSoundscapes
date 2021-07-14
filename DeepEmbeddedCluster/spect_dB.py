def spect_dB(x, Fs, nfft, ovlap):
    
    # assumes no internal averaging, for now
    
    import numpy as np
    import scipy.signal as sp
    
    f, t, Sxx = sp.spectrogram(x, fs=Fs, nperseg=nfft, nfft=nfft, noverlap=np.round(ovlap*nfft), window='hamming',mode='complex',detrend=False)
    
    rescale = (np.sqrt(Fs)*np.linalg.norm(np.hamming(nfft))) # rescale for equivalency to Matlab
    Sxx = rescale*Sxx # rescale for equivalency to Matlab
    
    SdB = 4 + 10*np.log10(2*np.abs(Sxx)**2/(nfft*Fs)) # One-sided power spectra, \muPa^2/Hz. unsure why 4 for scaling?
    
 
    return SdB, Sxx, f, t
