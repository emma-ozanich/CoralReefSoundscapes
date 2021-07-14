def refine_stft_detections(det, subband, subbandovlap, band, F, detspersubband, seplen, maxt):
    
    import numpy as np
    
    # convert freq. domain limits to array indices
    ndet = det.shape
    If0 = np.argmin(np.abs( F-band[0] ))
    If1 = np.argmin(np.abs( F-band[1] ))
    Fband = F[If0:If1]
    Nf = len(Fband)
    Isubband = np.argmin(np.abs(F - (F[0]+subband)))
    Iovlap = np.argmin(np.abs(F-(F[0]+(subband-subbandovlap))))
    Nbands = np.floor((Nf-Isubband)/(Iovlap))+1
    
    if detspersubband<1:
        detspersubband = np.floor(detspersubband*Isubband).astype(int) # if input as a fraction of total samples per subband
    det = det[If0:If1,:] # subset within desired freq band
    bandlims = [(np.arange(0, Isubband)+i*Iovlap).astype(int) for i in np.arange(Nbands)]
    Events = np.zeros((ndet)) # map back into original size
    t=0
    Edet = False
    for j in bandlims:
        detsub = det[j,:]
        detseq = np.sum( detsub, axis=0 )
        detseq[detseq<detspersubband] = 0 # if it doesn't meet the detection limit
        detseq[detseq>=detspersubband] = 1 # else 
        
        # merge close events
        Ievent = np.where(detseq) # where events
        Idiff = np.diff(Ievent[0]) # event separation
        Iclose = np.where(Idiff<seplen) # where separation small
        for i in range(len(Iclose[0])):
            fillevent = np.arange(Ievent[0][Iclose[0][i]], Ievent[0][Iclose[0][i]+1]) # indices to fill
            detseq[fillevent] = 1#np.ones((len(fillevent),))
      #  print(det[j,detseq.astype(bool)].shape)
        Events[j+If0,:] = np.tile(detseq, (len(j),1))#det[j,detseq.astype(bool)] # fill binary detection matrix
        if np.sum(detseq)>1:
            Edet = True # was there a detection?
        t=t+1
    
#    if Edet:
#        print('At least one event detected.')
    
    TotalEvents = np.sum(Events,axis=0)
    
    return Events, TotalEvents